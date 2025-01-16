import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import math
from thop import clever_format, profile


# Feature Rectify Module
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 6, self.dim * 6 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 6 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        std = torch.std(x, dim=(2, 3), keepdim=True).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, std, max), dim=1)  # B 6C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights





# 先空间校正再通道校正
class FeatureCorrection_s2c(nn.Module):
    def __init__(self, dim, reduction=1, eps=1e-8):
        super(FeatureCorrection_s2c, self).__init__()
        # 自定义可训练权重参数
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)

        self.apply(self._init_weights)

    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)

        spatial_weights = self.spatial_weights(x1, x2)
        x1_1 = x1 + fuse_weights[0] * spatial_weights[1] * x2
        x2_1 = x2 + fuse_weights[0] * spatial_weights[0] * x1

        channel_weights = self.channel_weights(x1_1, x2_1)

        main_out = x1_1 + fuse_weights[1] * channel_weights[1] * x2_1
        aux_out = x2_1 + fuse_weights[1] * channel_weights[0] * x1_1
        return main_out, aux_out


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr1 = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.norm1 = nn.LayerNorm(dim)

            self.sr2 = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.norm2 = nn.LayerNorm(dim)

    def forward(self, x1, x2, H, W):
        B, N, C = x1.shape
        # B num_heads N C//num_heads
        q1 = self.q1(x1).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = self.q2(x2).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            # B C//num_head N num_heads -> B N C//num_heads num_heads -> B C H W
            x_1 = x1.permute(0, 2, 1).reshape(B, C, H, W)
            # B C H W -> B C H/R W/R -> B C HW/R² -> B HW/R² C
            x_1 = self.sr1(x_1).reshape(B, C, -1).permute(0, 2, 1)
            x_1 = self.norm1(x_1)
            # B HW/R² C -> B HW/R² 2C -> B HW/R² 2 num_heads C//num_heads -> 2 B num_heads HW/R² C//num_heads
            kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            x_2 = x2.permute(0, 2, 1).reshape(B, C, H, W)
            x_2 = self.sr2(x_2).reshape(B, C, -1).permute(0, 2, 1)
            x_2 = self.norm2(x_2)
            kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            kv2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # B num_heads HW/R² C//num_heads
        k1, v1 = kv1[0], kv1[1]
        k2, v2 = kv2[0], kv2[1]

        # B num_heads N HW/R²
        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)

        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)

        # B num_heads N C//num_heads -> B N num_heads C//num_heads -> B N C
        main_out = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)
        aux_out = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)

        return main_out, aux_out


class FeatureInteraction(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, sr_ratio=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads, sr_ratio=sr_ratio)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2, H, W):
        y1, z1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, z2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        c1, c2 = self.cross_attn(z1, z2, H, W)
        y1 = torch.cat((y1, c1), dim=-1)
        y2 = torch.cat((y2, c2), dim=-1)
        main_out = self.norm1(x1 + self.end_proj1(y1))
        aux_out = self.norm2(x2 + self.end_proj2(y2))

        return main_out, aux_out


class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)

        return out


class ChannelEmbed2(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed2, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

        self.channel_embed1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=out_channels),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        self.channel_embed2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=out_channels),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x1, x2, H, W):
        B, N, _C = x1.shape

        x1 = x1.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        x2 = x2.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x1)

        x1 = self.channel_embed1(x1)
        x2 = self.channel_embed2(x2)
        fuse = self.norm(x1 * x2)

        out = fuse + residual

        return out


class FeatureFusion(nn.Module):
    def __init__(self, dim, reduction=1, sr_ratio=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = FeatureInteraction(dim=dim, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)
        self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction,
                                        norm_layer=norm_layer)
        self.apply(self._init_weights)

    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        # B C (HW)->B N(HW) C
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        # B N(HW) C
        x1, x2 = self.cross(x1, x2, H, W)
        # B N(HW) 2C
        fuse = torch.cat((x1, x2), dim=-1)
        # B C H W
        fuse = self.channel_emb(fuse, H, W)

        return fuse


class FeatureFusion2(nn.Module):
    def __init__(self, dim, reduction=1, sr_ratio=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = FeatureInteraction(dim=dim, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)
        self.channel_emb = ChannelEmbed2(in_channels=dim, out_channels=dim, norm_layer=norm_layer)
        self.apply(self._init_weights)

    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        # B C (HW)->B N(HW) C
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        # B N(HW) C
        x1, x2 = self.cross(x1, x2, H, W)
        # B C H W
        fuse = self.channel_emb(x1, x2, H, W)

        return fuse


if __name__ == '__main__':
    model = FeatureFusion(64, num_heads=1, sr_ratio=4).cuda()

    left = torch.randn(1, 64, 64, 64).cuda()
    right = torch.randn(1, 64, 64, 64).cuda()
    out = model(left, right)

    # summary(model, [(4, 256, 256), (1, 256, 256)])
    flops, params = profile(model, (left, right), verbose=False)

    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % flops)
    print('Total params: %s' % params)
