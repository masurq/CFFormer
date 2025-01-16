import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

import math
import time
from collections import OrderedDict

from thop import clever_format, profile
from models.new_model.modules import FeatureFusion as FFM
from models.new_model.modules import FeatureCorrection_s2c as FCM


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        # B, C//3, H, W
        B, d, H, W = q.shape
        q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
        k = self.unfold(k).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 2,
                                                                                                        3)  # B,h,N,d,k*k
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3,
                                                                                                        2)  # B,h,N,k*k,d
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class MultiDilatelocalAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        # num_dilation,3,B,C//num_dilation,H,W
        x = x.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2)
        # num_dilation, B, H, W, C//num_dilation
        for i in range(self.num_dilation):
            x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])  # B, H, W,C//num_dilation
        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DilateBlock(nn.Module):
    "Implementation of Dilate-attention block"

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, dilation=[1, 2, 3],
                 cpe_per_block=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.cpe_per_block = cpe_per_block
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = MultiDilatelocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        if self.cpe_per_block:
            x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)
        # B, C, H, W
        return x


class GlobalAttention(nn.Module):
    "Implementation of self-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalBlock(nn.Module):
    """
    Implementation of Transformer
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cpe_per_block=False):
        super().__init__()
        self.cpe_per_block = cpe_per_block
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = GlobalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale, attn_drop=attn_drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        if self.cpe_per_block:
            x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.
    """

    def __init__(self, img_size=224, in_chans=3, hidden_dim=16,
                 patch_size=4, embed_dim=96, patch_way=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.img_size = img_size
        assert patch_way in ['overlaping', 'nonoverlaping', 'pointconv'], \
            "the patch embedding way isn't exist!"
        if patch_way == "nonoverlaping":
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        elif patch_way == "overlaping":
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=3, stride=1,
                          padding=1, bias=False),  # 224x224
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, int(hidden_dim * 2), kernel_size=3, stride=2,
                          padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(int(hidden_dim * 2)),
                nn.GELU(),
                nn.Conv2d(int(hidden_dim * 2), int(hidden_dim * 4), kernel_size=3, stride=1,
                          padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(int(hidden_dim * 4)),
                nn.GELU(),
                nn.Conv2d(int(hidden_dim * 4), embed_dim, kernel_size=3, stride=2,
                          padding=1, bias=False),  # 56x56
            )
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=3, stride=2,
                          padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, int(hidden_dim * 2), kernel_size=1, stride=1,
                          padding=0, bias=False),  # 112x112
                nn.BatchNorm2d(int(hidden_dim * 2)),
                nn.GELU(),
                nn.Conv2d(int(hidden_dim * 2), int(hidden_dim * 4), kernel_size=3, stride=2,
                          padding=1, bias=False),  # 56x56
                nn.BatchNorm2d(int(hidden_dim * 4)),
                nn.GELU(),
                nn.Conv2d(int(hidden_dim * 4), embed_dim, kernel_size=1, stride=1,
                          padding=0, bias=False),  # 56x56
            )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  # B, C, H, W
        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """

    def __init__(self, in_channels, out_channels, merging_way, cpe_per_satge, norm_layer=nn.BatchNorm2d):
        super().__init__()
        assert merging_way in ['conv3_2', 'conv2_2', 'avgpool3_2', 'avgpool2_2'], \
            "the merging way is not exist!"
        self.cpe_per_satge = cpe_per_satge
        if merging_way == 'conv3_2':
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                norm_layer(out_channels),
            )
        elif merging_way == 'conv2_2':
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                norm_layer(out_channels),
            )
        elif merging_way == 'avgpool3_2':
            self.proj = nn.Sequential(
                nn.AvgPool2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                norm_layer(out_channels),
            )
        else:
            self.proj = nn.Sequential(
                nn.AvgPool2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                norm_layer(out_channels),
            )
        if self.cpe_per_satge:
            self.pos_embed = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels)

    def forward(self, x):
        # x: B, C, H ,W
        x = self.proj(x)
        if self.cpe_per_satge:
            x = x + self.pos_embed(x)
        return x


class Dilatestage(nn.Module):
    """ A basic Dilate Transformer layer for one stage.
    """

    def __init__(self, dim, depth, num_heads, kernel_size, dilation,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, cpe_per_satge=False, cpe_per_block=False,
                 downsample=True, merging_way=None):
        super().__init__()
        # build blocks
        self.blocks = nn.ModuleList([
            DilateBlock(dim=dim, num_heads=num_heads,
                        kernel_size=kernel_size, dilation=dilation,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer, act_layer=act_layer, cpe_per_block=cpe_per_block)
            for i in range(depth)])

        # patch merging layer
        self.downsample = PatchMerging(dim, int(dim * 2), merging_way, cpe_per_satge) if downsample else nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        return x


class Globalstage(nn.Module):
    """ A basic Transformer layer for one stage."""

    def __init__(self, dim, depth, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cpe_per_satge=False, cpe_per_block=False,
                 downsample=True, merging_way=None):
        super().__init__()
        # build blocks
        self.blocks = nn.ModuleList([
            GlobalBlock(dim=dim, num_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer, act_layer=act_layer, cpe_per_block=cpe_per_block)
            for i in range(depth)])

        # patch merging layer
        self.downsample = PatchMerging(dim, int(dim * 2), merging_way, cpe_per_satge) if downsample else nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        return x


class DualDilateformer(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96, sr_ratios=[8, 4, 2, 1],
                 norm_fuse=nn.BatchNorm2d, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], kernel_size=3,
                 dilation=[1, 2, 3], mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.1,
                 reduction=[1, 1, 1, 1], norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 merging_way='conv3_2',
                 patch_way='overlaping',
                 dilate_attention=[True, True, False, False],
                 downsamples=[True, True, True, False],
                 cpe_per_satge=False, cpe_per_block=True):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim, patch_way=patch_way)
        self.aux_patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                          in_chans=in_chans, embed_dim=embed_dim, patch_way=patch_way)
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.stages = nn.ModuleList()
        self.aux_stages = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.aux_norm = nn.ModuleList()

        self.FCMs = nn.ModuleList()
        self.FFMs = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if dilate_attention[i_layer]:
                stage = Dilatestage(dim=int(embed_dim * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    kernel_size=kernel_size,
                                    dilation=dilation,
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    downsample=downsamples[i_layer],
                                    cpe_per_block=cpe_per_block,
                                    cpe_per_satge=cpe_per_satge,
                                    merging_way=merging_way
                                    )
                self.norm.append(
                    norm_layer(int(2 * embed_dim * 2 ** i_layer) if i_layer != 3 else int(embed_dim * 2 ** i_layer)))
                aux_stage = Dilatestage(dim=int(embed_dim * 2 ** i_layer),
                                        depth=depths[i_layer],
                                        num_heads=num_heads[i_layer],
                                        kernel_size=kernel_size,
                                        dilation=dilation,
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop, attn_drop=attn_drop,
                                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                        norm_layer=norm_layer,
                                        downsample=downsamples[i_layer],
                                        cpe_per_block=cpe_per_block,
                                        cpe_per_satge=cpe_per_satge,
                                        merging_way=merging_way
                                        )
                self.aux_norm.append(
                    norm_layer(int(2 * embed_dim * 2 ** i_layer) if i_layer != 3 else int(embed_dim * 2 ** i_layer)))
            else:
                stage = Globalstage(dim=int(embed_dim * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    downsample=downsamples[i_layer],
                                    cpe_per_block=cpe_per_block,
                                    cpe_per_satge=cpe_per_satge,
                                    merging_way=merging_way
                                    )
                self.norm.append(
                    norm_layer(int(2 * embed_dim * 2 ** i_layer) if i_layer != 3 else int(embed_dim * 2 ** i_layer)))
                aux_stage = Globalstage(dim=int(embed_dim * 2 ** i_layer),
                                        depth=depths[i_layer],
                                        num_heads=num_heads[i_layer],
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop, attn_drop=attn_drop,
                                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                        norm_layer=norm_layer,
                                        downsample=downsamples[i_layer],
                                        cpe_per_block=cpe_per_block,
                                        cpe_per_satge=cpe_per_satge,
                                        merging_way=merging_way
                                        )
                self.aux_norm.append(
                    norm_layer(int(2 * embed_dim * 2 ** i_layer) if i_layer != 3 else int(embed_dim * 2 ** i_layer)))
            self.stages.append(stage)
            self.aux_stages.append(aux_stage)
        self.FCMs = nn.ModuleList([
            FCM(dim=int(2 * embed_dim), reduction=reduction[0]),
            FCM(dim=int(2 * embed_dim * 2), reduction=reduction[1]),
            FCM(dim=int(2 * embed_dim * 4), reduction=reduction[2]),
            FCM(dim=int(embed_dim * 8), reduction=reduction[3])])

        self.FFMs = nn.ModuleList([
            FFM(dim=int(2 * embed_dim), reduction=reduction[0], num_heads=num_heads[0], norm_layer=norm_fuse,
                sr_ratio=sr_ratios[0]),
            FFM(dim=int(2 * embed_dim * 2), reduction=reduction[1], num_heads=num_heads[1], norm_layer=norm_fuse,
                sr_ratio=sr_ratios[1]),
            FFM(dim=int(2 * embed_dim * 4), reduction=reduction[2], num_heads=num_heads[2], norm_layer=norm_fuse,
                sr_ratio=sr_ratios[2]),
            FFM(dim=int(embed_dim * 8), reduction=reduction[3], num_heads=num_heads[3], norm_layer=norm_fuse,
                sr_ratio=sr_ratios[3])])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):

        if isinstance(pretrained, str):
            load_dualpath_model(self, pretrained)
        else:
            raise TypeError('pretrained must be a str or None')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def forward_features(self, x1, x2):

        outs = []
        x1 = self.patch_embed(x1)
        x2 = self.aux_patch_embed(x2)
        for i in range(4):
            x1 = self.stages[i](x1)
            x2 = self.aux_stages[i](x2)

            x1, x2 = self.FCMs[i](x1, x2)

            x1_1 = self.norm[i](x1.permute(0, 2, 3, 1).contiguous())
            x2_1 = self.aux_norm[i](x2.permute(0, 2, 3, 1).contiguous())

            fuse = self.FFMs[i](x1_1.permute(0, 3, 1, 2).contiguous(), x2_1.permute(0, 3, 1, 2).contiguous())
            outs.append(fuse)

        return outs

    def forward(self, x1, x2):
        x = self.forward_features(x1, x2)
        return x


def load_dualpath_model(model, model_file, is_restore=False):
    # load raw state_dict
    t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        # raw_state_dict = torch.load(model_file)
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file

    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('patch_embed') >= 0:
            state_dict[k] = v
            state_dict[k.replace('patch_embed', 'aux_patch_embed')] = v
        elif k.find('stages') >= 0:
            state_dict[k] = v
            state_dict[k.replace('stages', 'aux_stages')] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v
            state_dict[k.replace('norm', 'aux_norm')] = v

    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)

    del state_dict
    t_end = time.time()
    print("Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
        t_ioend - t_start, t_end - t_ioend))


class dilateformer_t(DualDilateformer):
    def __init__(self, **kwargs):
        super(dilateformer_t, self).__init__(img_size=256, depths=[2, 2, 6, 2], embed_dim=72, num_heads=[3, 6, 12, 24],
                                             reduction=[2, 2, 2, 1], **kwargs)


class dilateformer_s(DualDilateformer):
    def __init__(self, **kwargs):
        super(dilateformer_s, self).__init__(img_size=256, depths=[3, 5, 8, 3], embed_dim=72, num_heads=[3, 6, 12, 24],
                                             reduction=[2, 2, 2, 1], **kwargs)


class dilateformer_b(DualDilateformer):
    def __init__(self, **kwargs):
        super(dilateformer_b, self).__init__(img_size=256, depths=[4, 8, 10, 3], embed_dim=96, num_heads=[3, 6, 12, 24],
                                             reduction=[2, 2, 2, 1], **kwargs)


if __name__ == '__main__':
    model = dilateformer_t().cuda()
    # print(model)
    left = torch.randn(1, 3, 256, 256).cuda()
    right = torch.randn(1, 3, 256, 256).cuda()

    # summary(model, [(4, 256, 256), (1, 256, 256)])
    flops, params = profile(model, (left, right), verbose=False)

    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % flops)
    print('Total params: %s' % params)
