from functools import partial
from einops import rearrange, reduce
from timm.layers.activations import *
from timm.models.layers import DropPath, trunc_normal_
from timm.models.efficientnet import SqueezeExcite as SE
from torch.nn.modules.batchnorm import _BatchNorm

from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from thop import clever_format, profile
import time
import math
from collections import OrderedDict

from models.new_model.modules import FeatureFusion as FFM
from models.new_model.modules import FeatureCorrection_s2c as FCM

inplace = True


class LayerNorm2d(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x


def get_norm(norm_layer='in_1d'):
    eps = 1e-6
    norm_dict = {
        'none': nn.Identity,
        'in_1d': partial(nn.InstanceNorm1d, eps=eps),
        'in_2d': partial(nn.InstanceNorm2d, eps=eps),
        'in_3d': partial(nn.InstanceNorm3d, eps=eps),
        'bn_1d': partial(nn.BatchNorm1d, eps=eps),
        'bn_2d': partial(nn.BatchNorm2d, eps=eps),
        'bn_3d': partial(nn.BatchNorm3d, eps=eps),
        'gn': partial(nn.GroupNorm, eps=eps),
        'ln_1d': partial(nn.LayerNorm, eps=eps),
        'ln_2d': partial(LayerNorm2d, eps=eps),
    }
    return norm_dict[norm_layer]


def get_act(act_layer='relu'):
    act_dict = {
        'none': nn.Identity,
        'sigmoid': Sigmoid,
        'swish': Swish,
        'mish': Mish,
        'hsigmoid': HardSigmoid,
        'hswish': HardSwish,
        'hmish': HardMish,
        'tanh': Tanh,
        'relu': nn.ReLU,
        'relu6': nn.ReLU6,
        'prelu': PReLU,
        'gelu': GELU,
        'silu': nn.SiLU
    }
    return act_dict[act_layer]


class ConvNormAct(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False,
                 skip=False, norm_layer='bn_2d', act_layer='relu', inplace=True, drop_path_rate=0.):
        super(ConvNormAct, self).__init__()
        self.has_skip = skip and dim_in == dim_out
        padding = math.ceil((kernel_size - stride) / 2)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = get_norm(norm_layer)(dim_out)
        self.act = get_act(act_layer)(inplace=inplace)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class LayerScale2D(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=True):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class MSPatchEmb(nn.Module):

    def __init__(self, dim_in, emb_dim, kernel_size=2, c_group=-1, stride=1, dilations=[1, 2, 3],
                 norm_layer='bn_2d', act_layer='silu'):
        super().__init__()
        self.dilation_num = len(dilations)
        assert dim_in % c_group == 0
        c_group = math.gcd(dim_in, emb_dim) if c_group == -1 else c_group
        self.convs = nn.ModuleList()
        for i in range(len(dilations)):
            padding = math.ceil(((kernel_size - 1) * dilations[i] + 1 - stride) / 2)
            self.convs.append(nn.Sequential(
                nn.Conv2d(dim_in, emb_dim, kernel_size, stride, padding, dilations[i], groups=c_group),
                get_norm(norm_layer)(emb_dim),
                get_act(act_layer)(emb_dim)))

    def forward(self, x):
        if self.dilation_num == 1:
            x = self.convs[0](x)
        else:
            x = torch.cat([self.convs[i](x).unsqueeze(dim=-1) for i in range(self.dilation_num)], dim=-1)
            x = reduce(x, 'b c h w n -> b c h w', 'mean').contiguous()
        return x


class iRMB(nn.Module):

    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_2d',
                 act_layer='relu', v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=64, window_size=7,
                 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
        super().__init__()
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s
        if self.attn_s:
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            self.dim_head = dim_head
            self.window_size = window_size
            self.num_head = dim_in // dim_head
            self.scale = self.dim_head ** -0.5
            self.attn_pre = attn_pre
            self.qk = ConvNormAct(dim_in, int(dim_in * 2), kernel_size=1, bias=qkv_bias, norm_layer='none',
                                  act_layer='none')
            self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias,
                                 norm_layer='none', act_layer=act_layer, inplace=inplace)
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            if v_proj:
                self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, bias=qkv_bias, norm_layer='none',
                                     act_layer=act_layer, inplace=inplace)
            else:
                self.v = nn.Identity()
        self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=dw_ks, stride=stride, dilation=dilation,
                                      groups=dim_mid, norm_layer='bn_2d', act_layer='silu', inplace=inplace)
        self.se = SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()

        self.proj_drop = nn.Dropout(drop)
        self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer='none', act_layer='none', inplace=inplace)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        B, C, H, W = x.shape
        if self.attn_s:
            # padding
            if self.window_size <= 0:
                window_size_W, window_size_H = W, H
            else:
                window_size_W, window_size_H = self.window_size, self.window_size
            pad_l, pad_t = 0, 0
            pad_r = (window_size_W - W % window_size_W) % window_size_W
            pad_b = (window_size_H - H % window_size_H) % window_size_H
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
            n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
            x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
            # attention
            b, c, h, w = x.shape
            qk = self.qk(x)
            qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head,
                           dim_head=self.dim_head).contiguous()
            q, k = qk[0], qk[1]
            attn_spa = (q @ k.transpose(-2, -1)) * self.scale
            attn_spa = attn_spa.softmax(dim=-1)
            attn_spa = self.attn_drop(attn_spa)
            if self.attn_pre:
                x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ x
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
                                  w=w).contiguous()
                x_spa = self.v(x_spa)
            else:
                v = self.v(x)
                v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ v
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
                                  w=w).contiguous()
            # unpadding
            x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        else:
            x = self.v(x)

        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))

        x = self.proj_drop(x)
        x = self.proj(x)

        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


class DualEMO(nn.Module):

    def __init__(self, dim_in=3,
                 depths=[1, 2, 4, 2], stem_dim=16, embed_dims=[64, 128, 256, 512], exp_ratios=[4., 4., 4., 4.],
                 norm_layers=['bn_2d', 'bn_2d', 'bn_2d', 'bn_2d'], act_layers=['relu', 'relu', 'relu', 'relu'],
                 dw_kss=[3, 3, 5, 5], se_ratios=[0.0, 0.0, 0.0, 0.0], dim_heads=[32, 32, 32, 32], heads=[1, 1, 1, 1],
                 window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True], qkv_bias=True,
                 attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False, norm_fuse=nn.BatchNorm2d,
                 sr_ratios=[8, 4, 2, 1], sync_bn=False, out_indices=(1, 2, 3, 4), frozen_stages=-1, norm_eval=False):
        super().__init__()
        self.sync_bn = sync_bn
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.stage0 = nn.ModuleList([
            MSPatchEmb(  # down to 112
                dim_in, stem_dim, kernel_size=dw_kss[0], c_group=1, stride=2, dilations=[1],
                norm_layer=norm_layers[0], act_layer='none'),
            iRMB(  # ds
                stem_dim, stem_dim, norm_in=False, has_skip=False, exp_ratio=1,
                norm_layer=norm_layers[0], act_layer=act_layers[0], v_proj=False, dw_ks=dw_kss[0],
                stride=1, dilation=1, se_ratio=1,
                dim_head=dim_heads[0], window_size=window_sizes[0], attn_s=False,
                qkv_bias=qkv_bias, attn_drop=attn_drop, drop=drop, drop_path=0.,
                attn_pre=attn_pre
            )
        ])

        self.aux_stage0 = nn.ModuleList([
            MSPatchEmb(  # down to 112
                dim_in, stem_dim, kernel_size=dw_kss[0], c_group=1, stride=2, dilations=[1],
                norm_layer=norm_layers[0], act_layer='none'),
            iRMB(  # ds
                stem_dim, stem_dim, norm_in=False, has_skip=False, exp_ratio=1,
                norm_layer=norm_layers[0], act_layer=act_layers[0], v_proj=False, dw_ks=dw_kss[0],
                stride=1, dilation=1, se_ratio=1,
                dim_head=dim_heads[0], window_size=window_sizes[0], attn_s=False,
                qkv_bias=qkv_bias, attn_drop=attn_drop, drop=drop, drop_path=0.,
                attn_pre=attn_pre
            )
        ])

        emb_dim_pre = stem_dim
        for i in range(len(depths)):
            layers = []
            aux_layers = []
            dpr = dprs[sum(depths[:i]):sum(depths[:i + 1])]
            for j in range(depths[i]):
                if j == 0:
                    stride, has_skip, attn_s, exp_ratio = 2, False, False, exp_ratios[i] * 2
                else:
                    stride, has_skip, attn_s, exp_ratio = 1, True, attn_ss[i], exp_ratios[i]
                layers.append(iRMB(
                    emb_dim_pre, embed_dims[i], norm_in=True, has_skip=has_skip, exp_ratio=exp_ratio,
                    norm_layer=norm_layers[i], act_layer=act_layers[i], v_proj=True, dw_ks=dw_kss[i],
                    stride=stride, dilation=1, se_ratio=se_ratios[i],
                    dim_head=dim_heads[i], window_size=window_sizes[i], attn_s=attn_s,
                    qkv_bias=qkv_bias, attn_drop=attn_drop, drop=drop, drop_path=dpr[j], v_group=v_group,
                    attn_pre=attn_pre
                ))
                aux_layers.append(iRMB(
                    emb_dim_pre, embed_dims[i], norm_in=True, has_skip=has_skip, exp_ratio=exp_ratio,
                    norm_layer=norm_layers[i], act_layer=act_layers[i], v_proj=True, dw_ks=dw_kss[i],
                    stride=stride, dilation=1, se_ratio=se_ratios[i],
                    dim_head=dim_heads[i], window_size=window_sizes[i], attn_s=attn_s,
                    qkv_bias=qkv_bias, attn_drop=attn_drop, drop=drop, drop_path=dpr[j], v_group=v_group,
                    attn_pre=attn_pre
                ))
                emb_dim_pre = embed_dims[i]
            self.__setattr__(f'stage{i + 1}', nn.ModuleList(layers))
            self.__setattr__(f'aux_stage{i + 1}', nn.ModuleList(aux_layers))

        self.FCMs = nn.ModuleList([
            FCM(dim=embed_dims[0], reduction=1),
            FCM(dim=embed_dims[1], reduction=1),
            FCM(dim=embed_dims[2], reduction=1),
            FCM(dim=embed_dims[3], reduction=1)])

        self.FFMs = nn.ModuleList([
            FFM(dim=embed_dims[0], reduction=1, num_heads=heads[0], norm_layer=norm_fuse, sr_ratio=sr_ratios[0]),
            FFM(dim=embed_dims[1], reduction=1, num_heads=heads[1], norm_layer=norm_fuse, sr_ratio=sr_ratios[1]),
            FFM(dim=embed_dims[2], reduction=1, num_heads=heads[2], norm_layer=norm_fuse, sr_ratio=sr_ratios[2]),
            FFM(dim=embed_dims[3], reduction=1, num_heads=heads[3], norm_layer=norm_fuse, sr_ratio=sr_ratios[3])])

        self.apply(self._init_weights)
        self._sync_bn() if sync_bn else None
        self._freeze_stages()

    def _init_weights(self, m):
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

    def init_weights(self, pretrained=None):

        if isinstance(pretrained, str):
            load_dualpath_model(self, pretrained)
        else:
            raise TypeError('pretrained must be a str or None')

    def _sync_bn(self):
        self.stage0 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.stage0)
        self.stage1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.stage1)
        self.stage2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.stage2)
        self.stage3 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.stage3)
        self.stage4 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.stage4)

    def forward(self, x1, x2):
        outs = []

        for blk in self.stage0:
            x1 = blk(x1)
        for blk in self.aux_stage0:
            x2 = blk(x2)

        for blocks, aux_blocks, fcm, ffm in zip([self.stage1, self.stage2, self.stage3, self.stage4],
                                                [self.aux_stage1, self.aux_stage2, self.aux_stage3, self.aux_stage4],
                                                [self.FCMs[0], self.FCMs[1], self.FCMs[2], self.FCMs[3]],
                                                [self.FFMs[0], self.FFMs[1], self.FFMs[2], self.FFMs[3]]):

            for blk in blocks:
                x1 = blk(x1)
            for blk in aux_blocks:
                x2 = blk(x2)

            x1, x2 = fcm(x1, x2)
            fuse = ffm(x1, x2)
            outs.append(fuse)

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(0, self.frozen_stages + 1):
            m = getattr(self, f'stage{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(DualEMO, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


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
        if k.find('stage') >= 0:
            state_dict[k] = v
            state_dict[k.replace('stage', 'aux_stage')] = v

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


class emo_1M(DualEMO):
    def __init__(self, **kwargs):
        super(emo_1M, self).__init__(
            depths=[2, 2, 8, 3], stem_dim=24, embed_dims=[32, 48, 80, 168], exp_ratios=[2., 2.5, 3.0, 3.5],
            norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
            dw_kss=[3, 3, 5, 5], dim_heads=[16, 16, 20, 21], window_sizes=[7, 7, 7, 7], heads=[2, 2, 4, 7],
            attn_ss=[False, False, True, True], qkv_bias=True, attn_drop=0., drop=0., drop_path=0.04036, v_group=False,
            attn_pre=True, sync_bn=False, out_indices=(1, 2, 3, 4), frozen_stages=-1, norm_eval=False, **kwargs)


class emo_2M(DualEMO):
    def __init__(self, **kwargs):
        super(emo_2M, self).__init__(

            depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[32, 48, 120, 200], exp_ratios=[2., 2.5, 3.0, 3.5],
            norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
            dw_kss=[3, 3, 5, 5], dim_heads=[16, 16, 20, 20], window_sizes=[7, 7, 7, 7], heads=[2, 2, 5, 5],
            attn_ss=[False, False, True, True], qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False,
            attn_pre=True, sync_bn=False, out_indices=(1, 2, 3, 4), frozen_stages=-1, norm_eval=False, **kwargs)


class emo_5M(DualEMO):
    def __init__(self, **kwargs):
        super(emo_5M, self).__init__(
            depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 288], exp_ratios=[2., 3., 4., 4.],
            norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
            dw_kss=[3, 3, 5, 5], dim_heads=[24, 24, 32, 32], window_sizes=[16, 16, 16, 16], heads=[2, 3, 5, 8],
            attn_ss=[False, False, True, True], qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False,
            attn_pre=True, sync_bn=False, out_indices=(1, 2, 3, 4), frozen_stages=-1, norm_eval=False, **kwargs)


if __name__ == '__main__':
    model = emo_5M().cuda()
    # print(model)
    left = torch.randn(1, 3, 256, 256).cuda()
    right = torch.randn(1, 3, 256, 256).cuda()

    # flops = FlopCountAnalysis(model, (left, right))
    # print(flop_count_table(flops))
    # summary(model, [(4, 256, 256), (1, 256, 256)])
    flops, params = profile(model, (left, right), verbose=False)

    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % flops)
    print('Total params: %s' % params)
