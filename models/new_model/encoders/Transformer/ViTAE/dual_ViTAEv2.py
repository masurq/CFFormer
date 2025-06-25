from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import numpy as np
import torch.utils.checkpoint as checkpoint
from models.new_model.encoders.Transformer.ViTAE.NormalCell import NormalCell
from models.new_model.encoders.Transformer.ViTAE.ReductionCell import ReductionCell

import math
import time
from collections import OrderedDict
from thop import clever_format, profile

from models.new_model.modules import FeatureFusion as FFM
from models.new_model.modules import FeatureCorrection_s2c as FCM


class PatchEmbedding(nn.Module):
    def __init__(self, inter_channel=32, out_channels=48, img_size=None):
        self.img_size = img_size
        self.inter_channel = inter_channel
        self.out_channel = out_channels
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, inter_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channel, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, size):
        x = self.conv3(self.conv2(self.conv1(x)))
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return x, (h, w)

    def flops(self, ) -> float:
        flops = 0
        flops += 3 * self.inter_channel * self.img_size[0] * self.img_size[1] // 4 * 9
        flops += self.img_size[0] * self.img_size[1] // 4 * self.inter_channel
        flops += self.inter_channel * self.out_channel * self.img_size[0] * self.img_size[1] // 16 * 9
        flops += self.img_size[0] * self.img_size[1] // 16 * self.out_channel
        flops += self.out_channel * self.out_channel * self.img_size[0] * self.img_size[1] // 16
        return flops


class BasicLayer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=64, token_dims=64, downsample_ratios=4, kernel_size=7,
                 RC_heads=1, NC_heads=6, dilations=[1, 2, 3, 4],
                 RC_op='cat', RC_tokens_type='performer', NC_tokens_type='transformer', RC_group=1, NC_group=64,
                 NC_depth=2, dpr=0.1, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0, attn_drop=0., norm_layer=nn.LayerNorm, class_token=False, gamma=False,
                 init_values=1e-4, SE=False, window_size=7,
                 use_checkpoint=False):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.token_dims = token_dims
        self.downsample_ratios = downsample_ratios
        self.out_size = self.img_size // self.downsample_ratios
        self.RC_kernel_size = kernel_size
        self.RC_heads = RC_heads
        self.NC_heads = NC_heads
        self.dilations = dilations
        self.RC_op = RC_op
        self.RC_tokens_type = RC_tokens_type
        self.RC_group = RC_group
        self.NC_group = NC_group
        self.NC_depth = NC_depth
        self.use_checkpoint = use_checkpoint
        if RC_tokens_type == 'stem':
            self.RC = PatchEmbedding(inter_channel=token_dims // 2, out_channels=token_dims, img_size=img_size)
        elif downsample_ratios > 1:
            self.RC = ReductionCell(img_size, in_chans, embed_dims, token_dims, downsample_ratios, kernel_size,
                                    RC_heads, dilations, op=RC_op, tokens_type=RC_tokens_type, group=RC_group,
                                    gamma=gamma, init_values=init_values, SE=SE)
        else:
            self.RC = nn.Identity()
        self.NC = nn.ModuleList([
            NormalCell(token_dims, NC_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                       attn_drop=attn_drop,
                       drop_path=dpr[i] if isinstance(dpr, list) else dpr, norm_layer=norm_layer,
                       class_token=class_token, group=NC_group, tokens_type=NC_tokens_type,
                       gamma=gamma, init_values=init_values, SE=SE, img_size=img_size // downsample_ratios,
                       window_size=window_size, shift_size=0)
            for i in range(NC_depth)])

    def forward(self, x, size):
        h, w = size
        x, (h, w) = self.RC(x, (h, w))
        # print(h, w)
        for nc in self.NC:
            nc.H = h
            nc.W = w
            if self.use_checkpoint:
                x = checkpoint.checkpoint(nc, x)
            else:
                x = nc(x)
            # print(h, w)
        return x, (h, w)


class DualViTAEv2(nn.Module):
    def __init__(self,
                 img_size=224,
                 in_chans1=3,
                 in_chans2=1,
                 embed_dims=64,
                 token_dims=64,
                 downsample_ratios=[4, 2, 2, 2],
                 kernel_size=[7, 3, 3, 3],
                 RC_heads=[1, 1, 1, 1],
                 NC_heads=4,
                 dilations=[[1, 2, 3, 4], [1, 2, 3], [1, 2], [1, 2]],
                 RC_op='cat',
                 RC_tokens_type='window',
                 NC_tokens_type='transformer',
                 RC_group=[1, 1, 1, 1],
                 NC_group=[1, 32, 64, 64],
                 NC_depth=[2, 2, 6, 2],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_fuse=nn.BatchNorm2d,
                 stages=4,
                 window_size=7,
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 load_ema=True):
        super().__init__()

        self.stages = stages
        self.load_ema = load_ema
        repeatOrNot = (lambda x, y, z=list: x if isinstance(x, z) else [x for _ in range(y)])
        self.embed_dims = repeatOrNot(embed_dims, stages)
        self.tokens_dims = token_dims if isinstance(token_dims, list) else [token_dims * (2 ** i) for i in
                                                                            range(stages)]
        self.downsample_ratios = repeatOrNot(downsample_ratios, stages)
        self.kernel_size = repeatOrNot(kernel_size, stages)
        self.RC_heads = repeatOrNot(RC_heads, stages)
        self.NC_heads = repeatOrNot(NC_heads, stages)
        self.dilaions = repeatOrNot(dilations, stages)
        self.RC_op = repeatOrNot(RC_op, stages)
        self.RC_tokens_type = repeatOrNot(RC_tokens_type, stages)
        self.NC_tokens_type = repeatOrNot(NC_tokens_type, stages)
        self.RC_group = repeatOrNot(RC_group, stages)
        self.NC_group = repeatOrNot(NC_group, stages)
        self.NC_depth = repeatOrNot(NC_depth, stages)
        self.mlp_ratio = repeatOrNot(mlp_ratio, stages)
        self.qkv_bias = repeatOrNot(qkv_bias, stages)
        self.qk_scale = repeatOrNot(qk_scale, stages)
        self.drop = repeatOrNot(drop_rate, stages)
        self.attn_drop = repeatOrNot(attn_drop_rate, stages)
        self.norm_layer = repeatOrNot(norm_layer, stages)
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint

        depth = np.sum(self.NC_depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        Layers = []
        aux_Layers = []

        for i in range(stages):
            startDpr = 0 if i == 0 else self.NC_depth[i - 1]
            Layers.append(
                BasicLayer(img_size, in_chans1, self.embed_dims[i], self.tokens_dims[i], self.downsample_ratios[i],
                           self.kernel_size[i], self.RC_heads[i], self.NC_heads[i], self.dilaions[i], self.RC_op[i],
                           self.RC_tokens_type[i], self.NC_tokens_type[i], self.RC_group[i], self.NC_group[i],
                           self.NC_depth[i], dpr[startDpr:self.NC_depth[i] + startDpr],
                           mlp_ratio=self.mlp_ratio[i], qkv_bias=self.qkv_bias[i], qk_scale=self.qk_scale[i],
                           drop=self.drop[i], attn_drop=self.attn_drop[i],
                           norm_layer=self.norm_layer[i], window_size=window_size, use_checkpoint=use_checkpoint)
            )
            aux_Layers.append(
                BasicLayer(img_size, in_chans2, self.embed_dims[i], self.tokens_dims[i], self.downsample_ratios[i],
                           self.kernel_size[i], self.RC_heads[i], self.NC_heads[i], self.dilaions[i], self.RC_op[i],
                           self.RC_tokens_type[i], self.NC_tokens_type[i], self.RC_group[i], self.NC_group[i],
                           self.NC_depth[i], dpr[startDpr:self.NC_depth[i] + startDpr],
                           mlp_ratio=self.mlp_ratio[i], qkv_bias=self.qkv_bias[i], qk_scale=self.qk_scale[i],
                           drop=self.drop[i], attn_drop=self.attn_drop[i],
                           norm_layer=self.norm_layer[i], window_size=window_size, use_checkpoint=use_checkpoint)
            )
            img_size = img_size // self.downsample_ratios[i]
            in_chans1 = self.tokens_dims[i]
            in_chans2 = self.tokens_dims[i]

        self.layers = nn.ModuleList(Layers)
        self.aux_layers = nn.ModuleList(aux_Layers)
        self.num_layers = len(Layers)

        self.FCMs = nn.ModuleList([
            FCM(dim=self.tokens_dims[0], reduction=1),
            FCM(dim=self.tokens_dims[1], reduction=1),
            FCM(dim=self.tokens_dims[2], reduction=1),
            FCM(dim=self.tokens_dims[3], reduction=1)])

        self.FFMs = nn.ModuleList([
            FFM(dim=self.tokens_dims[0], reduction=1, num_heads=self.NC_heads[0], norm_layer=norm_fuse,
                sr_ratio=sr_ratios[0]),
            FFM(dim=self.tokens_dims[1], reduction=1, num_heads=self.NC_heads[0], norm_layer=norm_fuse,
                sr_ratio=sr_ratios[1]),
            FFM(dim=self.tokens_dims[2], reduction=1, num_heads=self.NC_heads[0], norm_layer=norm_fuse,
                sr_ratio=sr_ratios[2]),
            FFM(dim=self.tokens_dims[3], reduction=1, num_heads=self.NC_heads[0], norm_layer=norm_fuse,
                sr_ratio=sr_ratios[3])])

        self.apply(self._init_weights)
        self._freeze_stages()

    def _freeze_stages(self):

        if self.frozen_stages > 0:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

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

    # def forwardTwoLayer(self, x):
    #     x1 = self.layers[0](x)
    #     x2 = self.layers[1](x1)
    #     return x1, x2
    #
    # def forwardThreeLayer(self, x):
    #     x0 = self.layers[1](x)
    #     x1 = self.layers[2](x0)
    #     x2 = self.layers[3](x1)
    #     return x0, x1, x2

    # def forward_features(self, x):
    #     b, c, h, w = x.shape
    #     for layer in self.layers:
    #         x, (h, w) = layer(x, (h, w))
    #     return x
    #
    # def forward(self, x):
    #     x = self.forward_features(x)
    #     # x = self.head(x)
    #     return x

    def forward(self, x1, x2):
        """Forward function."""
        # torch.cuda.empty_cache()
        # x.requires_grad = True
        outs = []
        # if self.use_checkpoint:
        #     x_out = checkpoint.checkpoint(self.layers[0], x)
        # else:
        b, _, h1, w1 = x1.shape
        b, _, h2, w2 = x2.shape
        for i in range(4):
            x1, (h1, w1) = self.layers[i](x1, (h1, w1))
            x2, (h2, w2) = self.aux_layers[i](x2, (h2, w2))

            x1, x2 = self.FCMs[i](x1.reshape(b, h1, w1, -1).permute(0, 3, 1, 2).contiguous(),
                                  x2.reshape(b, h2, w2, -1).permute(0, 3, 1, 2).contiguous())

            fuse = self.FFMs[i](x1, x2)
            outs.append(fuse)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(DualViTAEv2, self).train(mode)
        self._freeze_stages()


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
    for k, v in raw_state_dict['state_dict_ema'].items():
        if k.find('layers') >= 0:
            state_dict[k] = v
            state_dict[k.replace('layers', 'aux_layers')] = v

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


class ViTAEv2_S(DualViTAEv2):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(ViTAEv2_S, self).__init__(RC_tokens_type=['window', 'window', 'transformer', 'transformer'],
                                        NC_tokens_type=['window', 'window', 'transformer', 'transformer'],
                                        stages=4,
                                        embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512],
                                        downsample_ratios=[4, 2, 2, 2],
                                        NC_depth=[2, 2, 8, 2], NC_heads=[1, 2, 4, 8], RC_heads=[1, 1, 2, 4],
                                        mlp_ratio=4.,
                                        NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], window_size=7,
                                        **kwargs)


class ViTAEv2_48M(DualViTAEv2):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(ViTAEv2_48M, self).__init__(RC_tokens_type=['window', 'window', 'transformer', 'transformer'],
                                          NC_tokens_type=['window', 'window', 'transformer', 'transformer'],
                                          stages=4,
                                          embed_dims=[64, 64, 192, 384], token_dims=[96, 192, 384, 768],
                                          downsample_ratios=[4, 2, 2, 2],
                                          NC_depth=[2, 2, 11, 2], NC_heads=[1, 2, 4, 8], RC_heads=[1, 1, 2, 4],
                                          mlp_ratio=4.,
                                          NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], **kwargs)


class ViTAEv2_B(DualViTAEv2):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(ViTAEv2_B, self).__init__(RC_tokens_type=['window', 'window', 'transformer', 'transformer'],
                                        NC_tokens_type=['window', 'window', 'transformer', 'transformer'],
                                        stages=4,
                                        embed_dims=[96, 96, 256, 512], token_dims=[128, 256, 512, 1024],
                                        downsample_ratios=[4, 2, 2, 2],
                                        NC_depth=[2, 2, 12, 2], NC_heads=[1, 4, 8, 16], RC_heads=[1, 1, 4, 8],
                                        mlp_ratio=4.,
                                        NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], **kwargs)


if __name__ == '__main__':
    model = ViTAEv2_S().cuda()
    print(model)
    left = torch.randn(1, 3, 256, 256).cuda()
    right = torch.randn(1, 3, 256, 256).cuda()

    # summary(model, [(4, 256, 256), (1, 256, 256)])
    flops, params = profile(model, (left, right), verbose=False)

    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % flops)
    print('Total params: %s' % params)
