import torch
from torch import nn, Tensor
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from thop import clever_format, profile
import time
import math
from collections import OrderedDict

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.new_model.modules import FeatureFusion as FFM
from models.new_model.modules import FeatureCorrection_s2c as FCM


class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU6(True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, c1, c2, s, expand_ratio):
        super().__init__()
        ch = int(round(c1 * expand_ratio))
        self.use_res_connect = s == 1 and c1 == c2

        layers = []

        if expand_ratio != 1:
            layers.append(ConvModule(c1, ch, 1))

        layers.extend([
            ConvModule(ch, ch, 3, s, 1, g=ch),
            nn.Conv2d(ch, c2, 1, bias=False),
            nn.BatchNorm2d(c2)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


mobilenetv2_settings = {
    '1.0': []
}


class DualMobileNetV2(nn.Module):
    def __init__(self, norm_fuse=nn.BatchNorm2d, variant: str = None):
        super().__init__()
        self.out_indices = [3, 6, 13, 17]
        self.channels = [24, 32, 96, 320]
        input_channel = 32

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.features = nn.ModuleList([ConvModule(3, input_channel, 3, 2, 1)])
        self.aux_features = nn.ModuleList([ConvModule(3, input_channel, 3, 2, 1)])

        for t, c, n, s in inverted_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, t))
                self.aux_features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel

        self.FCMs = nn.ModuleList([
            FCM(dim=self.channels[0], reduction=1),
            FCM(dim=self.channels[1], reduction=1),
            FCM(dim=self.channels[2], reduction=1),
            FCM(dim=self.channels[3], reduction=1)])

        self.FFMs = nn.ModuleList([
            FFM(dim=self.channels[0], reduction=1, num_heads=1, norm_layer=norm_fuse, sr_ratio=4),
            FFM(dim=self.channels[1], reduction=1, num_heads=2, norm_layer=norm_fuse, sr_ratio=3),
            FFM(dim=self.channels[2], reduction=1, num_heads=3, norm_layer=norm_fuse, sr_ratio=2),
            FFM(dim=self.channels[3], reduction=1, num_heads=4, norm_layer=norm_fuse, sr_ratio=1)])

        self.apply(self._init_weights)

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

    def forward(self, x1, x2):
        outs = []

        for block, aux_block, fcm, ffm in zip(
                [self.features[:4], self.features[4:7], self.features[7:14], self.features[14:]],
                [self.aux_features[:4], self.aux_features[4:7], self.aux_features[7:14], self.aux_features[14:]],
                [self.FCMs[0], self.FCMs[1], self.FCMs[2], self.FCMs[3]],
                [self.FFMs[0], self.FFMs[1], self.FFMs[2], self.FFMs[3]]):

            for blk in block:
                x1 = blk(x1)
            for blk in aux_block:
                x2 = blk(x2)
            x1, x2 = fcm(x1, x2)
            fuse = ffm(x1, x2)
            outs.append(fuse)

        return tuple(outs)


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
        if k.find('features') >= 0:
            state_dict[k] = v
            state_dict[k.replace('features', 'aux_features')] = v

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


if __name__ == '__main__':
    model = DualMobileNetV2().cuda()
    print(model)
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
