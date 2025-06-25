import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from spatial_correlation_sampler import spatial_correlation_sample


class AlignedModulev1(nn.Module):

    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModulev1, self).__init__()
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_r = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 4, kernel_size=kernel_size, padding=1, bias=False)

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

    @classmethod
    def flow_warp(cls, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

    def forward(self, x1, x2):

        h, w = x1.size()[2:]
        size = (h, w)
        main_feature = self.down_l(x1)
        aux_feature = self.down_r(x2)

        flow = self.flow_make(torch.cat([main_feature, aux_feature, ], 1))
        flow_main, flow_aux = flow[:, :2, :, :], flow[:, 2:, :, :]

        main_feature_warp = self.flow_warp(x1, flow_main, size=size)
        aux_feature_warp = self.flow_warp(x2, flow_aux, size=size)

        return main_feature_warp, aux_feature_warp


