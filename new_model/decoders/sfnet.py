import torch
from torch import nn, Tensor
from torch.nn import functional as F


class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )


class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )


class PPM(nn.Module):
    """Pyramid Pooling Module in PSPNet
    """

    def __init__(self, c1, c2=128, scales=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvModule(c1, c2, 1)
                # ConvModule(c1, c2, 1, p=1)
            )
            for scale in scales])

        self.bottleneck = ConvModule(c1 + c2 * len(scales), c2, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        outs = []
        for stage in self.stages:
            outs.append(F.interpolate(stage(x), size=x.shape[-2:], mode='bilinear', align_corners=True))

        outs = [x] + outs[::-1]
        out = self.bottleneck(torch.cat(outs, dim=1))
        return out


class AlignedModule(nn.Module):
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.down_h = nn.Conv2d(c1, c2, 1, bias=False)
        self.down_l = nn.Conv2d(c1, c2, 1, bias=False)
        self.flow_make = nn.Conv2d(c2 * 2, 2, k, 1, 1, bias=False)

    def forward(self, low_feature: Tensor, high_feature: Tensor) -> Tensor:
        high_feature_origin = high_feature
        H, W = low_feature.shape[-2:]
        low_feature = self.down_l(low_feature)
        high_feature = self.down_h(high_feature)
        high_feature = F.interpolate(high_feature, size=(H, W), mode='bilinear', align_corners=True)
        flow = self.flow_make(torch.cat([high_feature, low_feature], dim=1))
        high_feature = self.flow_warp(high_feature_origin, flow, (H, W))
        return high_feature

    def flow_warp(self, x: Tensor, flow: Tensor, size: tuple) -> Tensor:
        norm = torch.tensor([[[[*size]]]]).type_as(x).to(x.device)
        H = torch.linspace(-1.0, 1.0, size[0]).view(-1, 1).repeat(1, size[1])
        W = torch.linspace(-1.0, 1.0, size[1]).repeat(size[0], 1)
        grid = torch.cat((W.unsqueeze(2), H.unsqueeze(2)), dim=2)
        grid = grid.repeat(x.shape[0], 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(x, grid, align_corners=False)
        return output


class SFHead(nn.Module):
    def __init__(self, in_channels, channel=256, num_classes=19, scales=(1, 2, 3, 6)):
        super().__init__()
        self.ppm = PPM(in_channels[-1], channel, scales)

        self.fpn_in = nn.ModuleList([])
        self.fpn_out = nn.ModuleList([])
        self.fpn_out_align = nn.ModuleList([])

        for in_ch in in_channels[:-1]:
            self.fpn_in.append(ConvModule(in_ch, channel, 1))
            self.fpn_out.append(ConvModule(channel, channel, 3, 1, 1))
            self.fpn_out_align.append(AlignedModule(channel, channel // 2))

        self.bottleneck = ConvModule(len(in_channels) * channel, channel, 3, 1, 1)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(channel, num_classes, 1)

    def forward(self, features: list) -> Tensor:
        f = self.ppm(features[-1])
        fpn_features = [f]

        for i in reversed(range(len(features) - 1)):
            feature = self.fpn_in[i](features[i])
            f = feature + self.fpn_out_align[i](feature, f)
            fpn_features.append(self.fpn_out[i](f))

        fpn_features.reverse()

        for i in range(1, len(fpn_features)):
            fpn_features[i] = F.interpolate(fpn_features[i], size=fpn_features[0].shape[-2:], mode='bilinear',
                                            align_corners=True)

        output = self.bottleneck(torch.cat(fpn_features, dim=1))
        output = self.conv_seg(self.dropout(output))
        return output
