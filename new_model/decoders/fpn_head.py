import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignedModule_v1(nn.Module):

    def __init__(self, inplane, outplane=256, kernel_size=3, eps=1e-8):
        super(AlignedModule_v1, self).__init__()
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 2, kernel_size=kernel_size, padding=1, bias=False)
        # 自定义可训练权重参数
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps

    def forward(self, low_feature, h_feature):
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        fuse_feature = fuse_weights[0] * h_feature + fuse_weights[1] * low_feature

        return fuse_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class AlignedModule_v2(nn.Module):

    def __init__(self, inplane, outplane=256, kernel_size=3, eps=1e-8):
        super(AlignedModule_v2, self).__init__()
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.l_flow_make = nn.Conv2d(outplane * 2, 2, kernel_size=kernel_size, padding=1, bias=False)
        self.h_flow_make = nn.Conv2d(outplane * 2, 2, kernel_size=kernel_size, padding=1, bias=False)

        # 自定义可训练权重参数
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps

    def forward(self, low_feature, high_feature):
        h, w = low_feature.size()[2:]
        size = (h, w)
        l_feature = self.down_l(low_feature)

        h_feature = F.interpolate(high_feature, size=size, mode="bilinear", align_corners=True)
        concat = torch.cat([h_feature, l_feature], 1)

        l_flow = self.l_flow_make(concat)
        h_flow = self.h_flow_make(concat)

        l_feature_warp = self.flow_warp(l_feature, l_flow, size=size)
        h_feature_warp = self.flow_warp(high_feature, h_flow, size=size)

        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        fuse_feature = fuse_weights[0] * h_feature_warp + fuse_weights[1] * l_feature_warp

        return fuse_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class AlignedModule_v3(nn.Module):

    def __init__(self, inplane, outplane=256, kernel_size=3, eps=1e-8):
        super(AlignedModule_v3, self).__init__()
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 4, kernel_size=kernel_size, padding=1, bias=False)

        # 自定义可训练权重参数
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps

    def forward(self, low_feature, high_feature):
        h, w = low_feature.size()[2:]
        size = (h, w)
        l_feature = self.down_l(low_feature)

        h_feature = F.interpolate(high_feature, size=size, mode="bilinear", align_corners=True)
        concat = torch.cat([h_feature, l_feature], 1)

        flow = self.flow_make(concat)
        flow_up, flow_down = flow[:, :2, :, :], flow[:, 2:, :, :]

        l_feature_warp = self.flow_warp(l_feature, flow_down, size=size)
        h_feature_warp = self.flow_warp(high_feature, flow_up, size=size)

        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        fuse_feature = fuse_weights[0] * h_feature_warp + fuse_weights[1] * l_feature_warp

        return fuse_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


# 先上采样再调整channel
class FPNBlock(nn.Module):
    def __init__(self, skip_channels, pyramid_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, skip, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    policy
                )
            )
        self.policy = policy

    def forward(self, x):
        if self.policy == 'add':
            return sum(x)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy)
            )


class AlignedFPNDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels=[64, 128, 320, 512],
            num_classes=40,
            pyramid_channels=512,
            segmentation_channels=256,
            dropout=0.1,
            merge_policy="cat",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4

        self.p5 = nn.Conv2d(encoder_channels[3], pyramid_channels, kernel_size=1)
        self.p4 = AlignedModule_v1(encoder_channels[2], pyramid_channels)
        self.p3 = AlignedModule_v1(encoder_channels[1], pyramid_channels)
        self.p2 = AlignedModule_v1(encoder_channels[0], pyramid_channels)

        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
            for n_upsamples in [3, 2, 1, 0]
        ])

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout)
        self.pred = nn.Conv2d(self.out_channels, self.num_classes, kernel_size=1)

    def forward(self, features):
        # len=4, 1/4,1/8,1/16,1/32
        c2, c3, c4, c5 = features

        p5 = self.p5(c5)
        p4 = self.p4(c4, p5)
        p3 = self.p3(c3, p4)
        p2 = self.p2(c2, p3)

        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)
        x = self.pred(x)

        return x
