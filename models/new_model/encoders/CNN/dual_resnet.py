import time
import torch
import torch.nn as nn

from models.new_model.modules import FeatureFusion as FFM
from models.new_model.modules import FeatureCorrection_s2c as FCM

from collections import OrderedDict
from thop import clever_format, profile


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class DualBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=None,
                 bn_eps=1e-5, bn_momentum=0.1, downsample=None, inplace=True):
        super(DualBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)

        self.extra_conv1 = conv3x3(inplanes, planes, stride)
        self.extra_bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.extra_relu = nn.ReLU(inplace=inplace)
        self.extra_relu_inplace = nn.ReLU(inplace=True)
        self.extra_conv2 = conv3x3(planes, planes)
        self.extra_bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)

        self.downsample = downsample
        self.extra_downsample = downsample

        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        #  first path
        x1 = x[0]
        residual1 = x1

        out1 = self.conv1(x1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)

        if self.downsample is not None:
            residual1 = self.downsample(x1)

        # second path
        x2 = x[1]
        residual2 = x2

        out2 = self.extra_conv1(x2)
        out2 = self.extra_bn1(out2)
        out2 = self.extra_relu(out2)

        out2 = self.extra_conv2(out2)
        out2 = self.extra_bn2(out2)

        if self.extra_downsample is not None:
            residual2 = self.extra_downsample(x2)

        out1 += residual1
        out2 += residual2

        out1 = self.relu_inplace(out1)
        out2 = self.relu_inplace(out2)

        return [out1, out2]


class DualBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 norm_layer=None, bn_eps=1e-5, bn_momentum=0.1,
                 downsample=None, inplace=True):
        super(DualBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = norm_layer(planes * self.expansion, eps=bn_eps,
                              momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.extra_conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.extra_bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.extra_conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                     padding=1, bias=False)
        self.extra_bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.extra_conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                                     bias=False)
        self.extra_bn3 = norm_layer(planes * self.expansion, eps=bn_eps,
                                    momentum=bn_momentum)
        self.extra_relu = nn.ReLU(inplace=inplace)
        self.extra_relu_inplace = nn.ReLU(inplace=True)
        self.extra_downsample = downsample

        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        # first path
        x1 = x[0]
        residual1 = x1

        out1 = self.conv1(x1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out1 = self.relu(out1)

        out1 = self.conv3(out1)
        out1 = self.bn3(out1)

        if self.downsample is not None:
            residual1 = self.downsample(x1)

        # second path
        x2 = x[1]
        residual2 = x2

        out2 = self.extra_conv1(x2)
        out2 = self.extra_bn1(out2)
        out2 = self.extra_relu(out2)

        out2 = self.extra_conv2(out2)
        out2 = self.extra_bn2(out2)
        out2 = self.extra_relu(out2)

        out2 = self.extra_conv3(out2)
        out2 = self.extra_bn3(out2)

        if self.extra_downsample is not None:
            residual2 = self.extra_downsample(x2)

        out1 += residual1
        out2 += residual2
        out1 = self.relu_inplace(out1)
        out2 = self.relu_inplace(out2)

        return [out1, out2]


class DualResNet(nn.Module):
    def __init__(self, block, layers, channels=[256, 512, 1024, 2048], num_heads=[4, 8, 16, 32], reduction=[4, 4, 4, 4],
                 sr_ratios=[8, 4, 2, 1], norm_layer=nn.BatchNorm2d, bn_eps=1e-5, bn_momentum=0.1, deep_stem=False,
                 stem_width=64, inplace=True):
        self.inplanes = stem_width * 2 if deep_stem else 64
        super().__init__()
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
            )
            self.extra_conv1 = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.extra_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False)

        self.bn1 = norm_layer(stem_width * 2 if deep_stem else 64, eps=bn_eps,
                              momentum=bn_momentum)
        self.extra_bn1 = norm_layer(stem_width * 2 if deep_stem else 64, eps=bn_eps,
                                    momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.extra_relu = nn.ReLU(inplace=inplace)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.extra_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, norm_layer, 64, layers[0],
                                       inplace,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, norm_layer, 128, layers[1],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, norm_layer, 256, layers[2],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, norm_layer, 512, layers[3],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)

        self.FCMs = nn.ModuleList([
            FCM(dim=channels[0], reduction=reduction[0]),
            FCM(dim=channels[1], reduction=reduction[1]),
            FCM(dim=channels[2], reduction=reduction[2]),
            FCM(dim=channels[3], reduction=reduction[3])])

        self.FFMs = nn.ModuleList([
            FFM(dim=channels[0], reduction=reduction[0], num_heads=num_heads[0], norm_layer=norm_layer,
                sr_ratio=sr_ratios[0]),
            FFM(dim=channels[1], reduction=reduction[1], num_heads=num_heads[1], norm_layer=norm_layer,
                sr_ratio=sr_ratios[1]),
            FFM(dim=channels[2], reduction=reduction[2], num_heads=num_heads[2], norm_layer=norm_layer,
                sr_ratio=sr_ratios[2]),
            FFM(dim=channels[3], reduction=reduction[3], num_heads=num_heads[3], norm_layer=norm_layer,
                sr_ratio=sr_ratios[3])])

        self.apply(self._init_weights)

    def _make_layer(self, block, norm_layer, planes, blocks, inplace=True,
                    stride=1, bn_eps=1e-5, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion, eps=bn_eps,
                           momentum=bn_momentum),
            )

        layers = [block(self.inplanes, planes, stride, norm_layer, bn_eps,
                        bn_momentum, downsample, inplace)]

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=norm_layer, bn_eps=bn_eps,
                                bn_momentum=bn_momentum, inplace=inplace))

        return nn.Sequential(*layers)

    def _init_weights(self, feat):
        for name, m in feat.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-5
                m.momentum = 0.1
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):

        if isinstance(pretrained, str):
            load_dualpath_model(self, pretrained)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x2 = self.extra_conv1(x2)
        x2 = self.extra_bn1(x2)
        x2 = self.extra_relu(x2)
        x2 = self.extra_maxpool(x2)

        outs = []
        x = [x1, x2]
        x = self.layer1(x)
        x = self.FCMs[0](x[0], x[1])
        x_fused = self.FFMs[0](x[0], x[1])
        outs.append(x_fused)

        x = self.layer2(x)
        x = self.FCMs[1](x[0], x[1])
        x_fused = self.FFMs[1](x[0], x[1])
        outs.append(x_fused)

        x = self.layer3(x)
        x = self.FCMs[2](x[0], x[1])
        x_fused = self.FFMs[2](x[0], x[1])
        outs.append(x_fused)

        x = self.layer4(x)
        x = self.FCMs[3](x[0], x[1])
        x_fused = self.FFMs[3](x[0], x[1])
        outs.append(x_fused)

        return tuple(outs)


def load_dualpath_model(model, model_file, is_restore=False):
    t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))

        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file

    state_dict = {}
    for k, v in raw_state_dict.items():
        state_dict[k.replace('.bn.', '.')] = v
        if k.find('conv1') >= 0:
            state_dict[k] = v
            state_dict[k.replace('conv1', 'extra_conv1')] = v
        if k.find('conv2') >= 0:
            state_dict[k] = v
            state_dict[k.replace('conv2', 'extra_conv2')] = v
        if k.find('conv3') >= 0:
            state_dict[k] = v
            state_dict[k.replace('conv3', 'extra_conv3')] = v
        if k.find('bn1') >= 0:
            state_dict[k] = v
            state_dict[k.replace('bn1', 'extra_bn1')] = v
        if k.find('bn2') >= 0:
            state_dict[k] = v
            state_dict[k.replace('bn2', 'extra_bn2')] = v
        if k.find('bn3') >= 0:
            state_dict[k] = v
            state_dict[k.replace('bn3', 'extra_bn3')] = v
        if k.find('downsample') >= 0:
            state_dict[k] = v
            state_dict[k.replace('downsample', 'extra_downsample')] = v
    t_ioend = time.time()

    # 用于判断保存权重时是否使用了并行多GPU训练，因为使用DataParallel时模型的state_dict中的键值对会被自动加上前缀"module"，因此如果训练过程
    # 使用了DataParallel，则在加载模型权重时也需要加上对应的前缀
    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)

    # ckpt_keys = set(state_dict.keys())
    # own_keys = set(model.state_dict().keys())
    # missing_keys = own_keys - ckpt_keys
    # unexpected_keys = ckpt_keys - own_keys
    #
    # if len(missing_keys) > 0:
    #     print('Missing key(s) in state_dict: {}'.format(
    #         ', '.join('{}'.format(k) for k in missing_keys)))
    # if len(unexpected_keys) > 0:
    #     print('Unexpected key(s) in state_dict: {}'.format(
    #         ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    print("Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
        t_ioend - t_start, t_end - t_ioend))


class resnet18(DualResNet):
    def __init__(self, **kwargs):
        super(resnet18, self).__init__(DualBasicBlock, [2, 2, 2, 2], [64, 128, 256, 512], [1, 2, 4, 8], [1, 1, 1, 2],
                                       **kwargs)


class resnet34(DualResNet):
    def __init__(self, **kwargs):
        super(resnet34, self).__init__(DualBasicBlock, [3, 4, 6, 3], [64, 128, 256, 512], [1, 2, 4, 8], [1, 1, 1, 1],
                                       **kwargs)


class resnet50(DualResNet):
    def __init__(self, **kwargs):
        super(resnet50, self).__init__(DualBottleneck, [3, 4, 6, 3], [256, 512, 1024, 2048], [1, 2, 4, 8], **kwargs)


class resnet101(DualResNet):
    def __init__(self, **kwargs):
        super(resnet101, self).__init__(DualBottleneck, [3, 4, 23, 3], [256, 512, 1024, 2048], [1, 2, 4, 8], **kwargs)


class resnet152(DualResNet):
    def __init__(self, **kwargs):
        super(resnet152, self).__init__(DualBottleneck, [3, 8, 36, 3], [256, 512, 1024, 2048], [1, 2, 4, 8], **kwargs)


if __name__ == '__main__':
    pre_3_3 = r'G:\for_deepl\ZMSeg\pretrained\sa_gate_resnet\resnet101_v1c.pth'
    pre_7_7 = r'G:\for_deepl\ZMSeg\pretrained\sa_gate_resnet\resnet101-5d3b4d8f.pth'
    model = resnet18().cuda()
    # print(model)
    left = torch.randn(1, 3, 256, 256).cuda()
    right = torch.randn(1, 3, 256, 256).cuda()

    # summary(model, [(4, 256, 256), (1, 256, 256)])
    flops, params = profile(model, (left, right), verbose=False)

    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % flops)
    print('Total params: %s' % params)
