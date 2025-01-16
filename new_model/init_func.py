import torch
import torch.nn as nn


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)

        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def group_weight(weight_group, module, norm_layer, lr):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

        elif isinstance(m, norm_layer) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) \
                or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

        elif isinstance(m, nn.Parameter):
            group_decay.append(m)

    assert len(list(module.parameters())) >= len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group


def nostride_dilate(m, dilate):
    if isinstance(m, nn.Conv2d):
        if m.stride == (2, 2):
            m.stride = (1, 1)
            if m.kernel_size == (3, 3):
                m.dilation = (dilate, dilate)
                m.padding = (dilate, dilate)
        else:
            if m.kernel_size == (3, 3):
                m.dilation = (dilate, dilate)
                m.padding = (dilate, dilate)


def patch_first_conv_single_biformer(model, in_channel1, in_channel2):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    conv1_found = False

    for name, module in model.named_modules():
        if not conv1_found and isinstance(module, nn.Conv2d) and "downsample_layers" in name:
            conv1_found = True
            in_channel = in_channel1 + in_channel2
            module.in_channels = in_channel
            weight = module.weight.detach()
            reset = False

            if in_channel == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel)

            module.weight = nn.parameter.Parameter(weight)

        if conv1_found:
            break


def patch_first_conv_mit(model, in_channel1, in_channel2):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    conv1_found = False
    conv2_found = False

    for name, module in model.named_modules():
        if not conv1_found and isinstance(module, nn.Conv2d) and "patch_embed1" in name:
            conv1_found = True
            module.in_channels = in_channel1
            weight = module.weight.detach()
            reset = False

            if in_channel1 == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel1 == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel1):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel1)

            module.weight = nn.parameter.Parameter(weight)

        if not conv2_found and isinstance(module, nn.Conv2d) and "extra_patch_embed1" in name:
            conv2_found = True
            module.in_channels = in_channel2
            weight = module.weight.detach()
            reset = False

            if in_channel2 == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel2 == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel2):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel2)

            module.weight = nn.parameter.Parameter(weight)

        if conv1_found and conv2_found:
            break


def patch_first_conv_DilateFormer(model, in_channel1, in_channel2):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    conv1_found = False
    conv2_found = False

    for name, module in model.named_modules():
        if not conv1_found and isinstance(module, nn.Conv2d) and "patch_embed" in name:
            conv1_found = True
            module.in_channels = in_channel1
            weight = module.weight.detach()
            reset = False

            if in_channel1 == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel1 == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel1):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel1)

            module.weight = nn.parameter.Parameter(weight)

        if not conv2_found and isinstance(module, nn.Conv2d) and "aux_patch_embed" in name:
            conv2_found = True
            module.in_channels = in_channel2
            weight = module.weight.detach()
            reset = False

            if in_channel2 == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel2 == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel2):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel2)

            module.weight = nn.parameter.Parameter(weight)

        if conv1_found and conv2_found:
            break


def patch_first_conv_swin(model, in_channel1, in_channel2):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    conv1_found = False
    conv2_found = False

    for name, module in model.named_modules():
        if not conv1_found and isinstance(module, nn.Conv2d) and "patch_embed" in name:
            conv1_found = True
            module.in_channels = in_channel1
            weight = module.weight.detach()
            reset = False

            if in_channel1 == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel1 == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel1):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel1)

            module.weight = nn.parameter.Parameter(weight)

        if not conv2_found and isinstance(module, nn.Conv2d) and "patch_embed_d" in name:
            conv2_found = True
            module.in_channels = in_channel2
            weight = module.weight.detach()
            reset = False

            if in_channel2 == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel2 == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel2):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel2)

            module.weight = nn.parameter.Parameter(weight)

        if conv1_found and conv2_found:
            break


def patch_first_conv_cswin(model, in_channel1, in_channel2):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    conv1_found = False
    conv2_found = False

    for name, module in model.named_modules():
        if not conv1_found and isinstance(module, nn.Conv2d) and "stage1_conv_embed" in name:
            conv1_found = True
            module.in_channels = in_channel1
            weight = module.weight.detach()
            reset = False

            if in_channel1 == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel1 == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel1):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel1)

            module.weight = nn.parameter.Parameter(weight)

        if not conv2_found and isinstance(module, nn.Conv2d) and "aux_stage1_conv_embed" in name:
            conv2_found = True
            module.in_channels = in_channel2
            weight = module.weight.detach()
            reset = False

            if in_channel2 == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel2 == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel2):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel2)

            module.weight = nn.parameter.Parameter(weight)

        if conv1_found and conv2_found:
            break


def patch_first_conv_biformer(model, in_channel1, in_channel2):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    conv1_found = False
    conv2_found = False

    for name, module in model.named_modules():
        if not conv1_found and isinstance(module, nn.Conv2d) and "downsample_layers" in name:
            conv1_found = True
            module.in_channels = in_channel1
            weight = module.weight.detach()
            reset = False

            if in_channel1 == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel1 == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel1):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel1)

            module.weight = nn.parameter.Parameter(weight)

        if not conv2_found and isinstance(module, nn.Conv2d) and "aux_downsample_layers" in name:
            conv2_found = True
            module.in_channels = in_channel2
            weight = module.weight.detach()
            reset = False

            if in_channel2 == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel2 == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel2):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel2)

            module.weight = nn.parameter.Parameter(weight)

        if conv1_found and conv2_found:
            break


def patch_first_conv_vitae(model, in_channel1, in_channel2):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    conv1_found = False
    conv2_found = False
    conv3_found = False

    for name, module in model.named_modules():
        if not conv1_found and isinstance(module, nn.Conv2d) and "layers.0.RC.PRM" in name:
            print(name)
            conv1_found = True
            module.in_channels = in_channel1
            weight = module.weight.detach()
            reset = False

            if in_channel1 == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel1 == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel1):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel1)

            module.weight = nn.parameter.Parameter(weight)
        if not conv3_found and isinstance(module, nn.Conv2d) and "layers.0.RC.PCM" in name:
            print(name)
            conv3_found = True
            module.in_channels = in_channel1
            weight = module.weight.detach()
            reset = False

            if in_channel1 == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel1 == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel1):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel1)

            module.weight = nn.parameter.Parameter(weight)
        if not conv2_found and isinstance(module, nn.Conv2d) and "aux_layers.0.RC.PRM" in name:
            print(name)
            conv2_found = True
            module.in_channels = in_channel2
            weight = module.weight.detach()
            reset = False

            if in_channel2 == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel2 == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel2):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel2)

            module.weight = nn.parameter.Parameter(weight)

        if conv1_found and conv2_found and conv3_found:
            break


def patch_first_conv_SMT(model, in_channel1, in_channel2):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    conv1_found = False
    conv2_found = False

    for name, module in model.named_modules():
        if not conv1_found and isinstance(module, nn.Conv2d) and "patch_embed1" in name:
            conv1_found = True
            module.in_channels = in_channel1
            weight = module.weight.detach()
            reset = False

            if in_channel1 == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel1 == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel1):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel1)

            module.weight = nn.parameter.Parameter(weight)

        if not conv2_found and isinstance(module, nn.Conv2d) and "aux_patch_embed1" in name:
            conv2_found = True
            module.in_channels = in_channel2
            weight = module.weight.detach()
            reset = False

            if in_channel2 == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel2 == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel2):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel2)

            module.weight = nn.parameter.Parameter(weight)

        if conv1_found and conv2_found:
            break


def patch_first_conv_EMO(model, in_channel1, in_channel2):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    conv1_found = False
    conv2_found = False

    for name, module in model.named_modules():
        if not conv1_found and isinstance(module, nn.Conv2d) and "stage0" in name:
            conv1_found = True
            module.in_channels = in_channel1
            weight = module.weight.detach()
            reset = False

            if in_channel1 == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel1 == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel1):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel1)

            module.weight = nn.parameter.Parameter(weight)

        if not conv2_found and isinstance(module, nn.Conv2d) and "aux_stage0" in name:
            conv2_found = True
            module.in_channels = in_channel2
            weight = module.weight.detach()
            reset = False

            if in_channel2 == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel2 == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel2):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel2)

            module.weight = nn.parameter.Parameter(weight)

        if conv1_found and conv2_found:
            break


def patch_first_conv_resnet(model, in_channel1, in_channel2):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    conv1_found = False
    conv2_found = False

    for name, module in model.named_modules():
        if not conv1_found and isinstance(module, nn.Conv2d) and "conv1" in name:
            conv1_found = True
            module.in_channels = in_channel1
            weight = module.weight.detach()
            reset = False

            if in_channel1 == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel1 == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel1):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel1)

            module.weight = nn.parameter.Parameter(weight)

        if not conv2_found and isinstance(module, nn.Conv2d) and "extra_conv1" in name:
            conv2_found = True
            module.in_channels = in_channel2
            weight = module.weight.detach()
            reset = False

            if in_channel2 == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel2 == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel2):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel2)

            module.weight = nn.parameter.Parameter(weight)

        if conv1_found and conv2_found:
            break


def patch_first_conv_mobilenet(model, in_channel1, in_channel2):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    conv1_found = False
    conv2_found = False

    for name, module in model.named_modules():
        if not conv1_found and isinstance(module, nn.Conv2d) and "features" in name:
            conv1_found = True
            module.in_channels = in_channel1
            weight = module.weight.detach()
            reset = False

            if in_channel1 == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel1 == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel1):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel1)

            module.weight = nn.parameter.Parameter(weight)

        if not conv2_found and isinstance(module, nn.Conv2d) and "aux_features" in name:
            conv2_found = True
            module.in_channels = in_channel2
            weight = module.weight.detach()
            reset = False

            if in_channel2 == 1:
                weight = weight.sum(1, keepdim=True)
            elif in_channel2 == 2:
                weight[:, 0] = weight[:, 0] + 0.5 * weight[:, 1]
                weight[:, 1] = weight[:, 2] + 0.5 * weight[:, 1]
                weight = weight[:, :2]
            else:
                for i in range(3, in_channel2):
                    weight = torch.cat((weight, weight[:, (i % 3):(i % 3 + 1)]), dim=1)
                weight *= (3 / in_channel2)

            module.weight = nn.parameter.Parameter(weight)

        if conv1_found and conv2_found:
            break
