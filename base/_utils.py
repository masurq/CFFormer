import torch
import torch.nn as nn


def patch_first_conv(model, in_channel1, in_channel2):
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

        if not conv2_found and isinstance(module, nn.Conv2d) and "hha_conv1" in name:
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
