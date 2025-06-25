import torch.nn as nn
from torch import Tensor


class VerboseExecution(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Register a hook for each layer
        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layers, _, outputs: print(f"{layers.__name__}：{outputs.shape}")
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
