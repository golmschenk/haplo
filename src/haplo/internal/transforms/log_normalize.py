import torch
from torch import Tensor
from torch.nn import Module


class LogTransform(Module):
    @classmethod
    def new(cls):
        return cls()

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.log(x)


class ExpTransform(Module):
    @classmethod
    def new(cls):
        return cls()

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.log(x)
