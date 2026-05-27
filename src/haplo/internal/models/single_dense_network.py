import torch
from torch.nn import Module, Linear, LeakyReLU


class SingleDenseNetwork(Module):
    def __init__(self):
        super().__init__()
        self.dense = Linear(11, 64)
        self.activation = LeakyReLU()

    def forward(self, x):
        x = self.dense(x)
        x = self.activation(x)
        return x


class SingleDenseNetworkWith2dOutput(Module):
    def __init__(self):
        super().__init__()
        self.dense = Linear(11, 100 * 100)
        self.activation = LeakyReLU()

    def forward(self, x):
        x = self.dense(x)
        x = self.activation(x)
        x = torch.reshape(x, [-1, 100, 100])
        return x
