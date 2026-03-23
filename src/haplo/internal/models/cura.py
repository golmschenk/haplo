from __future__ import annotations

from typing import Self

from torch import Tensor
from torch.nn import Module, ModuleList, Conv1d, LeakyReLU, Identity

from haplo.internal.models.residual_generation_light_curve_network_block import ResidualGenerationLightCurveNetworkBlock


class Cura(Module):
    @classmethod
    def new(cls, number_of_input_features: int = 11, input_transformation: Module | None = None,
            output_transformation: Module | None = None) -> Self:
        """
        Constructor for the model.

        :param number_of_input_features: The number of input features.
        :param input_transformation: The transformation to be applied to the input data.
        :param output_transformation: The transformation to be applied to the output data.
        :return: An instance of the network model.
        """
        if input_transformation is None:
            input_transformation = Identity()
        if output_transformation is None:
            output_transformation = Identity()
        instance = cls(number_of_input_features=number_of_input_features, input_transformation=input_transformation,
                       output_transformation=output_transformation)
        return instance

    def __init__(self, number_of_input_features: int, input_transformation: Module, output_transformation: Module):
        super().__init__()
        self.number_of_input_features: int = number_of_input_features
        self.input_transformation: Module = input_transformation
        self.output_transformation: Module = output_transformation

        self.blocks = ModuleList()
        self.dense0 = Conv1d(self.number_of_input_features, 400, kernel_size=1)
        self.activation = LeakyReLU()
        self.dense1 = Conv1d(self.dense0.out_channels, 400, kernel_size=1)
        output_channels = 128
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(input_channels=400, output_channels=output_channels,
                                                                    batch_normalization=False, dropout_rate=0.0,
                                                                    activation_type=LeakyReLU))
        input_channels = output_channels
        for output_channels in [512, 512, 1024, 1024, 2048, 2048]:
            self.blocks.append(
                ResidualGenerationLightCurveNetworkBlock(input_channels=input_channels, output_channels=output_channels,
                                                         upsampling_scale_factor=2, batch_normalization=False,
                                                         dropout_rate=0.0, activation_type=LeakyReLU))
            input_channels = output_channels
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(input_channels=input_channels,
                                                                            output_channels=output_channels,
                                                                            batch_normalization=False, dropout_rate=0.0,
                                                                            activation_type=LeakyReLU))
                input_channels = output_channels
        self.end_conv = Conv1d(input_channels, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        The forward pass of the model.

        :param x: The input data to infer on.
        :return: The network prediction.
        """
        x = self.input_transformation(x)
        x = x.reshape([-1, self.number_of_input_features, 1])
        x = self.dense0(x)
        x = self.activation(x)
        x = self.dense1(x)
        x = self.activation(x)
        for index, block in enumerate(self.blocks):
            x = block(x)
        x = self.end_conv(x)
        x = x.reshape([-1, 64])
        x = self.output_transformation(x)
        return x
