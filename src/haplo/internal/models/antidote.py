from __future__ import annotations

import math
from typing import Self

import torch
from torch.nn.init import xavier_uniform_, uniform_
from torch import permute
from torch.nn import Module, ModuleList, Conv1d, LeakyReLU, Identity, GELU, BatchNorm1d, ConvTranspose1d, Upsample, \
    ConstantPad1d, Dropout1d, ReLU

from haplo.internal.models.legacy_models import ResidualGenerationLightCurveNetworkBlock


class AntidotePrototype0(Module):
    @classmethod
    def new(cls, input_features_shape: int = 11, input_transformation: Module | None = None,
            output_transformation: Module | None = None) -> Self:
        """
        Constructor for the model.

        :param input_features_shape: The shape of input features.
        :param input_transformation: The transformation to be applied to the input data.
        :param output_transformation: The transformation to be applied to the output data.
        :return: An instance of the network model.
        """
        if input_transformation is None:
            input_transformation = Identity()
        if output_transformation is None:
            output_transformation = Identity()
        instance = cls(input_features_shape=input_features_shape, input_transformation=input_transformation,
                       output_transformation=output_transformation)
        return instance

    def __init__(self, input_features_shape: int, input_transformation: Module, output_transformation: Module):
        super().__init__()
        self.input_features: int = input_features_shape
        self.input_transformation: Module = input_transformation
        self.output_transformation: Module = output_transformation

        self.blocks = ModuleList()
        self.dense0 = Conv1d(self.input_features, 400, kernel_size=1)
        self.activation = LeakyReLU()
        self.dense1 = Conv1d(self.dense0.out_channels, 400, kernel_size=1)
        output_channels = 128
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=400, dropout_rate=0.0,
            batch_normalization=False))
        input_channels = output_channels
        for output_channels in [512, 512, 256, 128, 64, 32]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, upsampling_scale_factor=2,
                dropout_rate=0.0,
                batch_normalization=False))
            input_channels = output_channels
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                    input_channels=input_channels, output_channels=output_channels, dropout_rate=0.0,
                    batch_normalization=False))
                input_channels = output_channels
        self.end_conv = Conv1d(input_channels, 1, kernel_size=1)

    def forward(self, x):
        """
        The forward pass of the model.

        :param x: The input data to infer on.
        :return: The network prediction.
        """
        x = self.input_transformation(x)
        x = x.reshape([-1, self.input_features, 1])
        x = self.dense0(x)
        x = self.activation(x)
        x = self.dense1(x)
        x = self.activation(x)
        for index, block in enumerate(self.blocks):
            x = block(x)
        x = self.end_conv(x)
        outputs = x.reshape([-1, 64])
        x = self.output_transformation(x)
        return outputs



class AntidotePrototype1(Module):
    @classmethod
    def new(cls, input_features_shape: int = 11, input_transformation: Module | None = None,
            output_transformation: Module | None = None) -> Self:
        """
        Constructor for the model.

        :param input_features_shape: The shape of input features.
        :param input_transformation: The transformation to be applied to the input data.
        :param output_transformation: The transformation to be applied to the output data.
        :return: An instance of the network model.
        """
        if input_transformation is None:
            input_transformation = Identity()
        if output_transformation is None:
            output_transformation = Identity()
        instance = cls(input_features_shape=input_features_shape, input_transformation=input_transformation,
                       output_transformation=output_transformation)
        return instance

    def __init__(self, input_features_shape: int, input_transformation: Module, output_transformation: Module):
        super().__init__()
        self.input_features: int = input_features_shape
        self.input_transformation: Module = input_transformation
        self.output_transformation: Module = output_transformation

        self.blocks = ModuleList()
        self.dense0 = Conv1d(self.input_features, 400, kernel_size=1)
        self.activation = LeakyReLU()
        self.dense1 = Conv1d(self.dense0.out_channels, 400, kernel_size=1)
        output_channels = 128
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=400, dropout_rate=0.0,
            batch_normalization=False))
        input_channels = output_channels
        for output_channels in [512, 512, 256, 128, 64, 32]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, upsampling_scale_factor=2,
                dropout_rate=0.0,
                batch_normalization=False))
            input_channels = output_channels
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                    input_channels=input_channels, output_channels=output_channels, dropout_rate=0.0,
                    batch_normalization=False))
                input_channels = output_channels
        self.end_conv = Conv1d(input_channels, 1, kernel_size=1)


    def forward(self, x):
        """
        The forward pass of the model.

        :param x: The input data to infer on.
        :return: The network prediction.
        """
        x = self.input_transformation(x)
        x = x.reshape([-1, self.input_features, 1])
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



class AntidotePrototype2(Module):
    @classmethod
    def new(cls, input_features_shape: int = 11, input_transformation: Module | None = None,
            output_transformation: Module | None = None) -> Self:
        """
        Constructor for the model.

        :param input_features_shape: The shape of input features.
        :param input_transformation: The transformation to be applied to the input data.
        :param output_transformation: The transformation to be applied to the output data.
        :return: An instance of the network model.
        """
        if input_transformation is None:
            input_transformation = Identity()
        if output_transformation is None:
            output_transformation = Identity()
        instance = cls(input_features_shape=input_features_shape, input_transformation=input_transformation,
                       output_transformation=output_transformation)
        return instance

    def __init__(self, input_features_shape: int, input_transformation: Module, output_transformation: Module):
        super().__init__()
        self.input_features: int = input_features_shape
        self.input_transformation: Module = input_transformation
        self.output_transformation: Module = output_transformation

        self.blocks = ModuleList()
        self.dense0 = Conv1d(self.input_features, 400, kernel_size=1)
        self.activation = LeakyReLU()
        self.dense1 = Conv1d(self.dense0.out_channels, 400, kernel_size=1)
        output_channels = 128
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=self.dense1.out_channels, dropout_rate=0.0,
            batch_normalization=False))
        input_channels = output_channels
        for output_channels in [512, 512, 256, 128, 64, 32]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, upsampling_scale_factor=2,
                dropout_rate=0.0,
                batch_normalization=False))
            input_channels = output_channels
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                    input_channels=input_channels, output_channels=output_channels, dropout_rate=0.0,
                    batch_normalization=False))
                input_channels = output_channels
        self.end_conv0 = Conv1d(input_channels, 20, kernel_size=1)
        self.end_conv1 = Conv1d(self.end_conv0.out_channels, 10, kernel_size=1)
        self.end_conv2 = Conv1d(self.end_conv1.out_channels, 1, kernel_size=1)

    def forward(self, x):
        """
        The forward pass of the model.

        :param x: The input data to infer on.
        :return: The network prediction.
        """
        x = self.input_transformation(x)
        x = x.reshape([-1, self.input_features, 1])
        x = self.dense0(x)
        x = self.activation(x)
        x = self.dense1(x)
        x = self.activation(x)
        for index, block in enumerate(self.blocks):
            x = block(x)
        x = self.end_conv0(x)
        x = self.activation(x)
        x = self.end_conv1(x)
        x = self.activation(x)
        x = self.end_conv2(x)
        x = x.reshape([-1, 64])
        x = self.output_transformation(x)
        return x


class GeluResidualGenerationLightCurveNetworkBlock(Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3,
                 upsampling_scale_factor: float = 1, batch_normalization: bool = False, dropout_rate: float = 0.0,
                 renorm: bool = False):
        super().__init__()
        self.activation = GELU()
        dimension_decrease_factor = 4
        if batch_normalization:
            self.batch_normalization = BatchNorm1d(num_features=input_channels, track_running_stats=renorm)
        else:
            self.batch_normalization = None
        reduced_channels = output_channels // dimension_decrease_factor
        self.dimension_decrease_layer = ConvTranspose1d(
            in_channels=input_channels, out_channels=reduced_channels, kernel_size=1)
        self.convolutional_layer = ConvTranspose1d(
            in_channels=reduced_channels, out_channels=reduced_channels, kernel_size=kernel_size,
            padding=math.floor(kernel_size / 2)
        )
        self.dimension_increase_layer = ConvTranspose1d(
            in_channels=reduced_channels, out_channels=output_channels, kernel_size=1)
        if upsampling_scale_factor > 1:
            self.upsampling_layer = Upsample(scale_factor=upsampling_scale_factor)
        else:
            self.upsampling_layer = None
        self.input_to_output_channel_difference = input_channels - output_channels
        if output_channels != input_channels:
            if output_channels < input_channels:
                self.input_channels = input_channels
                self.output_channels = output_channels
            else:
                self.dimension_change_layer = ConstantPad1d(padding=(0, -self.input_to_output_channel_difference),
                                                            value=0)
        else:
            self.dimension_change_layer = None
        if dropout_rate > 0:
            self.dropout_layer = Dropout1d(p=dropout_rate)
        else:
            self.dropout_layer = None

    def forward(self, x):
        """
        The forward pass of the block.

        :param x: The input tensor.
        :return: The output tensor of the layer.
        """
        y = x
        if self.batch_normalization is not None:
            y = self.batch_normalization(y)
        y = self.dimension_decrease_layer(y)
        y = self.activation(y)
        y = self.convolutional_layer(y)
        y = self.activation(y)
        y = self.dimension_increase_layer(y)
        y = self.activation(y)
        if self.upsampling_layer is not None:
            x = self.upsampling_layer(x)
            y = self.upsampling_layer(y)
        if self.input_to_output_channel_difference != 0:
            x = permute(x, (0, 2, 1))
            if self.input_to_output_channel_difference < 0:
                x = self.dimension_change_layer(x)
            else:
                x = x[:, :, 0:self.output_channels]
                # x = interpolate(x, [self.output_channels], mode='linear')
            x = permute(x, (0, 2, 1))
        if self.dropout_layer is not None:
            y = self.dropout_layer(y)
        return x + y


class AntidotePrototype3(Module):
    """
    P1 with GELU
    """
    @classmethod
    def new(cls, input_features_shape: int = 11, input_transformation: Module | None = None,
            output_transformation: Module | None = None) -> Self:
        """
        Constructor for the model.

        :param input_features_shape: The shape of input features.
        :param input_transformation: The transformation to be applied to the input data.
        :param output_transformation: The transformation to be applied to the output data.
        :return: An instance of the network model.
        """
        if input_transformation is None:
            input_transformation = Identity()
        if output_transformation is None:
            output_transformation = Identity()
        instance = cls(input_features_shape=input_features_shape, input_transformation=input_transformation,
                       output_transformation=output_transformation)
        return instance

    def __init__(self, input_features_shape: int, input_transformation: Module, output_transformation: Module):
        super().__init__()
        self.input_features: int = input_features_shape
        self.input_transformation: Module = input_transformation
        self.output_transformation: Module = output_transformation

        self.blocks = ModuleList()
        self.dense0 = Conv1d(self.input_features, 400, kernel_size=1)
        self.activation = GELU()
        self.dense1 = Conv1d(self.dense0.out_channels, 400, kernel_size=1)
        output_channels = 128
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=400, dropout_rate=0.0,
            batch_normalization=False))
        input_channels = output_channels
        for output_channels in [512, 512, 256, 128, 64, 32]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, upsampling_scale_factor=2,
                dropout_rate=0.0,
                batch_normalization=False))
            input_channels = output_channels
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                    input_channels=input_channels, output_channels=output_channels, dropout_rate=0.0,
                    batch_normalization=False))
                input_channels = output_channels
        self.end_conv = Conv1d(input_channels, 1, kernel_size=1)

    def forward(self, x):
        """
        The forward pass of the model.

        :param x: The input data to infer on.
        :return: The network prediction.
        """
        x = self.input_transformation(x)
        x = x.reshape([-1, self.input_features, 1])
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


class ReluResidualGenerationLightCurveNetworkBlock(Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3,
                 upsampling_scale_factor: float = 1, batch_normalization: bool = False, dropout_rate: float = 0.0,
                 renorm: bool = False):
        super().__init__()
        self.activation = ReLU()
        dimension_decrease_factor = 4
        if batch_normalization:
            self.batch_normalization = BatchNorm1d(num_features=input_channels, track_running_stats=renorm)
        else:
            self.batch_normalization = None
        reduced_channels = output_channels // dimension_decrease_factor
        self.dimension_decrease_layer = ConvTranspose1d(
            in_channels=input_channels, out_channels=reduced_channels, kernel_size=1)
        self.convolutional_layer = ConvTranspose1d(
            in_channels=reduced_channels, out_channels=reduced_channels, kernel_size=kernel_size,
            padding=math.floor(kernel_size / 2)
        )
        self.dimension_increase_layer = ConvTranspose1d(
            in_channels=reduced_channels, out_channels=output_channels, kernel_size=1)
        if upsampling_scale_factor > 1:
            self.upsampling_layer = Upsample(scale_factor=upsampling_scale_factor)
        else:
            self.upsampling_layer = None
        self.input_to_output_channel_difference = input_channels - output_channels
        if output_channels != input_channels:
            if output_channels < input_channels:
                self.input_channels = input_channels
                self.output_channels = output_channels
            else:
                self.dimension_change_layer = ConstantPad1d(padding=(0, -self.input_to_output_channel_difference),
                                                            value=0)
        else:
            self.dimension_change_layer = None
        if dropout_rate > 0:
            self.dropout_layer = Dropout1d(p=dropout_rate)
        else:
            self.dropout_layer = None

    def forward(self, x):
        """
        The forward pass of the block.

        :param x: The input tensor.
        :return: The output tensor of the layer.
        """
        y = x
        if self.batch_normalization is not None:
            y = self.batch_normalization(y)
        y = self.dimension_decrease_layer(y)
        y = self.activation(y)
        y = self.convolutional_layer(y)
        y = self.activation(y)
        y = self.dimension_increase_layer(y)
        y = self.activation(y)
        if self.upsampling_layer is not None:
            x = self.upsampling_layer(x)
            y = self.upsampling_layer(y)
        if self.input_to_output_channel_difference != 0:
            x = permute(x, (0, 2, 1))
            if self.input_to_output_channel_difference < 0:
                x = self.dimension_change_layer(x)
            else:
                x = x[:, :, 0:self.output_channels]
                # x = interpolate(x, [self.output_channels], mode='linear')
            x = permute(x, (0, 2, 1))
        if self.dropout_layer is not None:
            y = self.dropout_layer(y)
        return x + y


class AntidotePrototype4(Module):
    """
    P1 with ReLU
    """
    @classmethod
    def new(cls, input_features_shape: int = 11, input_transformation: Module | None = None,
            output_transformation: Module | None = None) -> Self:
        """
        Constructor for the model.

        :param input_features_shape: The shape of input features.
        :param input_transformation: The transformation to be applied to the input data.
        :param output_transformation: The transformation to be applied to the output data.
        :return: An instance of the network model.
        """
        if input_transformation is None:
            input_transformation = Identity()
        if output_transformation is None:
            output_transformation = Identity()
        instance = cls(input_features_shape=input_features_shape, input_transformation=input_transformation,
                       output_transformation=output_transformation)
        return instance

    def __init__(self, input_features_shape: int, input_transformation: Module, output_transformation: Module):
        super().__init__()
        self.input_features: int = input_features_shape
        self.input_transformation: Module = input_transformation
        self.output_transformation: Module = output_transformation

        self.blocks = ModuleList()
        self.dense0 = Conv1d(self.input_features, 400, kernel_size=1)
        self.activation = ReLU()
        self.dense1 = Conv1d(self.dense0.out_channels, 400, kernel_size=1)
        output_channels = 128
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=400, dropout_rate=0.0,
            batch_normalization=False))
        input_channels = output_channels
        for output_channels in [512, 512, 256, 128, 64, 32]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, upsampling_scale_factor=2,
                dropout_rate=0.0,
                batch_normalization=False))
            input_channels = output_channels
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                    input_channels=input_channels, output_channels=output_channels, dropout_rate=0.0,
                    batch_normalization=False))
                input_channels = output_channels
        self.end_conv = Conv1d(input_channels, 1, kernel_size=1)

    def forward(self, x):
        """
        The forward pass of the model.

        :param x: The input data to infer on.
        :return: The network prediction.
        """
        x = self.input_transformation(x)
        x = x.reshape([-1, self.input_features, 1])
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



class AntidotePrototype5(Module):
    """
    P1 with GELU
    """
    @classmethod
    def new(cls, input_features_shape: int = 11, input_transformation: Module | None = None,
            output_transformation: Module | None = None) -> Self:
        """
        Constructor for the model.

        :param input_features_shape: The shape of input features.
        :param input_transformation: The transformation to be applied to the input data.
        :param output_transformation: The transformation to be applied to the output data.
        :return: An instance of the network model.
        """
        if input_transformation is None:
            input_transformation = Identity()
        if output_transformation is None:
            output_transformation = Identity()
        instance = cls(input_features_shape=input_features_shape, input_transformation=input_transformation,
                       output_transformation=output_transformation)
        return instance

    def __init__(self, input_features_shape: int, input_transformation: Module, output_transformation: Module):
        super().__init__()
        self.input_features: int = input_features_shape
        self.input_transformation: Module = input_transformation
        self.output_transformation: Module = output_transformation

        self.blocks = ModuleList()
        self.dense0 = Conv1d(self.input_features, 400, kernel_size=1)
        self.activation = GELU()
        self.dense1 = Conv1d(self.dense0.out_channels, 400, kernel_size=1)
        output_channels = 128
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=400, dropout_rate=0.0,
            batch_normalization=False))
        input_channels = output_channels
        for output_channels in [512, 512, 256, 128, 64, 32]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, upsampling_scale_factor=2,
                dropout_rate=0.0,
                batch_normalization=True))
            input_channels = output_channels
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                    input_channels=input_channels, output_channels=output_channels, dropout_rate=0.0,
                    batch_normalization=True))
                input_channels = output_channels
        self.end_conv = Conv1d(input_channels, 1, kernel_size=1)

    def forward(self, x):
        """
        The forward pass of the model.

        :param x: The input data to infer on.
        :return: The network prediction.
        """
        x = self.input_transformation(x)
        x = x.reshape([-1, self.input_features, 1])
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


class AntidotePrototype6(Module):
    """
    P1 with GELU
    """
    @classmethod
    def new(cls, input_features_shape: int = 11, input_transformation: Module | None = None,
            output_transformation: Module | None = None) -> Self:
        """
        Constructor for the model.

        :param input_features_shape: The shape of input features.
        :param input_transformation: The transformation to be applied to the input data.
        :param output_transformation: The transformation to be applied to the output data.
        :return: An instance of the network model.
        """
        if input_transformation is None:
            input_transformation = Identity()
        if output_transformation is None:
            output_transformation = Identity()
        instance = cls(input_features_shape=input_features_shape, input_transformation=input_transformation,
                       output_transformation=output_transformation)
        return instance

    def __init__(self, input_features_shape: int, input_transformation: Module, output_transformation: Module):
        super().__init__()
        self.input_features: int = input_features_shape
        self.input_transformation: Module = input_transformation
        self.output_transformation: Module = output_transformation

        self.blocks = ModuleList()
        self.dense0 = Conv1d(self.input_features, 400, kernel_size=1)
        self.activation = GELU()
        self.dense1 = Conv1d(self.dense0.out_channels, 400, kernel_size=1)
        output_channels = 128
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=400, dropout_rate=0.0,
            batch_normalization=False))
        input_channels = output_channels
        for output_channels in [512, 512, 256, 128, 64, 32]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, upsampling_scale_factor=2,
                dropout_rate=0.1,
                batch_normalization=False))
            input_channels = output_channels
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                    input_channels=input_channels, output_channels=output_channels, dropout_rate=0.1,
                    batch_normalization=False))
                input_channels = output_channels
        self.end_conv = Conv1d(input_channels, 1, kernel_size=1)

    def forward(self, x):
        """
        The forward pass of the model.

        :param x: The input data to infer on.
        :return: The network prediction.
        """
        x = self.input_transformation(x)
        x = x.reshape([-1, self.input_features, 1])
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


class AntidotePrototype7(Module):
    @classmethod
    def new(cls, input_features_shape: int = 11, input_transformation: Module | None = None,
            output_transformation: Module | None = None) -> Self:
        """
        Constructor for the model.

        :param input_features_shape: The shape of input features.
        :param input_transformation: The transformation to be applied to the input data.
        :param output_transformation: The transformation to be applied to the output data.
        :return: An instance of the network model.
        """
        if input_transformation is None:
            input_transformation = Identity()
        if output_transformation is None:
            output_transformation = Identity()
        instance = cls(input_features_shape=input_features_shape, input_transformation=input_transformation,
                       output_transformation=output_transformation)
        return instance

    def __init__(self, input_features_shape: int, input_transformation: Module, output_transformation: Module):
        super().__init__()
        self.input_features: int = input_features_shape
        self.input_transformation: Module = input_transformation
        self.output_transformation: Module = output_transformation

        self.blocks = ModuleList()
        self.dense0 = Conv1d(self.input_features, 400, kernel_size=1)
        self.do0 = Dropout1d(p=0.1)
        self.activation = GELU()
        self.dense1 = Conv1d(self.dense0.out_channels, 400, kernel_size=1)
        self.do1 = Dropout1d(p=0.1)
        output_channels = 128
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=400, dropout_rate=0.1,
            batch_normalization=False))
        input_channels = output_channels
        for output_channels in [512, 512, 256, 128, 64, 32]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, upsampling_scale_factor=2,
                dropout_rate=0.1,
                batch_normalization=False))
            input_channels = output_channels
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                    input_channels=input_channels, output_channels=output_channels, dropout_rate=0.1,
                    batch_normalization=False))
                input_channels = output_channels
        self.end_conv = Conv1d(input_channels, 1, kernel_size=1)

    def forward(self, x):
        """
        The forward pass of the model.

        :param x: The input data to infer on.
        :return: The network prediction.
        """
        x = self.input_transformation(x)
        x = x.reshape([-1, self.input_features, 1])
        x = self.dense0(x)
        x = self.activation(x)
        x = self.do0(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.do1(x)
        for index, block in enumerate(self.blocks):
            x = block(x)
        x = self.end_conv(x)
        x = x.reshape([-1, 64])
        x = self.output_transformation(x)
        return x


class AntidotePrototype8(Module):
    @classmethod
    def new(cls, input_features_shape: int = 11, input_transformation: Module | None = None,
            output_transformation: Module | None = None) -> Self:
        """
        Constructor for the model.

        :param input_features_shape: The shape of input features.
        :param input_transformation: The transformation to be applied to the input data.
        :param output_transformation: The transformation to be applied to the output data.
        :return: An instance of the network model.
        """
        if input_transformation is None:
            input_transformation = Identity()
        if output_transformation is None:
            output_transformation = Identity()
        instance = cls(input_features_shape=input_features_shape, input_transformation=input_transformation,
                       output_transformation=output_transformation)
        return instance

    def __init__(self, input_features_shape: int, input_transformation: Module, output_transformation: Module):
        super().__init__()
        self.input_features: int = input_features_shape
        self.input_transformation: Module = input_transformation
        self.output_transformation: Module = output_transformation

        self.blocks = ModuleList()
        self.dense0 = Conv1d(self.input_features, 400, kernel_size=1)
        self.activation = GELU()
        self.dense1 = Conv1d(self.dense0.out_channels, 400, kernel_size=1)
        output_channels = 128
        self.blocks.append(GeluResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=400, dropout_rate=0.0,
            batch_normalization=False))
        input_channels = output_channels
        for output_channels in [512, 512, 256, 128, 64, 32]:
            self.blocks.append(GeluResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, upsampling_scale_factor=2,
                dropout_rate=0.0,
                batch_normalization=False))
            input_channels = output_channels
            for _ in range(2):
                self.blocks.append(GeluResidualGenerationLightCurveNetworkBlock(
                    input_channels=input_channels, output_channels=output_channels, dropout_rate=0.0,
                    batch_normalization=False))
                input_channels = output_channels
        self.end_conv = Conv1d(input_channels, 1, kernel_size=1)

    def forward(self, x):
        """
        The forward pass of the model.

        :param x: The input data to infer on.
        :return: The network prediction.
        """
        x = self.input_transformation(x)
        x = x.reshape([-1, self.input_features, 1])
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


if __name__ == '__main__':
    x = torch.rand(size=[7, 11])
    model = AntidotePrototype0.new()
    y = model(x)
    pass