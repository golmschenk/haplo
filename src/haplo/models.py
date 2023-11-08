import math


from torch import permute
from torch.nn import Module, Conv1d, LeakyReLU, BatchNorm1d, Upsample, ConstantPad1d, Dropout1d, ModuleList, \
    ConvTranspose1d, Linear


class LiraTraditionalShape8xWidthWithNoDoNoBn(Module):
    def __init__(self):
        super().__init__()
        self.blocks = ModuleList()
        self.dense0 = Conv1d(11, 400, kernel_size=1)
        self.activation = LeakyReLU()
        self.dense1 = Conv1d(self.dense0.out_channels, 400, kernel_size=1)
        output_channels = 128
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=400, dropout_rate=0.0,
            batch_normalization=False))
        input_channels = output_channels
        for output_channels in [512, 512, 1024, 1024, 2048, 2048]:
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
        x = x.reshape([-1, 11, 1])
        x = self.dense0(x)
        x = self.activation(x)
        x = self.dense1(x)
        x = self.activation(x)
        for index, block in enumerate(self.blocks):
            x = block(x)
        x = self.end_conv(x)
        outputs = x.reshape([-1, 64])
        return outputs


class ResidualGenerationLightCurveNetworkBlock(Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3,
                 upsampling_scale_factor: int = 1, batch_normalization: bool = False, dropout_rate: float = 0.0,
                 renorm: bool = False):
        super().__init__()
        self.activation = LeakyReLU()
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
            x = permute(x, (0, 2, 1))
        if self.dropout_layer is not None:
            y = self.dropout_layer(y)
        return x + y


class Cura(Module):
    def __init__(self):
        super().__init__()
        self.blocks = ModuleList()
        self.dense0 = Conv1d(11, 400, kernel_size=1)
        self.activation = LeakyReLU()
        self.dense1 = Conv1d(self.dense0.out_channels, 400, kernel_size=1)
        output_channels = 128
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=400, dropout_rate=0.0,
            batch_normalization=False))
        input_channels = output_channels
        for output_channels in [512, 512, 1024, 1024, 2048, 2048]:
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
        x = x.reshape([-1, 11, 1])
        x = self.dense0(x)
        # x = self.activation(x)
        x = self.dense1(x)
        x = self.activation(x)
        for index, block in enumerate(self.blocks):
            x = block(x)
        x = self.end_conv(x)
        outputs = x.reshape([-1, 64])
        return outputs


class LiraTraditionalShape8xWidthWith0d5DoNoBnOldFirstLayers(Module):
    def __init__(self):
        super().__init__()
        self.blocks = ModuleList()
        self.dense0 = Conv1d(11, 400, kernel_size=1)
        self.activation = LeakyReLU()
        self.dense1 = Conv1d(self.dense0.out_channels, 400, kernel_size=1)
        output_channels = 128
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=400, dropout_rate=0.5,
            batch_normalization=False))
        input_channels = output_channels
        for output_channels in [512, 512, 1024, 1024, 2048, 2048]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, upsampling_scale_factor=2,
                dropout_rate=0.5,
                batch_normalization=False))
            input_channels = output_channels
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                    input_channels=input_channels, output_channels=output_channels, dropout_rate=0.5,
                    batch_normalization=False))
                input_channels = output_channels
        self.end_conv = Conv1d(input_channels, 1, kernel_size=1)

    def forward(self, x):
        x = x.reshape([-1, 11, 1])
        x = self.dense0(x)
        x = self.activation(x)
        x = self.dense1(x)
        x = self.activation(x)
        for index, block in enumerate(self.blocks):
            x = block(x)
        x = self.end_conv(x)
        outputs = x.reshape([-1, 64])
        return outputs


class SingleDenseNetwork(Module):
    def __init__(self):
        super().__init__()
        self.dense = Linear(11, 64)
        self.activation = LeakyReLU()

    def forward(self, x):
        x = self.dense(x)
        x = self.activation(x)
        return x
