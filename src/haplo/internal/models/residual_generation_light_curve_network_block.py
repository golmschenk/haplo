import math
from torch import permute
from torch.nn import Module, GELU, BatchNorm1d, ConvTranspose1d, Upsample, ConstantPad1d, Dropout1d


class ResidualGenerationLightCurveNetworkBlock(Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3,
                 upsampling_scale_factor: float = 1, batch_normalization: bool = False, dropout_rate: float = 0.0,
                 renorm: bool = False, activation_type: type[Module]=GELU):
        super().__init__()
        self.activation = activation_type()
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
            x = permute(x, (0, 2, 1))
        if self.dropout_layer is not None:
            y = self.dropout_layer(y)
        return x + y
