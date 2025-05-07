import math

from torch import permute
from torch.nn import LeakyReLU, BatchNorm1d, ConvTranspose1d, ConstantPad1d, Dropout1d, Upsample, ModuleList, Conv1d, \
    Module, Conv2d, Dropout2d, ConstantPad2d, ConvTranspose2d, BatchNorm2d



class ResidualGenerationLightCurveNetworkBlock2D(Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3,
                 upsampling_scale_factor: int = 1, batch_normalization: bool = False, dropout_rate: float = 0.0,
                 renorm: bool = False):
        super().__init__()
        self.activation = LeakyReLU()
        dimension_decrease_factor = 4
        if batch_normalization:
            self.batch_normalization = BatchNorm2d(num_features=input_channels, track_running_stats=renorm)
        else:
            self.batch_normalization = None
        reduced_channels = output_channels // dimension_decrease_factor
        self.dimension_decrease_layer = ConvTranspose2d(
            in_channels=input_channels, out_channels=reduced_channels, kernel_size=1)
        self.convolutional_layer = ConvTranspose2d(
            in_channels=reduced_channels, out_channels=reduced_channels, kernel_size=kernel_size,
            padding=math.floor(kernel_size / 2)
        )
        self.dimension_increase_layer = ConvTranspose2d(
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
                self.dimension_change_layer = ConstantPad2d(padding=(0, -self.input_to_output_channel_difference,0,0),
                                                            value=0)
        else:
            self.dimension_change_layer = None
        if dropout_rate > 0:
            self.dropout_layer = Dropout2d(p=dropout_rate)
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
            x = permute(x, (0, 3, 2, 1))
            if self.input_to_output_channel_difference < 0:
                x = self.dimension_change_layer(x)
            else:
                x = x[:, :, :, 0:self.output_channels]
            x = permute(x, (0, 3, 2, 1))
        if self.dropout_layer is not None:
            y = self.dropout_layer(y)
        return x + y

class Cura2D(Module):
    def __init__(self, input_features: int = 11):
        super().__init__()
        self.input_features = input_features
        self.blocks = ModuleList()
        self.dense0 = Conv2d(self.input_features, 400, kernel_size=1)
        self.activation = LeakyReLU()
        self.dense1 = Conv2d(self.dense0.out_channels, 400, kernel_size=1)
        output_channels = 128
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock2D(
            output_channels=output_channels, input_channels=400, dropout_rate=0.0,
            batch_normalization=False))
        input_channels = output_channels
        for output_channels in [512, 512, 1024, 1024, 2048, 2048]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock2D(
                output_channels=output_channels, input_channels=input_channels, upsampling_scale_factor=2.25,
                dropout_rate=0.0,
                batch_normalization=False))
            input_channels = output_channels
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock2D(
                    input_channels=input_channels, output_channels=output_channels, dropout_rate=0.0,
                    batch_normalization=False))
                input_channels = output_channels
        self.end_conv = Conv2d(input_channels, 1, kernel_size=1)


    def forward(self, x):
        x = x.reshape([-1, self.input_features, 1, 1])
        x = self.dense0(x)
        # x = self.activation(x)
        x = self.dense1(x)
        x = self.activation(x)
        for index, block in enumerate(self.blocks):
            x = block(x)
        x = self.end_conv(x)
        x = x[:, :, :100, :100]
        outputs = x.reshape([-1, 100, 100])

        return outputs


# def count_parameters(model):
#     """
#     Count the number of trainable parameters in a model.

#     Args:
#         model: PyTorch model

#     Returns:
#         int: Number of trainable parameters
#     """
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def compare_model_parameters():
#     """
#     Compare the number of parameters between Cura2D and Cura100 models.

#     Returns:
#         dict: Dictionary containing parameter counts for both models
#     """
#     from Cura1D_100 import Cura100
#     from haplo.models import Cura

#     model_2d = Cura2D()
#     model_1d = Cura100()
#     haplo_cura = Cura()

#     params_2d = count_parameters(model_2d)
#     params_1d = count_parameters(model_1d)
#     params_haplo = count_parameters(haplo_cura)

#     comparison = {
#         "Cura2D parameters": params_2d,
#         "Cura100 parameters": params_1d,
#         "Haplo Cura parameters": params_haplo,
#         "Difference (2D-1D)": params_2d - params_1d,
#         "Ratio 2D/1D": params_2d / params_1d if params_1d > 0 else float('inf'),
#         "Ratio 2D/Haplo": params_2d / params_haplo if params_haplo > 0 else float('inf'),
#         "Ratio 1D/Haplo": params_1d / params_haplo if params_haplo > 0 else float('inf')
#     }

#     print(f"Cura2D parameters: {params_2d:,}")
#     print(f"Cura100 parameters: {params_1d:,}")
#     print(f"Haplo Cura parameters: {params_haplo:,}")
#     print(f"Difference (2D-1D): {params_2d - params_1d:,}")
#     print(f"Ratio 2D/1D: {params_2d / params_1d:.2f}x")
#     print(f"Ratio 2D/Haplo: {params_2d / params_haplo:.2f}x")
#     print(f"Ratio 1D/Haplo: {params_1d / params_haplo:.2f}x")

#     return comparison


# if __name__ == "__main__":
#     compare_model_parameters()

