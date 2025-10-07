from __future__ import annotations

from torch.nn import Module, ModuleList, Conv1d, LeakyReLU, Identity

from haplo.internal.models.legacy_models import ResidualGenerationLightCurveNetworkBlock
from haplo.internal.transforms.affine_normalize import default_input_affine_transform, default_output_affine_transform


class Cura(Module):
    @classmethod
    def new(cls, number_of_input_features: int = 11, input_transformation: Module | None = None,
            output_transformation: Module | None = None):
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
