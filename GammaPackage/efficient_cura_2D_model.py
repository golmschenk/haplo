from torch.nn import ModuleList, Conv2d, LeakyReLU, Module

from GammaPackage.cura_2D_model import ResidualGenerationLightCurveNetworkBlock2D


class SelectiveThetaComputeCura2D(Module):
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

    def forward(self, x, theta_bin):
        x = x[:11]
        x = x.reshape([-1, self.input_features, 1, 1])
        x = self.dense0(x)
        # x = self.activation(x)
        x = self.dense1(x)
        x = self.activation(x)
        for index, block in enumerate(self.blocks[:-3]):
            x = block(x)
        x = self.blocks[-3](x)
        x = self.blocks[-2](x)
        x = self.blocks[-1](x)
        x = self.end_conv(x)
        x = x[:, :, :100, :100]
        x = x[:, :, theta_bin]
        outputs = x.reshape([-1, 100])
        return outputs
