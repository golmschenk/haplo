from torch.nn import Module, Conv1d, LeakyReLU, ModuleList

from haplo.internal.models.residual_generation_light_curve_network_block import ResidualGenerationLightCurveNetworkBlock


class LiraTraditionalShape8xWidthWithNoDoNoBn(Module):
    def __init__(self):
        super().__init__()
        self.blocks = ModuleList()
        self.dense0 = Conv1d(11, 400, kernel_size=1)
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


class LiraTraditionalShape8xWidthWith0d5DoNoBnOldFirstLayers(Module):
    def __init__(self):
        super().__init__()
        self.blocks = ModuleList()
        self.dense0 = Conv1d(11, 400, kernel_size=1)
        self.activation = LeakyReLU()
        self.dense1 = Conv1d(self.dense0.out_channels, 400, kernel_size=1)
        output_channels = 128
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(input_channels=400, output_channels=output_channels,
                                                                    batch_normalization=False, dropout_rate=0.5,
                                                                    activation_type=LeakyReLU))
        input_channels = output_channels
        for output_channels in [512, 512, 1024, 1024, 2048, 2048]:
            self.blocks.append(
                ResidualGenerationLightCurveNetworkBlock(input_channels=input_channels, output_channels=output_channels,
                                                         upsampling_scale_factor=2, batch_normalization=False,
                                                         dropout_rate=0.5, activation_type=LeakyReLU))
            input_channels = output_channels
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(input_channels=input_channels,
                                                                            output_channels=output_channels,
                                                                            batch_normalization=False, dropout_rate=0.5,
                                                                            activation_type=LeakyReLU))
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


class BrontPrototype0(Module):
    def __init__(self, input_features: int = 11):
        super().__init__()
        self.input_features = input_features
        self.blocks = ModuleList()
        self.dense0 = Conv1d(self.input_features, 400, kernel_size=1)
        self.activation = LeakyReLU()
        self.dense1 = Conv1d(self.dense0.out_channels, 400, kernel_size=1)
        output_channels = 128
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(input_channels=400, output_channels=output_channels,
                                                                    batch_normalization=False, dropout_rate=0.0,
                                                                    activation_type=LeakyReLU))
        input_channels = output_channels
        for output_channels in [512, 512, 256, 256, 128, 128]:
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

    def forward(self, x):
        x = x.reshape([-1, self.input_features, 1])
        x = self.dense0(x)
        x = self.activation(x)
        x = self.dense1(x)
        x = self.activation(x)
        for index, block in enumerate(self.blocks):
            x = block(x)
        x = self.end_conv(x)
        outputs = x.reshape([-1, 64])
        return outputs
