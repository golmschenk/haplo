import torch
from torch.nn import ModuleList, Conv2d, LeakyReLU, Module

from cura_2D_model import ResidualGenerationLightCurveNetworkBlock2D


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
        self.upsampling_factor = 2.25
        for output_channels in [512, 512, 1024, 1024, 2048, 2048]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock2D(
                output_channels=output_channels, input_channels=input_channels, upsampling_scale_factor=self.upsampling_factor,
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

        # Process the first blocks normally (up to the point where selective activation begins)
        for block in self.blocks[:-9]:
            x = block(x)

        # For earliest blocks, we need very large windows due to the cumulative effect of upsampling
        # Calculate initial capture window based on target theta_bin
        scale_factor = self.upsampling_factor ** 3  # Increased power for earlier blocks
        window_size = 9  # Largest window for earliest selective block
        lower_capture_theta_bin = max(0, int(torch.round(theta_bin / scale_factor) - window_size))
        upper_capture_theta_bin = int(torch.round(theta_bin / scale_factor) + window_size + 1)

        # First selective activation (9th from last block)
        sub_x = x[:, :, lower_capture_theta_bin:upper_capture_theta_bin]
        x = self.blocks[-9](sub_x)

        # 8th from last block
        scale_factor = self.upsampling_factor ** 2.5
        window_size = 8
        lower_capture_theta_bin = max(0, int(torch.round(theta_bin / scale_factor) - window_size))
        upper_capture_theta_bin = int(torch.round(theta_bin / scale_factor) + window_size + 1)
        sub_x = x[:, :, lower_capture_theta_bin:upper_capture_theta_bin]
        x = self.blocks[-8](sub_x)

        # 7th from last block
        scale_factor = self.upsampling_factor ** 2.25
        window_size = 7
        lower_capture_theta_bin = max(0, int(torch.round(theta_bin / scale_factor) - window_size))
        upper_capture_theta_bin = int(torch.round(theta_bin / scale_factor) + window_size + 1)
        sub_x = x[:, :, lower_capture_theta_bin:upper_capture_theta_bin]
        x = self.blocks[-7](sub_x)

        # 6th from last block
        scale_factor = self.upsampling_factor ** 2
        window_size = 6
        lower_capture_theta_bin = max(0, int(torch.round(theta_bin / scale_factor) - window_size))
        upper_capture_theta_bin = int(torch.round(theta_bin / scale_factor) + window_size + 1)
        sub_x = x[:, :, lower_capture_theta_bin:upper_capture_theta_bin]
        x = self.blocks[-6](sub_x)

        # 5th from last block
        scale_factor = self.upsampling_factor * 1.75
        window_size = 5
        lower_capture_theta_bin = max(0, int(torch.round(theta_bin / scale_factor) - window_size))
        upper_capture_theta_bin = int(torch.round(theta_bin / scale_factor) + window_size + 1)
        sub_x = x[:, :, lower_capture_theta_bin:upper_capture_theta_bin]
        x = self.blocks[-5](sub_x)

        # 4th from last block
        scale_factor = self.upsampling_factor * 1.25
        window_size = 4
        lower_capture_theta_bin = max(0, int(torch.round(theta_bin / scale_factor) - window_size))
        upper_capture_theta_bin = int(torch.round(theta_bin / scale_factor) + window_size + 1)
        sub_x = x[:, :, lower_capture_theta_bin:upper_capture_theta_bin]
        x = self.blocks[-4](sub_x)

        # 3rd from last block
        lower_capture_theta_bin = max(0, int(torch.round(theta_bin / 2.25) - 3))
        upper_capture_theta_bin = int(torch.round(theta_bin / 2.25) + 4)
        sub_x = x[:, :, lower_capture_theta_bin:upper_capture_theta_bin]
        x = self.blocks[-3](sub_x)

        # 2nd from last block - dynamic window centered on scaled theta_bin
        center = theta_bin  # Same scaling as previous block
        lower_capture_theta_bin = max(0, center - 2)  # 2 bins below center
        upper_capture_theta_bin = center + 3  # 2 bins above center (exclusive)
        sub_x = x[:, :, lower_capture_theta_bin:upper_capture_theta_bin]
        x = self.blocks[-2](sub_x)

        # Last block - keeping original slicing which was working
        sub_x = x[:, :, 1:-1]
        x = self.blocks[-1](sub_x)

        # Final processing - keeping original slicing which was working
        sub_x = x[:, :, 1:-1]
        x = self.end_conv(sub_x)
        x = x[:, :, :, :100]
        outputs = x.reshape([-1, 100])
        return outputs
