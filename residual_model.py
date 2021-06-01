from tensorflow import sigmoid
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, Convolution1D, MaxPooling1D, AveragePooling1D, Permute

from residual_light_curve_network_block import ResidualGenerationLightCurveNetworkBlock


class ResModel0(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_input_channels: int = 11):
        super().__init__()
        self.blocks = []
        output_channels = 32
        self.reshape0 = Reshape([1, 11])
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, dropout_rate=0.5))
        input_channels = output_channels
        for output_channels in [28, 24, 20, 16, 12, 8]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.5))
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(output_channels=output_channels,
                                                                            dropout_rate=0.5))
            input_channels = output_channels
        self.pool_permute0 = Permute((2, 1))
        self.pool = MaxPooling1D(pool_size=8)
        self.pool_permute1 = Permute((2, 1))
        self.reshape = Reshape([64])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.reshape0(x, training=training)
        for index, block in enumerate(self.blocks):
            x = block(x, training=training)
        x = self.pool_permute0(x, training=training)
        x = self.pool(x, training=training)
        x = self.pool_permute1(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs
