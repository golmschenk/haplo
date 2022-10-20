from tensorflow import sigmoid
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, Convolution1D, MaxPooling1D, AveragePooling1D, Permute

from ml4a.residual_light_curve_network_block import ResidualGenerationLightCurveNetworkBlock


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

class ResModel1NoDoAvgPoolEnd(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_input_channels: int = 11):
        super().__init__()
        self.blocks = []
        output_channels = 32
        self.reshape0 = Reshape([1, 11])
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, dropout_rate=0.0))
        input_channels = output_channels
        for output_channels in [28, 24, 20, 16, 12, 8]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.0))
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(output_channels=output_channels,
                                                                            dropout_rate=0.0))
            input_channels = output_channels
        self.pool_permute0 = Permute((2, 1))
        self.pool = AveragePooling1D(pool_size=8)
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

class ResModel1NoDoAvgPoolEnd8Wider(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_input_channels: int = 11):
        super().__init__()
        self.blocks = []
        wider_factor = 8
        output_channels = 32 * wider_factor
        self.reshape0 = Reshape([1, 11])
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, dropout_rate=0.0))
        input_channels = output_channels
        for output_channels in [28 * wider_factor, 24 * wider_factor, 20 * wider_factor, 16 * wider_factor, 12 * wider_factor, 8 * wider_factor]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.0))
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(output_channels=output_channels,
                                                                            dropout_rate=0.0))
            input_channels = output_channels
        self.pool_permute0 = Permute((2, 1))
        self.pool = AveragePooling1D(pool_size=8)
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


class ResModel1NoDoAvgPoolEndDoublingWider(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_input_channels: int = 11):
        super().__init__()
        self.blocks = []
        wider_factor = 8
        output_channels = 32 * wider_factor
        self.reshape0 = Reshape([1, 11])
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, dropout_rate=0.0))
        input_channels = output_channels
        for output_channels in [256, 128, 64, 32, 16, 8]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.0))
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(output_channels=output_channels,
                                                                            dropout_rate=0.0))
            input_channels = output_channels
        self.pool_permute0 = Permute((2, 1))
        self.pool = AveragePooling1D(pool_size=8)
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

class ResModel1InitialDenseNoDoConvEndDoublingWider(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_input_channels: int = 11):
        super().__init__()
        self.blocks = []
        self.reshape0 = Reshape([1, 11])
        self.dense0 = Convolution1D(50, kernel_size=1)
        self.dense1 = Convolution1D(50, kernel_size=1)
        output_channels = 256
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=50, dropout_rate=0.0))
        input_channels = output_channels
        for output_channels in [256, 128, 64, 32, 16, 8]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.0))
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(output_channels=output_channels,
                                                                            dropout_rate=0.0))
            input_channels = output_channels
        self.end_conv = Convolution1D(1, kernel_size=1)
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
        x = self.dense0(x, training=training)
        x = self.dense1(x, training=training)
        for index, block in enumerate(self.blocks):
            x = block(x, training=training)
        x = self.end_conv(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs


class ResModel1InitialDenseNoDoConvEndDoublingWiderer(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_input_channels: int = 11):
        super().__init__()
        self.blocks = []
        self.reshape0 = Reshape([1, 11])
        self.dense0 = Convolution1D(50, kernel_size=1)
        self.dense1 = Convolution1D(50, kernel_size=1)
        output_channels = 256 * 8
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=50, dropout_rate=0.0))
        input_channels = output_channels
        for output_channels in [256 * 8, 128 * 8, 64 * 8, 32 * 8, 16 * 8, 8 * 8]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.0))
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(output_channels=output_channels,
                                                                            dropout_rate=0.0))
            input_channels = output_channels
        self.end_conv = Convolution1D(1, kernel_size=1)
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
        x = self.dense0(x, training=training)
        x = self.dense1(x, training=training)
        for index, block in enumerate(self.blocks):
            x = block(x, training=training)
        x = self.end_conv(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs

class ResModel1InitialDenseNoDoConvEndDoublingWidererL2(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_input_channels: int = 11):
        super().__init__()
        self.blocks = []
        self.reshape0 = Reshape([1, 11])
        self.dense0 = Convolution1D(50, kernel_size=1)
        self.dense1 = Convolution1D(50, kernel_size=1)
        output_channels = 256 * 8
        l2_rate = 0.0001
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=50, dropout_rate=0.0, l2_regularization=l2_rate))
        input_channels = output_channels
        for output_channels in [256 * 8, 128 * 8, 64 * 8, 32 * 8, 16 * 8, 8 * 8]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.0, l2_regularization=l2_rate))
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(output_channels=output_channels,
                                                                            dropout_rate=0.0, l2_regularization=l2_rate))
            input_channels = output_channels
        self.end_conv = Convolution1D(1, kernel_size=1)
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
        x = self.dense0(x, training=training)
        x = self.dense1(x, training=training)
        for index, block in enumerate(self.blocks):
            x = block(x, training=training)
        x = self.end_conv(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs


class Lira(ResModel1InitialDenseNoDoConvEndDoublingWidererL2):
    pass

class LiraWithDoNoLrExtraEndLayer(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_input_channels: int = 11):
        super().__init__()
        self.blocks = []
        self.reshape0 = Reshape([1, 11])
        self.dense0 = Convolution1D(50, kernel_size=1)
        self.dense1 = Convolution1D(50, kernel_size=1)
        output_channels = 256 * 8
        l2_rate = 0.0
        dropout_rate = 0.1
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=50, dropout_rate=dropout_rate, l2_regularization=l2_rate))
        input_channels = output_channels
        for output_channels in [256 * 8, 128 * 8, 64 * 8, 32 * 8, 16 * 8, 8 * 8]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2,
                dropout_rate=dropout_rate, l2_regularization=l2_rate))
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                    output_channels=output_channels, dropout_rate=dropout_rate, l2_regularization=l2_rate))
            input_channels = output_channels
        self.end_conv0 = Convolution1D(10, kernel_size=1)
        self.end_conv1 = Convolution1D(1, kernel_size=1)
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
        x = self.dense0(x, training=training)
        x = self.dense1(x, training=training)
        for index, block in enumerate(self.blocks):
            x = block(x, training=training)
        x = self.end_conv0(x, training=training)
        x = self.end_conv1(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs

class LiraNoL2(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_input_channels: int = 11):
        super().__init__()
        self.blocks = []
        self.reshape0 = Reshape([1, 11])
        self.dense0 = Convolution1D(50, kernel_size=1)
        self.dense1 = Convolution1D(50, kernel_size=1)
        output_channels = 256 * 8
        l2_rate = 0.0
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=50, dropout_rate=0.0, l2_regularization=l2_rate))
        input_channels = output_channels
        for output_channels in [256 * 8, 128 * 8, 64 * 8, 32 * 8, 16 * 8, 8 * 8]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.0, l2_regularization=l2_rate))
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(output_channels=output_channels,
                                                                            dropout_rate=0.0, l2_regularization=l2_rate))
            input_channels = output_channels
        self.end_conv = Convolution1D(1, kernel_size=1)
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
        x = self.dense0(x, training=training)
        x = self.dense1(x, training=training)
        for index, block in enumerate(self.blocks):
            x = block(x, training=training)
        x = self.end_conv(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs

class LiraExtraEndLayer(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_input_channels: int = 11):
        super().__init__()
        self.blocks = []
        self.reshape0 = Reshape([1, 11])
        self.dense0 = Convolution1D(50, kernel_size=1)
        self.dense1 = Convolution1D(50, kernel_size=1)
        output_channels = 256 * 8
        l2_rate = 0.0001
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=50, dropout_rate=0.0, l2_regularization=l2_rate))
        input_channels = output_channels
        for output_channels in [256 * 8, 128 * 8, 64 * 8, 32 * 8, 16 * 8, 8 * 8]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.0, l2_regularization=l2_rate))
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(output_channels=output_channels,
                                                                            dropout_rate=0.0, l2_regularization=l2_rate))
            input_channels = output_channels
        self.end_conv0 = Convolution1D(10, kernel_size=1)
        self.end_conv1 = Convolution1D(1, kernel_size=1)
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
        x = self.dense0(x, training=training)
        x = self.dense1(x, training=training)
        for index, block in enumerate(self.blocks):
            x = block(x, training=training)
        x = self.end_conv0(x, training=training)
        x = self.end_conv1(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs

class LiraWithDoExtraEndLayer(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_input_channels: int = 11):
        super().__init__()
        self.blocks = []
        self.reshape0 = Reshape([1, 11])
        self.dense0 = Convolution1D(50, kernel_size=1)
        self.dense1 = Convolution1D(50, kernel_size=1)
        output_channels = 256 * 8
        l2_rate = 0.0001
        dropout_rate = 0.1
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=50, dropout_rate=dropout_rate, l2_regularization=l2_rate))
        input_channels = output_channels
        for output_channels in [256 * 8, 128 * 8, 64 * 8, 32 * 8, 16 * 8, 8 * 8]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2,
                dropout_rate=dropout_rate, l2_regularization=l2_rate))
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                    output_channels=output_channels, dropout_rate=dropout_rate, l2_regularization=l2_rate))
            input_channels = output_channels
        self.end_conv0 = Convolution1D(10, kernel_size=1)
        self.end_conv1 = Convolution1D(1, kernel_size=1)
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
        x = self.dense0(x, training=training)
        x = self.dense1(x, training=training)
        for index, block in enumerate(self.blocks):
            x = block(x, training=training)
        x = self.end_conv0(x, training=training)
        x = self.end_conv1(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs