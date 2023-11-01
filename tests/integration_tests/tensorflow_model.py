from typing import Optional

from tensorflow.keras import Model
from tensorflow.keras.layers import LeakyReLU, Convolution1D, MaxPooling1D, \
    Layer, Permute, ZeroPadding1D, SpatialDropout1D, Conv1DTranspose, UpSampling1D, Cropping1D, ReLU, Dropout, Reshape
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import BatchNormalization

class ResidualGenerationLightCurveNetworkBlock(Layer):
    def __init__(self, output_channels: int, input_channels: Optional[int] = None, kernel_size: int = 3,
                 pooling_size: int = 1, batch_normalization: bool = False, dropout_rate: float = 0.0,
                 l2_regularization: float = 0.0, leaky_relu_alpha: float = 0.01, renorm: bool = False):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=leaky_relu_alpha)
        dimension_decrease_factor = 4
        if batch_normalization:
            self.batch_normalization = BatchNormalization(scale=False, renorm=renorm)
        else:
            self.batch_normalization = None
        if l2_regularization > 0:
            l2_regularizer = L2(l2_regularization)
        else:
            l2_regularizer = None
        self.dimension_decrease_layer = Conv1DTranspose(
            output_channels // dimension_decrease_factor, kernel_size=1, activation=leaky_relu,
            kernel_regularizer=l2_regularizer)
        self.convolutional_layer = Conv1DTranspose(
            output_channels // dimension_decrease_factor, kernel_size=kernel_size, activation=leaky_relu,
            padding='same', kernel_regularizer=l2_regularizer)
        self.dimension_increase_layer = Conv1DTranspose(
            output_channels, kernel_size=1, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        if pooling_size > 1:
            self.upsampling_layer = UpSampling1D(size=pooling_size)
        else:
            self.upsampling_layer = None
        if input_channels is not None and output_channels != input_channels:
            if output_channels < input_channels:
                self.dimension_change_permute0 = Permute((2, 1))
                self.dimension_change_layer = Cropping1D((0, input_channels - output_channels))
                self.dimension_change_permute1 = Permute((2, 1))
            else:
                self.dimension_change_permute0 = Permute((2, 1))
                self.dimension_change_layer = ZeroPadding1D(padding=(0, output_channels - input_channels))
                self.dimension_change_permute1 = Permute((2, 1))
        else:
            self.dimension_change_layer = None
        if dropout_rate > 0:
            self.dropout_layer = SpatialDropout1D(rate=dropout_rate)
        else:
            self.dropout_layer = None

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the block.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        y = x
        if self.batch_normalization is not None:
            y = self.batch_normalization(y, training=training)
        y = self.dimension_decrease_layer(y, training=training)
        y = self.convolutional_layer(y, training=training)
        y = self.dimension_increase_layer(y, training=training)
        if self.upsampling_layer is not None:
            x = self.upsampling_layer(x, training=training)
            y = self.upsampling_layer(y, training=training)
        if self.dimension_change_layer is not None:
            x = self.dimension_change_permute0(x, training=training)
            x = self.dimension_change_layer(x, training=training)
            x = self.dimension_change_permute1(x, training=training)
        if self.dropout_layer is not None:
            y = self.dropout_layer(y, training=training)
        return x + y

class LiraTraditionalShape8xWidthWithNoDoNoBn(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_input_channels: int = 11):
        super().__init__()
        self.blocks = []
        self.reshape0 = Reshape([1, 11])
        self.dense0 = Convolution1D(400, kernel_size=1)
        self.dense1 = Convolution1D(400, kernel_size=1)
        output_channels = 128
        l2_rate = 0.0001
        self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=400, dropout_rate=0.0, l2_regularization=l2_rate,
            batch_normalization=False))
        input_channels = output_channels
        for output_channels in [512, 512, 1024, 1024, 2048, 2048]:
            self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.0,
                l2_regularization=l2_rate, batch_normalization=False))
            for _ in range(2):
                self.blocks.append(ResidualGenerationLightCurveNetworkBlock(
                    output_channels=output_channels, dropout_rate=0.0, l2_regularization=l2_rate,
                    batch_normalization=False))
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