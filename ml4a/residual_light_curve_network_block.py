"""
Code for a residual light curve network block.
"""
from typing import Optional, List

from tensorflow.keras.layers import LeakyReLU, Convolution1D, MaxPooling1D, BatchNormalization, \
    Layer, Permute, ZeroPadding1D, SpatialDropout1D, Conv1DTranspose, UpSampling1D, Cropping1D, ReLU, Dropout
from tensorflow.keras.regularizers import L2


class ResidualGenerationLightCurveNetworkBlock(Layer):
    def __init__(self, output_channels: int, input_channels: Optional[int] = None, kernel_size: int = 3,
                 pooling_size: int = 1, batch_normalization: bool = True, dropout_rate: float = 0.0,
                 l2_regularization: float = 0.0, leaky_relu_alpha: float = 0.01):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=leaky_relu_alpha)
        dimension_decrease_factor = 4
        if batch_normalization:
            self.batch_normalization = BatchNormalization(scale=False)
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


class ResidualGenerationLightCurveNetworkBlockWithoutDimensionDecrease(Layer):
    def __init__(self, output_channels: int, input_channels: Optional[int] = None, kernel_size: int = 3,
                 pooling_size: int = 1, batch_normalization: bool = True, dropout_rate: float = 0.0,
                 l2_regularization: float = 0.0):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        if batch_normalization:
            self.batch_normalization = BatchNormalization(scale=False)
        else:
            self.batch_normalization = None
        if l2_regularization > 0:
            l2_regularizer = L2(l2_regularization)
        else:
            l2_regularizer = None
        self.convolutional_layer = Conv1DTranspose(output_channels, kernel_size=kernel_size, activation=leaky_relu,
                                                   padding='same', kernel_regularizer=l2_regularizer)
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
        y = self.convolutional_layer(y, training=training)
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


class MultiResidualGenerationLightCurveNetworkBlockWithoutDimensionDecrease(Layer):
    def __init__(self, output_channels: int, input_channels: Optional[int] = None, kernel_size: int = 3,
                 pooling_size: int = 1, batch_normalization: bool = True, dropout_rate: float = 0.0,
                 l2_regularization: float = 0.0, number_of_layers: int = 2):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        if batch_normalization:
            self.batch_normalization = BatchNormalization(scale=False)
        else:
            self.batch_normalization = None
        if l2_regularization > 0:
            l2_regularizer = L2(l2_regularization)
        else:
            l2_regularizer = None
        self.convolutional_layers: List[Conv1DTranspose] = []
        for _ in range(number_of_layers - 1):
            self.convolutional_layers.append(Conv1DTranspose(output_channels, kernel_size=kernel_size,
                                                             activation=leaky_relu,
                                                             padding='same', kernel_regularizer=l2_regularizer))
        self.convolutional_layers.append(Conv1DTranspose(output_channels, kernel_size=kernel_size,
                                                         activation=leaky_relu,
                                                         padding='same', kernel_regularizer=l2_regularizer))
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
        for convolutional_layer in self.convolutional_layers:
            y = convolutional_layer(y, training=training)
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


class ResnetLikeMultiResidualGenerationLightCurveNetworkBlock(Layer):
    def __init__(self, output_channels: int, input_channels: Optional[int] = None, kernel_size: int = 3,
                 pooling_size: int = 1, batch_normalization: bool = True, dropout_rate: float = 0.0,
                 l2_regularization: float = 0.0, number_of_layers: int = 2):
        super().__init__()
        self.leaky_relu = LeakyReLU(alpha=0.01)
        if l2_regularization > 0:
            l2_regularizer = L2(l2_regularization)
        else:
            l2_regularizer = None
        self.convolutional_layers: List[Conv1DTranspose] = []
        self.batch_normalizations = []
        for _ in range(number_of_layers - 1):
            self.convolutional_layers.append(Conv1DTranspose(output_channels, kernel_size=kernel_size,
                                                             padding='same', kernel_regularizer=l2_regularizer))
            if batch_normalization:
                self.batch_normalizations.append(BatchNormalization(scale=False))
        self.convolutional_layers.append(Conv1DTranspose(output_channels, kernel_size=kernel_size,
                                                         padding='same', kernel_regularizer=l2_regularizer))
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
        for layer_index, convolutional_layer in enumerate(self.convolutional_layers):
            y = convolutional_layer(y, training=training)
            if layer_index != len(self.convolutional_layers) - 1:
                if len(self.batch_normalizations) != 0:
                    y = self.batch_normalizations[layer_index](y)
                y = self.leaky_relu(y)
        if self.upsampling_layer is not None:
            x = self.upsampling_layer(x, training=training)
            y = self.upsampling_layer(y, training=training)
        if self.dimension_change_layer is not None:
            x = self.dimension_change_permute0(x, training=training)
            x = self.dimension_change_layer(x, training=training)
            x = self.dimension_change_permute1(x, training=training)
        if self.dropout_layer is not None:
            y = self.dropout_layer(y, training=training)
        return self.leaky_relu(x + y)


class ResnetLikeReluMultiResidualGenerationLightCurveNetworkBlock(Layer):
    def __init__(self, output_channels: int, input_channels: Optional[int] = None, kernel_size: int = 3,
                 pooling_size: int = 1, batch_normalization: bool = True, dropout_rate: float = 0.0,
                 l2_regularization: float = 0.0, number_of_layers: int = 2):
        super().__init__()
        self.relu = ReLU()
        if l2_regularization > 0:
            l2_regularizer = L2(l2_regularization)
        else:
            l2_regularizer = None
        self.convolutional_layers: List[Conv1DTranspose] = []
        self.batch_normalizations = []
        for _ in range(number_of_layers - 1):
            self.convolutional_layers.append(Conv1DTranspose(output_channels, kernel_size=kernel_size,
                                                             padding='same', kernel_regularizer=l2_regularizer))
            if batch_normalization:
                self.batch_normalizations.append(BatchNormalization(scale=False))
        self.convolutional_layers.append(Conv1DTranspose(output_channels, kernel_size=kernel_size,
                                                         padding='same', kernel_regularizer=l2_regularizer))
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
        for layer_index, convolutional_layer in enumerate(self.convolutional_layers):
            y = convolutional_layer(y, training=training)
            if layer_index != len(self.convolutional_layers) - 1:
                if len(self.batch_normalizations) != 0:
                    y = self.batch_normalizations[layer_index](y)
                y = self.relu(y)
        if self.upsampling_layer is not None:
            x = self.upsampling_layer(x, training=training)
            y = self.upsampling_layer(y, training=training)
        if self.dimension_change_layer is not None:
            x = self.dimension_change_permute0(x, training=training)
            x = self.dimension_change_layer(x, training=training)
            x = self.dimension_change_permute1(x, training=training)
        if self.dropout_layer is not None:
            y = self.dropout_layer(y, training=training)
        return self.relu(x + y)


class ResnetLikeReluNonSpatialDropoutMultiResidualGenerationLightCurveNetworkBlock(Layer):
    def __init__(self, output_channels: int, input_channels: Optional[int] = None, kernel_size: int = 3,
                 pooling_size: int = 1, batch_normalization: bool = True, dropout_rate: float = 0.0,
                 l2_regularization: float = 0.0, number_of_layers: int = 2):
        super().__init__()
        self.relu = ReLU()
        if l2_regularization > 0:
            l2_regularizer = L2(l2_regularization)
        else:
            l2_regularizer = None
        self.convolutional_layers: List[Conv1DTranspose] = []
        self.batch_normalizations = []
        for _ in range(number_of_layers - 1):
            self.convolutional_layers.append(Conv1DTranspose(output_channels, kernel_size=kernel_size,
                                                             padding='same', kernel_regularizer=l2_regularizer))
            if batch_normalization:
                self.batch_normalizations.append(BatchNormalization(scale=False))
        self.convolutional_layers.append(Conv1DTranspose(output_channels, kernel_size=kernel_size,
                                                         padding='same', kernel_regularizer=l2_regularizer))
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
            self.dropout_layer = Dropout(rate=dropout_rate)
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
        for layer_index, convolutional_layer in enumerate(self.convolutional_layers):
            y = convolutional_layer(y, training=training)
            if layer_index != len(self.convolutional_layers) - 1:
                if len(self.batch_normalizations) != 0:
                    y = self.batch_normalizations[layer_index](y)
                y = self.relu(y)
        if self.upsampling_layer is not None:
            x = self.upsampling_layer(x, training=training)
            y = self.upsampling_layer(y, training=training)
        if self.dimension_change_layer is not None:
            x = self.dimension_change_permute0(x, training=training)
            x = self.dimension_change_layer(x, training=training)
            x = self.dimension_change_permute1(x, training=training)
        if self.dropout_layer is not None:
            y = self.dropout_layer(y, training=training)
        return self.relu(x + y)
