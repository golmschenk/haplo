from tensorflow.keras import Model
from tensorflow.keras.layers import LeakyReLU, Dense, Conv1DTranspose, Reshape, Cropping1D, Layer, Dropout,\
    SpatialDropout1D, BatchNormalization


class SimpleModel(Model):
    def __init__(self, number_of_label_types=1):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        # l2_regularizer = l2(0.001)
        self.dense0 = Dense(11, activation=leaky_relu)
        self.dense1 = Dense(11, activation=leaky_relu)
        self.dense2 = Dense(11, activation=leaky_relu)
        self.reshape0 = Reshape([1, 11])
        self.transposed_convolution0 = Conv1DTranspose(filters=8, kernel_size=4, strides=2, activation=leaky_relu)
        self.transposed_convolution1 = Conv1DTranspose(filters=4, kernel_size=4, strides=2, activation=leaky_relu)
        self.transposed_convolution2 = Conv1DTranspose(filters=2, kernel_size=4, strides=2, activation=leaky_relu)
        self.transposed_convolution3 = Conv1DTranspose(filters=1, kernel_size=4, strides=3, activation=leaky_relu)
        self.cropping0 = Cropping1D((0, 3))
        self.reshape1 = Reshape([64])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.dense0(x, training=training)
        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)
        x = self.reshape0(x, training=training)
        x = self.transposed_convolution0(x, training=training)
        x = self.transposed_convolution1(x, training=training)
        x = self.transposed_convolution2(x, training=training)
        x = self.transposed_convolution3(x, training=training)
        x = self.cropping0(x, training=training)
        x = self.reshape1(x, training=training)
        return x


class Conv1DTransposeBlock(Layer):
    def __init__(self, filters: int, kernel_size: int, strides: int, dropout_rate: float = 0.1,
                 batch_normalization: bool = True, spatial: bool = True):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        self.convolution = Conv1DTranspose(filters, kernel_size=kernel_size, strides=strides, activation=leaky_relu)
        if dropout_rate > 0:
            if spatial:
                self.dropout = SpatialDropout1D(dropout_rate)
            else:
                self.dropout = Dropout(dropout_rate)
        else:
            self.dropout = None
        if batch_normalization:
            self.batch_normalization = BatchNormalization(scale=False)
            if not spatial:
                self.batch_normalization_input_reshape = Reshape([-1])
                self.batch_normalization_output_reshape = Reshape([-1, filters])
            else:
                self.batch_normalization_input_reshape = None
                self.batch_normalization_output_reshape = None
        else:
            self.batch_normalization = None

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.convolution(x, training=training)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        if self.batch_normalization is not None:
            if self.batch_normalization_input_reshape is not None:
                x = self.batch_normalization_input_reshape(x, training=training)
            x = self.batch_normalization(x, training=training)
            if self.batch_normalization_output_reshape is not None:
                x = self.batch_normalization_output_reshape(x, training=training)
        return x

class DenseBlock(Layer):
    def __init__(self, filters: int, dropout_rate: float = 0.1,
                 batch_normalization: bool = True, spatial: bool = False):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        self.dense = Dense(filters, activation=leaky_relu)
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)
        else:
            self.dropout = None
        if batch_normalization:
            self.batch_normalization = BatchNormalization(scale=False)
            if not spatial:
                self.batch_normalization_input_reshape = Reshape([-1, filters])
                self.batch_normalization_output_reshape = Reshape([-1])
            else:
                self.batch_normalization_input_reshape = None
                self.batch_normalization_output_reshape = None
        else:
            self.batch_normalization = None

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.dense(x, training=training)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        if self.batch_normalization is not None:
            if self.batch_normalization_input_reshape is not None:
                x = self.batch_normalization_input_reshape(x, training=training)
            x = self.batch_normalization(x, training=training)
            if self.batch_normalization_output_reshape is not None:
                x = self.batch_normalization_output_reshape(x, training=training)
        return x


class WiderWithDropoutModel(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(20, batch_normalization=False)
        self.dense1 = DenseBlock(30, batch_normalization=False)
        self.dense2 = DenseBlock(40, batch_normalization=False, spatial=True)
        self.reshape0 = Reshape([1, 40])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=30, kernel_size=4, strides=2, batch_normalization=False)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=20, kernel_size=4, strides=2, batch_normalization=False)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=10, kernel_size=4, strides=2, batch_normalization=False)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=1, kernel_size=4, strides=3, batch_normalization=False, dropout_rate=0)
        self.cropping0 = Cropping1D((0, 3))
        self.reshape1 = Reshape([64])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.dense0(x, training=training)
        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)
        x = self.reshape0(x, training=training)
        x = self.transposed_convolution0(x, training=training)
        x = self.transposed_convolution1(x, training=training)
        x = self.transposed_convolution2(x, training=training)
        x = self.transposed_convolution3(x, training=training)
        x = self.cropping0(x, training=training)
        x = self.reshape1(x, training=training)
        return x


class Nyx(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(20, batch_normalization=False)
        self.dense1 = DenseBlock(30, batch_normalization=False)
        self.dense2 = DenseBlock(40, batch_normalization=False)
        self.dense3 = DenseBlock(50, batch_normalization=True)
        self.dense4 = DenseBlock(60, batch_normalization=True)
        self.dense5 = DenseBlock(70, batch_normalization=True, spatial=True)
        self.reshape0 = Reshape([1, 70])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=60, kernel_size=2, strides=1, batch_normalization=True)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=50, kernel_size=3, strides=1, batch_normalization=True)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=40, kernel_size=4, strides=1, batch_normalization=True)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=30, kernel_size=4, strides=2, batch_normalization=False)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=20, kernel_size=4, strides=2, batch_normalization=False)
        self.transposed_convolution5 = Conv1DTransposeBlock(filters=1, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0)
        self.reshape1 = Reshape([64])
        self.cropping0 = Cropping1D((3, 3))

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.dense0(x, training=training)
        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)
        x = self.dense3(x, training=training)
        x = self.dense4(x, training=training)
        x = self.dense5(x, training=training)
        x = self.reshape0(x, training=training)
        x = self.transposed_convolution0(x, training=training)
        x = self.transposed_convolution1(x, training=training)
        x = self.transposed_convolution2(x, training=training)
        x = self.transposed_convolution3(x, training=training)
        x = self.transposed_convolution4(x, training=training)
        x = self.transposed_convolution5(x, training=training)
        x = self.cropping0(x, training=training)
        x = self.reshape1(x, training=training)
        return x
