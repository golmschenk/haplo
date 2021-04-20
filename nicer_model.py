from tensorflow.keras import Model, regularizers
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, Dense, Conv1DTranspose, Reshape, Cropping1D, Layer, Dropout,\
    SpatialDropout1D, BatchNormalization, GaussianDropout, AlphaDropout
from tensorflow.keras.layers import GaussianNoise


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
        if self.batch_normalization is not None:
            if self.batch_normalization_input_reshape is not None:
                x = self.batch_normalization_input_reshape(x, training=training)
            x = self.batch_normalization(x, training=training)
            if self.batch_normalization_output_reshape is not None:
                x = self.batch_normalization_output_reshape(x, training=training)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
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
        if self.batch_normalization is not None:
            if self.batch_normalization_input_reshape is not None:
                x = self.batch_normalization_input_reshape(x, training=training)
            x = self.batch_normalization(x, training=training)
            if self.batch_normalization_output_reshape is not None:
                x = self.batch_normalization_output_reshape(x, training=training)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
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

class Nyx2(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(30, batch_normalization=False)
        self.dense1 = DenseBlock(60, batch_normalization=False)
        self.dense2 = DenseBlock(120, batch_normalization=False)
        self.dense3 = DenseBlock(200, batch_normalization=True)
        self.dense4 = DenseBlock(400, batch_normalization=True)
        self.dense5 = DenseBlock(800, batch_normalization=True, spatial=True)
        self.reshape0 = Reshape([1, 800])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400, kernel_size=2, strides=1, batch_normalization=True)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=200, kernel_size=3, strides=1, batch_normalization=True)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=120, kernel_size=4, strides=1, batch_normalization=True)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=60, kernel_size=4, strides=2, batch_normalization=False)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=30, kernel_size=4, strides=2, batch_normalization=False)
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


class Nyx3(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(30, batch_normalization=False)
        self.dense1 = DenseBlock(60, batch_normalization=False)
        self.dense2 = DenseBlock(120, batch_normalization=False)
        self.dense3 = DenseBlock(200, batch_normalization=False)
        self.dense4 = DenseBlock(400, batch_normalization=False)
        self.dense5 = DenseBlock(800, batch_normalization=False, spatial=True)
        self.reshape0 = Reshape([1, 800])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400, kernel_size=2, strides=1, batch_normalization=False)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=200, kernel_size=3, strides=1, batch_normalization=False)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=120, kernel_size=4, strides=1, batch_normalization=False)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=60, kernel_size=4, strides=2, batch_normalization=False)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=30, kernel_size=4, strides=2, batch_normalization=False)
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

class Nyx4(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(30, batch_normalization=False, dropout_rate=0)
        self.dense1 = DenseBlock(60, batch_normalization=False, dropout_rate=0)
        self.dense2 = DenseBlock(120, batch_normalization=False, dropout_rate=0)
        self.dense3 = DenseBlock(200, batch_normalization=True, dropout_rate=0)
        self.dense4 = DenseBlock(400, batch_normalization=True, dropout_rate=0)
        self.dense5 = DenseBlock(800, batch_normalization=True, spatial=True, dropout_rate=0)
        self.reshape0 = Reshape([1, 800])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400, kernel_size=2, strides=1, batch_normalization=True, dropout_rate=0)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=200, kernel_size=3, strides=1, batch_normalization=True, dropout_rate=0)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=120, kernel_size=4, strides=1, batch_normalization=True, dropout_rate=0)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=60, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=30, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0)
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

class Nyx5(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(30, batch_normalization=False, dropout_rate=0)
        self.dense1 = DenseBlock(60, batch_normalization=False, dropout_rate=0)
        self.dense2 = DenseBlock(120, batch_normalization=True, dropout_rate=0)
        self.dense3 = DenseBlock(200, batch_normalization=True, dropout_rate=0)
        self.dense4 = DenseBlock(400, batch_normalization=True, dropout_rate=0)
        self.dense5 = DenseBlock(800, batch_normalization=True, spatial=True, dropout_rate=0)
        self.reshape0 = Reshape([1, 800])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400, kernel_size=2, strides=1, batch_normalization=True, dropout_rate=0)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=200, kernel_size=3, strides=1, batch_normalization=True, dropout_rate=0)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=120, kernel_size=4, strides=1, batch_normalization=True, dropout_rate=0)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=60, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0.5)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=30, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0.5)
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

class Nyx4Narrow(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(15, batch_normalization=False, dropout_rate=0)
        self.dense1 = DenseBlock(20, batch_normalization=False, dropout_rate=0)
        self.dense2 = DenseBlock(25, batch_normalization=False, dropout_rate=0)
        self.dense3 = DenseBlock(30, batch_normalization=True, dropout_rate=0)
        self.dense4 = DenseBlock(35, batch_normalization=True, dropout_rate=0)
        self.dense5 = DenseBlock(40, batch_normalization=True, spatial=True, dropout_rate=0)
        self.reshape0 = Reshape([1, 40])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=35, kernel_size=2, strides=1, batch_normalization=True, dropout_rate=0)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=30, kernel_size=3, strides=1, batch_normalization=True, dropout_rate=0)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=25, kernel_size=4, strides=1, batch_normalization=True, dropout_rate=0)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=20, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=15, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0)
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

class Nyx6(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(30, batch_normalization=False)
        self.dense1 = DenseBlock(60, batch_normalization=False)
        self.dense2 = DenseBlock(120, batch_normalization=False)
        self.dense3 = DenseBlock(200, batch_normalization=True)
        self.dense4 = DenseBlock(400, batch_normalization=True)
        self.dense5 = DenseBlock(800, batch_normalization=True, spatial=True)
        self.reshape0 = Reshape([1, 800])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=800, kernel_size=2, strides=1, batch_normalization=True)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=800, kernel_size=3, strides=1, batch_normalization=True)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=800, kernel_size=4, strides=1, batch_normalization=True)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=800, kernel_size=4, strides=2, batch_normalization=False)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=800, kernel_size=4, strides=2, batch_normalization=False)
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

class Nyx7(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(1000, batch_normalization=False)
        self.dense1 = DenseBlock(900, batch_normalization=True)
        self.dense2 = DenseBlock(800, batch_normalization=True)
        self.dense3 = DenseBlock(700, batch_normalization=True)
        self.dense4 = DenseBlock(600, batch_normalization=True)
        self.dense5 = DenseBlock(500, batch_normalization=True, spatial=True)
        self.reshape0 = Reshape([1, 500])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400, kernel_size=2, strides=1, batch_normalization=True)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=200, kernel_size=3, strides=1, batch_normalization=True)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=120, kernel_size=4, strides=1, batch_normalization=True)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=60, kernel_size=4, strides=2, batch_normalization=False)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=30, kernel_size=4, strides=2, batch_normalization=False)
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

class Nyx8(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(1000, batch_normalization=False, dropout_rate=0.5)
        self.dense1 = DenseBlock(900, batch_normalization=False, dropout_rate=0.5)
        self.dense2 = DenseBlock(800, batch_normalization=True)
        self.dense3 = DenseBlock(700, batch_normalization=True)
        self.dense4 = DenseBlock(600, batch_normalization=True)
        self.dense5 = DenseBlock(500, batch_normalization=True, spatial=True)
        self.reshape0 = Reshape([1, 500])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400, kernel_size=2, strides=1, batch_normalization=True, dropout_rate=0)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=200, kernel_size=3, strides=1, batch_normalization=True, dropout_rate=0)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=120, kernel_size=4, strides=1, batch_normalization=True, dropout_rate=0)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=60, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=30, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0)
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
    
class Nyx9(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(4000, batch_normalization=False, dropout_rate=0.5)
        self.dense1 = DenseBlock(3000, batch_normalization=False, dropout_rate=0.5)
        self.dense2 = DenseBlock(2000, batch_normalization=False, dropout_rate=0.5)
        self.dense3 = DenseBlock(1000, batch_normalization=False, dropout_rate=0.5)
        self.dense4 = DenseBlock(750, batch_normalization=False)
        self.dense5 = DenseBlock(500, batch_normalization=False, spatial=True)
        self.reshape0 = Reshape([1, 500])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400, kernel_size=2, strides=1, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=200, kernel_size=3, strides=1, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=120, kernel_size=4, strides=1, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=60, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=30, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0)
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

class Nyx10(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(4000, batch_normalization=False)
        self.dense1 = DenseBlock(3000, batch_normalization=True)
        self.dense2 = DenseBlock(2000, batch_normalization=True)
        self.dense3 = DenseBlock(1000, batch_normalization=True)
        self.dense4 = DenseBlock(750, batch_normalization=True)
        self.dense5 = DenseBlock(500, batch_normalization=True, spatial=True)
        self.reshape0 = Reshape([1, 500])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400, kernel_size=2, strides=1, batch_normalization=True)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=200, kernel_size=3, strides=1, batch_normalization=True)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=120, kernel_size=4, strides=1, batch_normalization=True)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=60, kernel_size=4, strides=2, batch_normalization=False)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=30, kernel_size=4, strides=2, batch_normalization=False)
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

class Nyx11(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(1000, batch_normalization=False, dropout_rate=0.5)
        self.dense1 = DenseBlock(900, batch_normalization=False, dropout_rate=0.5)
        self.dense2 = DenseBlock(800, batch_normalization=False, dropout_rate=0.5)
        self.dense3 = DenseBlock(700, batch_normalization=True, dropout_rate=0)
        self.dense4 = DenseBlock(600, batch_normalization=True, dropout_rate=0)
        self.dense5 = DenseBlock(500, batch_normalization=True, spatial=True, dropout_rate=0)
        self.reshape0 = Reshape([1, 500])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400, kernel_size=2, strides=1, batch_normalization=True, dropout_rate=0)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=200, kernel_size=3, strides=1, batch_normalization=True, dropout_rate=0)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=120, kernel_size=4, strides=1, batch_normalization=True, dropout_rate=0)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=60, kernel_size=4, strides=2, batch_normalization=True, dropout_rate=0)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=30, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0)
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

class Nyx12(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(1000, batch_normalization=False, dropout_rate=0.5)
        self.dense1 = DenseBlock(900, batch_normalization=False, dropout_rate=0.5)
        self.dense2 = DenseBlock(800, batch_normalization=False, dropout_rate=0.5)
        self.dense3 = DenseBlock(700, batch_normalization=False, dropout_rate=0.5)
        self.dense4 = DenseBlock(600, batch_normalization=False, dropout_rate=0.5)
        self.dense5 = DenseBlock(500, batch_normalization=False, spatial=True, dropout_rate=0.5)
        self.reshape0 = Reshape([1, 500])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400, kernel_size=2, strides=1, batch_normalization=True, dropout_rate=0)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=200, kernel_size=3, strides=1, batch_normalization=True, dropout_rate=0)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=120, kernel_size=4, strides=1, batch_normalization=True, dropout_rate=0)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=60, kernel_size=4, strides=2, batch_normalization=True, dropout_rate=0)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=30, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0)
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


class Nyx13(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(4000, batch_normalization=False, dropout_rate=0.5)
        self.dense1 = DenseBlock(3000, batch_normalization=False, dropout_rate=0.5)
        self.dense2 = DenseBlock(2000, batch_normalization=False, dropout_rate=0.5)
        self.dense3 = DenseBlock(1000, batch_normalization=False, dropout_rate=0.5)
        self.dense4 = DenseBlock(750, batch_normalization=False)
        self.dense5 = DenseBlock(500, batch_normalization=False, spatial=True)
        self.reshape0 = Reshape([1, 500])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400, kernel_size=4, strides=1, batch_normalization=False)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=350, kernel_size=4, strides=1, batch_normalization=False)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=300, kernel_size=4, strides=1, batch_normalization=False)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=250, kernel_size=4, strides=2, batch_normalization=False)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=200, kernel_size=4, strides=1, batch_normalization=False)
        self.transposed_convolution5 = Conv1DTransposeBlock(filters=150, kernel_size=4, strides=1, batch_normalization=False)
        self.transposed_convolution6 = Conv1DTransposeBlock(filters=100, kernel_size=4, strides=1, batch_normalization=False)
        self.transposed_convolution7 = Conv1DTransposeBlock(filters=50, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution8 = Conv1DTranspose(filters=1, kernel_size=4, strides=1, activation='linear')
        self.reshape1 = Reshape([64])
        self.cropping0 = Cropping1D((1, 2))

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
        x = self.transposed_convolution6(x, training=training)
        x = self.transposed_convolution7(x, training=training)
        x = self.transposed_convolution8(x, training=training)
        x = self.cropping0(x, training=training)
        x = self.reshape1(x, training=training)
        return x

class Nyx14(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(4000, batch_normalization=False, dropout_rate=0.5)
        self.dense1 = DenseBlock(3500, batch_normalization=False, dropout_rate=0.5)
        self.dense2 = DenseBlock(3000, batch_normalization=False, dropout_rate=0.5)
        self.dense3 = DenseBlock(2500, batch_normalization=False, dropout_rate=0.5)
        self.dense4 = DenseBlock(2000, batch_normalization=False, dropout_rate=0.5)
        self.dense5 = DenseBlock(1500, batch_normalization=False, dropout_rate=0.5)
        self.dense6 = DenseBlock(1000, batch_normalization=False, dropout_rate=0.5)
        self.dense7 = DenseBlock(750, batch_normalization=False)
        self.dense8 = DenseBlock(500, batch_normalization=False, spatial=True)
        self.reshape0 = Reshape([1, 500])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400, kernel_size=2, strides=1, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=200, kernel_size=3, strides=1, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=120, kernel_size=4, strides=1, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=60, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=30, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0)
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
        x = self.dense6(x, training=training)
        x = self.dense7(x, training=training)
        x = self.dense8(x, training=training)
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


class Nyx15(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(1000, batch_normalization=False)
        self.dense1 = DenseBlock(1000, batch_normalization=False)
        self.dense2 = DenseBlock(900, batch_normalization=True)
        self.dense3 = DenseBlock(900, batch_normalization=True)
        self.dense4 = DenseBlock(800, batch_normalization=True)
        self.dense5 = DenseBlock(800, batch_normalization=True)
        self.dense6 = DenseBlock(700, batch_normalization=True)
        self.dense7 = DenseBlock(700, batch_normalization=True)
        self.dense8 = DenseBlock(600, batch_normalization=True)
        self.dense9 = DenseBlock(600, batch_normalization=True)
        self.dense10 = DenseBlock(500, batch_normalization=True)
        self.dense11 = DenseBlock(500, batch_normalization=True, spatial=True)
        self.reshape0 = Reshape([1, 500])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400, kernel_size=4, strides=1,
                                                            batch_normalization=True)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=350, kernel_size=4, strides=1,
                                                            batch_normalization=True)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=300, kernel_size=4, strides=1,
                                                            batch_normalization=True)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=250, kernel_size=4, strides=2,
                                                            batch_normalization=True)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=200, kernel_size=4, strides=1,
                                                            batch_normalization=True)
        self.transposed_convolution5 = Conv1DTransposeBlock(filters=150, kernel_size=4, strides=1,
                                                            batch_normalization=False)
        self.transposed_convolution6 = Conv1DTransposeBlock(filters=100, kernel_size=4, strides=1,
                                                            batch_normalization=False)
        self.transposed_convolution7 = Conv1DTransposeBlock(filters=50, kernel_size=4, strides=2,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution8 = Conv1DTranspose(filters=1, kernel_size=4, strides=1, activation='linear')
        self.reshape1 = Reshape([64])
        self.cropping0 = Cropping1D((1, 2))

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
        x = self.dense6(x, training=training)
        x = self.dense7(x, training=training)
        x = self.dense8(x, training=training)
        x = self.dense9(x, training=training)
        x = self.dense10(x, training=training)
        x = self.dense11(x, training=training)
        x = self.reshape0(x, training=training)
        x = self.transposed_convolution0(x, training=training)
        x = self.transposed_convolution1(x, training=training)
        x = self.transposed_convolution2(x, training=training)
        x = self.transposed_convolution3(x, training=training)
        x = self.transposed_convolution4(x, training=training)
        x = self.transposed_convolution5(x, training=training)
        x = self.transposed_convolution6(x, training=training)
        x = self.transposed_convolution7(x, training=training)
        x = self.transposed_convolution8(x, training=training)
        x = self.cropping0(x, training=training)
        x = self.reshape1(x, training=training)
        return x


class Nyx16(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(1000, batch_normalization=False, dropout_rate=0.5)
        self.dense1 = DenseBlock(1000, batch_normalization=False, dropout_rate=0.5)
        self.dense2 = DenseBlock(900, batch_normalization=True, dropout_rate=0.5)
        self.dense3 = DenseBlock(900, batch_normalization=True, dropout_rate=0.5)
        self.dense4 = DenseBlock(800, batch_normalization=True, dropout_rate=0.5)
        self.dense5 = DenseBlock(800, batch_normalization=True, dropout_rate=0.5)
        self.dense6 = DenseBlock(700, batch_normalization=True, dropout_rate=0.5)
        self.dense7 = DenseBlock(700, batch_normalization=True, dropout_rate=0.5)
        self.dense8 = DenseBlock(600, batch_normalization=True, dropout_rate=0.5)
        self.dense9 = DenseBlock(600, batch_normalization=True, dropout_rate=0.5)
        self.dense10 = DenseBlock(500, batch_normalization=True, dropout_rate=0.5)
        self.dense11 = DenseBlock(500, batch_normalization=True, spatial=True, dropout_rate=0.5)
        self.reshape0 = Reshape([1, 500])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400, kernel_size=4, strides=1,
                                                            batch_normalization=True, dropout_rate=0.5)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=350, kernel_size=4, strides=1,
                                                            batch_normalization=True, dropout_rate=0.5)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=300, kernel_size=4, strides=1,
                                                            batch_normalization=True, dropout_rate=0.5)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=250, kernel_size=4, strides=2,
                                                            batch_normalization=True, dropout_rate=0.5)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=200, kernel_size=4, strides=1,
                                                            batch_normalization=True, dropout_rate=0.5)
        self.transposed_convolution5 = Conv1DTransposeBlock(filters=150, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.5)
        self.transposed_convolution6 = Conv1DTransposeBlock(filters=100, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.5)
        self.transposed_convolution7 = Conv1DTransposeBlock(filters=50, kernel_size=4, strides=2,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution8 = Conv1DTranspose(filters=1, kernel_size=4, strides=1, activation='linear')
        self.reshape1 = Reshape([64])
        self.cropping0 = Cropping1D((1, 2))

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
        x = self.dense6(x, training=training)
        x = self.dense7(x, training=training)
        x = self.dense8(x, training=training)
        x = self.dense9(x, training=training)
        x = self.dense10(x, training=training)
        x = self.dense11(x, training=training)
        x = self.reshape0(x, training=training)
        x = self.transposed_convolution0(x, training=training)
        x = self.transposed_convolution1(x, training=training)
        x = self.transposed_convolution2(x, training=training)
        x = self.transposed_convolution3(x, training=training)
        x = self.transposed_convolution4(x, training=training)
        x = self.transposed_convolution5(x, training=training)
        x = self.transposed_convolution6(x, training=training)
        x = self.transposed_convolution7(x, training=training)
        x = self.transposed_convolution8(x, training=training)
        x = self.cropping0(x, training=training)
        x = self.reshape1(x, training=training)
        return x

class Nyx17(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(1000, batch_normalization=False, dropout_rate=0.5)
        self.dense1 = DenseBlock(1000, batch_normalization=False, dropout_rate=0.5)
        self.dense2 = DenseBlock(900, batch_normalization=False, dropout_rate=0.5)
        self.dense3 = DenseBlock(900, batch_normalization=False, dropout_rate=0.5)
        self.dense4 = DenseBlock(800, batch_normalization=False, dropout_rate=0.5)
        self.dense5 = DenseBlock(800, batch_normalization=False, dropout_rate=0.5)
        self.dense6 = DenseBlock(700, batch_normalization=False, dropout_rate=0.5)
        self.dense7 = DenseBlock(700, batch_normalization=False, dropout_rate=0.5)
        self.dense8 = DenseBlock(600, batch_normalization=False, dropout_rate=0.5)
        self.dense9 = DenseBlock(600, batch_normalization=False, dropout_rate=0.5)
        self.dense10 = DenseBlock(500, batch_normalization=False, dropout_rate=0.5)
        self.dense11 = DenseBlock(500, batch_normalization=False, spatial=True, dropout_rate=0.5)
        self.reshape0 = Reshape([1, 500])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.5)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=350, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.5)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=300, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.5)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=250, kernel_size=4, strides=2,
                                                            batch_normalization=False, dropout_rate=0.5)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=200, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.5)
        self.transposed_convolution5 = Conv1DTransposeBlock(filters=150, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.5)
        self.transposed_convolution6 = Conv1DTransposeBlock(filters=100, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.5)
        self.transposed_convolution7 = Conv1DTransposeBlock(filters=50, kernel_size=4, strides=2,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution8 = Conv1DTranspose(filters=1, kernel_size=4, strides=1, activation='linear')
        self.reshape1 = Reshape([64])
        self.cropping0 = Cropping1D((1, 2))

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
        x = self.dense6(x, training=training)
        x = self.dense7(x, training=training)
        x = self.dense8(x, training=training)
        x = self.dense9(x, training=training)
        x = self.dense10(x, training=training)
        x = self.dense11(x, training=training)
        x = self.reshape0(x, training=training)
        x = self.transposed_convolution0(x, training=training)
        x = self.transposed_convolution1(x, training=training)
        x = self.transposed_convolution2(x, training=training)
        x = self.transposed_convolution3(x, training=training)
        x = self.transposed_convolution4(x, training=training)
        x = self.transposed_convolution5(x, training=training)
        x = self.transposed_convolution6(x, training=training)
        x = self.transposed_convolution7(x, training=training)
        x = self.transposed_convolution8(x, training=training)
        x = self.cropping0(x, training=training)
        x = self.reshape1(x, training=training)
        return x


class Nyx18(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(1000*4, batch_normalization=False, dropout_rate=0.5)
        self.dense1 = DenseBlock(1000*4, batch_normalization=False, dropout_rate=0.5)
        self.dense2 = DenseBlock(900*4, batch_normalization=False, dropout_rate=0.5)
        self.dense3 = DenseBlock(900*4, batch_normalization=False, dropout_rate=0.5)
        self.dense4 = DenseBlock(800*4, batch_normalization=False, dropout_rate=0.5)
        self.dense5 = DenseBlock(800*4, batch_normalization=False, dropout_rate=0.5)
        self.dense6 = DenseBlock(700*4, batch_normalization=False, dropout_rate=0.5)
        self.dense7 = DenseBlock(700*4, batch_normalization=False, dropout_rate=0.5)
        self.dense8 = DenseBlock(600*4, batch_normalization=False, dropout_rate=0.5)
        self.dense9 = DenseBlock(600*4, batch_normalization=False, dropout_rate=0.5)
        self.dense10 = DenseBlock(500*4, batch_normalization=False, dropout_rate=0.5)
        self.dense11 = DenseBlock(500*4, batch_normalization=False, spatial=True, dropout_rate=0.5)
        self.reshape0 = Reshape([1, 500*4])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400*4, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.5)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=350*4, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.5)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=300*4, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.5)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=250*4, kernel_size=4, strides=2,
                                                            batch_normalization=False, dropout_rate=0.5)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=200*4, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.5)
        self.transposed_convolution5 = Conv1DTransposeBlock(filters=150*4, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.5)
        self.transposed_convolution6 = Conv1DTransposeBlock(filters=100*4, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.5)
        self.transposed_convolution7 = Conv1DTransposeBlock(filters=50*4, kernel_size=4, strides=2,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution8 = Conv1DTranspose(filters=1, kernel_size=4, strides=1, activation='linear')
        self.reshape1 = Reshape([64])
        self.cropping0 = Cropping1D((1, 2))

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
        x = self.dense6(x, training=training)
        x = self.dense7(x, training=training)
        x = self.dense8(x, training=training)
        x = self.dense9(x, training=training)
        x = self.dense10(x, training=training)
        x = self.dense11(x, training=training)
        x = self.reshape0(x, training=training)
        x = self.transposed_convolution0(x, training=training)
        x = self.transposed_convolution1(x, training=training)
        x = self.transposed_convolution2(x, training=training)
        x = self.transposed_convolution3(x, training=training)
        x = self.transposed_convolution4(x, training=training)
        x = self.transposed_convolution5(x, training=training)
        x = self.transposed_convolution6(x, training=training)
        x = self.transposed_convolution7(x, training=training)
        x = self.transposed_convolution8(x, training=training)
        x = self.cropping0(x, training=training)
        x = self.reshape1(x, training=training)
        return x

class Nyx19(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(1000*4, batch_normalization=False, dropout_rate=0.5)
        self.dense1 = DenseBlock(1000*4, batch_normalization=False, dropout_rate=0.5)
        self.dense2 = DenseBlock(900*4, batch_normalization=True, dropout_rate=0.5)
        self.dense3 = DenseBlock(900*4, batch_normalization=True, dropout_rate=0.5)
        self.dense4 = DenseBlock(800*4, batch_normalization=True, dropout_rate=0.5)
        self.dense5 = DenseBlock(800*4, batch_normalization=True, dropout_rate=0.5)
        self.dense6 = DenseBlock(700*4, batch_normalization=True, dropout_rate=0.5)
        self.dense7 = DenseBlock(700*4, batch_normalization=True, dropout_rate=0.5)
        self.dense8 = DenseBlock(600*4, batch_normalization=True, dropout_rate=0.5)
        self.dense9 = DenseBlock(600*4, batch_normalization=True, dropout_rate=0.5)
        self.dense10 = DenseBlock(500*4, batch_normalization=True, dropout_rate=0.5)
        self.dense11 = DenseBlock(500*4, batch_normalization=True, spatial=True, dropout_rate=0.5)
        self.reshape0 = Reshape([1, 500*4])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400*4, kernel_size=4, strides=1,
                                                            batch_normalization=True, dropout_rate=0.5)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=350*4, kernel_size=4, strides=1,
                                                            batch_normalization=True, dropout_rate=0.5)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=300*4, kernel_size=4, strides=1,
                                                            batch_normalization=True, dropout_rate=0.5)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=250*4, kernel_size=4, strides=2,
                                                            batch_normalization=True, dropout_rate=0.5)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=200*4, kernel_size=4, strides=1,
                                                            batch_normalization=True, dropout_rate=0.5)
        self.transposed_convolution5 = Conv1DTransposeBlock(filters=150*4, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.5)
        self.transposed_convolution6 = Conv1DTransposeBlock(filters=100*4, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.5)
        self.transposed_convolution7 = Conv1DTransposeBlock(filters=50*4, kernel_size=4, strides=2,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution8 = Conv1DTranspose(filters=1, kernel_size=4, strides=1, activation='linear')
        self.reshape1 = Reshape([64])
        self.cropping0 = Cropping1D((1, 2))

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
        x = self.dense6(x, training=training)
        x = self.dense7(x, training=training)
        x = self.dense8(x, training=training)
        x = self.dense9(x, training=training)
        x = self.dense10(x, training=training)
        x = self.dense11(x, training=training)
        x = self.reshape0(x, training=training)
        x = self.transposed_convolution0(x, training=training)
        x = self.transposed_convolution1(x, training=training)
        x = self.transposed_convolution2(x, training=training)
        x = self.transposed_convolution3(x, training=training)
        x = self.transposed_convolution4(x, training=training)
        x = self.transposed_convolution5(x, training=training)
        x = self.transposed_convolution6(x, training=training)
        x = self.transposed_convolution7(x, training=training)
        x = self.transposed_convolution8(x, training=training)
        x = self.cropping0(x, training=training)
        x = self.reshape1(x, training=training)
        return x

class Nyx20(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(1000*4, batch_normalization=False, dropout_rate=0.9)
        self.dense1 = DenseBlock(1000*4, batch_normalization=False, dropout_rate=0.9)
        self.dense2 = DenseBlock(900*4, batch_normalization=False, dropout_rate=0.9)
        self.dense3 = DenseBlock(900*4, batch_normalization=False, dropout_rate=0.9)
        self.dense4 = DenseBlock(800*4, batch_normalization=False, dropout_rate=0.9)
        self.dense5 = DenseBlock(800*4, batch_normalization=False, dropout_rate=0.9)
        self.dense6 = DenseBlock(700*4, batch_normalization=False, dropout_rate=0.9)
        self.dense7 = DenseBlock(700*4, batch_normalization=False, dropout_rate=0.9)
        self.dense8 = DenseBlock(600*4, batch_normalization=False, dropout_rate=0.9)
        self.dense9 = DenseBlock(600*4, batch_normalization=False, dropout_rate=0.9)
        self.dense10 = DenseBlock(500*4, batch_normalization=False, dropout_rate=0.9)
        self.dense11 = DenseBlock(500*4, batch_normalization=False, spatial=True, dropout_rate=0.9)
        self.reshape0 = Reshape([1, 500*4])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400*4, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.9)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=350*4, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.9)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=300*4, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.9)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=250*4, kernel_size=4, strides=2,
                                                            batch_normalization=False, dropout_rate=0.9)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=200*4, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.9)
        self.transposed_convolution5 = Conv1DTransposeBlock(filters=150*4, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.9)
        self.transposed_convolution6 = Conv1DTransposeBlock(filters=100*4, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.9)
        self.transposed_convolution7 = Conv1DTransposeBlock(filters=50*4, kernel_size=4, strides=2,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution8 = Conv1DTranspose(filters=1, kernel_size=4, strides=1, activation='linear')
        self.reshape1 = Reshape([64])
        self.cropping0 = Cropping1D((1, 2))

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
        x = self.dense6(x, training=training)
        x = self.dense7(x, training=training)
        x = self.dense8(x, training=training)
        x = self.dense9(x, training=training)
        x = self.dense10(x, training=training)
        x = self.dense11(x, training=training)
        x = self.reshape0(x, training=training)
        x = self.transposed_convolution0(x, training=training)
        x = self.transposed_convolution1(x, training=training)
        x = self.transposed_convolution2(x, training=training)
        x = self.transposed_convolution3(x, training=training)
        x = self.transposed_convolution4(x, training=training)
        x = self.transposed_convolution5(x, training=training)
        x = self.transposed_convolution6(x, training=training)
        x = self.transposed_convolution7(x, training=training)
        x = self.transposed_convolution8(x, training=training)
        x = self.cropping0(x, training=training)
        x = self.reshape1(x, training=training)
        return x


class Nyx21(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(1000*4, batch_normalization=False, dropout_rate=0.9)
        self.dense1 = DenseBlock(1000*4, batch_normalization=False, dropout_rate=0.9)
        self.dense2 = DenseBlock(900*4, batch_normalization=False, dropout_rate=0.9)
        self.dense3 = DenseBlock(900*4, batch_normalization=False, dropout_rate=0.9)
        self.dense4 = DenseBlock(800*4, batch_normalization=False, dropout_rate=0.9)
        self.dense5 = DenseBlock(800*4, batch_normalization=False, dropout_rate=0.9)
        self.dense6 = DenseBlock(700*4, batch_normalization=False, dropout_rate=0)
        self.dense7 = DenseBlock(700*4, batch_normalization=False, dropout_rate=0)
        self.dense8 = DenseBlock(600*4, batch_normalization=False, dropout_rate=0)
        self.dense9 = DenseBlock(600*4, batch_normalization=False, dropout_rate=0)
        self.dense10 = DenseBlock(500*4, batch_normalization=False, dropout_rate=0)
        self.dense11 = DenseBlock(500*4, batch_normalization=False, spatial=True, dropout_rate=0)
        self.reshape0 = Reshape([1, 500*4])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400*4, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=350*4, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=300*4, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=250*4, kernel_size=4, strides=2,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=200*4, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution5 = Conv1DTransposeBlock(filters=150*4, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution6 = Conv1DTransposeBlock(filters=100*4, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution7 = Conv1DTransposeBlock(filters=50*4, kernel_size=4, strides=2,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution8 = Conv1DTranspose(filters=1, kernel_size=4, strides=1, activation='linear')
        self.reshape1 = Reshape([64])
        self.cropping0 = Cropping1D((1, 2))

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
        x = self.dense6(x, training=training)
        x = self.dense7(x, training=training)
        x = self.dense8(x, training=training)
        x = self.dense9(x, training=training)
        x = self.dense10(x, training=training)
        x = self.dense11(x, training=training)
        x = self.reshape0(x, training=training)
        x = self.transposed_convolution0(x, training=training)
        x = self.transposed_convolution1(x, training=training)
        x = self.transposed_convolution2(x, training=training)
        x = self.transposed_convolution3(x, training=training)
        x = self.transposed_convolution4(x, training=training)
        x = self.transposed_convolution5(x, training=training)
        x = self.transposed_convolution6(x, training=training)
        x = self.transposed_convolution7(x, training=training)
        x = self.transposed_convolution8(x, training=training)
        x = self.cropping0(x, training=training)
        x = self.reshape1(x, training=training)
        return x

class Nyx22(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(1000, batch_normalization=False, dropout_rate=0.5)
        self.dense1 = DenseBlock(1000, batch_normalization=False, dropout_rate=0.5)
        self.dense2 = DenseBlock(900, batch_normalization=True, dropout_rate=0.5)
        self.dense3 = DenseBlock(900, batch_normalization=True, dropout_rate=0.5)
        self.dense4 = DenseBlock(800, batch_normalization=True, dropout_rate=0.5)
        self.dense5 = DenseBlock(800, batch_normalization=True, dropout_rate=0.5)
        self.dense6 = DenseBlock(700, batch_normalization=True, dropout_rate=0)
        self.dense7 = DenseBlock(700, batch_normalization=True, dropout_rate=0)
        self.dense8 = DenseBlock(600, batch_normalization=True, dropout_rate=0)
        self.dense9 = DenseBlock(600, batch_normalization=True, dropout_rate=0)
        self.dense10 = DenseBlock(500, batch_normalization=True, dropout_rate=0)
        self.dense11 = DenseBlock(500, batch_normalization=True, spatial=True, dropout_rate=0)
        self.reshape0 = Reshape([1, 500])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400, kernel_size=4, strides=1,
                                                            batch_normalization=True, dropout_rate=0)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=350, kernel_size=4, strides=1,
                                                            batch_normalization=True, dropout_rate=0)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=300, kernel_size=4, strides=1,
                                                            batch_normalization=True, dropout_rate=0)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=250, kernel_size=4, strides=2,
                                                            batch_normalization=True, dropout_rate=0)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=200, kernel_size=4, strides=1,
                                                            batch_normalization=True, dropout_rate=0)
        self.transposed_convolution5 = Conv1DTransposeBlock(filters=150, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution6 = Conv1DTransposeBlock(filters=100, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution7 = Conv1DTransposeBlock(filters=50, kernel_size=4, strides=2,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution8 = Conv1DTranspose(filters=1, kernel_size=4, strides=1, activation='linear')
        self.reshape1 = Reshape([64])
        self.cropping0 = Cropping1D((1, 2))

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
        x = self.dense6(x, training=training)
        x = self.dense7(x, training=training)
        x = self.dense8(x, training=training)
        x = self.dense9(x, training=training)
        x = self.dense10(x, training=training)
        x = self.dense11(x, training=training)
        x = self.reshape0(x, training=training)
        x = self.transposed_convolution0(x, training=training)
        x = self.transposed_convolution1(x, training=training)
        x = self.transposed_convolution2(x, training=training)
        x = self.transposed_convolution3(x, training=training)
        x = self.transposed_convolution4(x, training=training)
        x = self.transposed_convolution5(x, training=training)
        x = self.transposed_convolution6(x, training=training)
        x = self.transposed_convolution7(x, training=training)
        x = self.transposed_convolution8(x, training=training)
        x = self.cropping0(x, training=training)
        x = self.reshape1(x, training=training)
        return x

class Nyx23(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(1000, batch_normalization=False, dropout_rate=0.5)
        self.dense1 = DenseBlock(1000, batch_normalization=False, dropout_rate=0.5)
        self.dense2 = DenseBlock(900, batch_normalization=True, dropout_rate=0.5)
        self.dense3 = DenseBlock(900, batch_normalization=True, dropout_rate=0.5)
        self.dense4 = DenseBlock(800, batch_normalization=True, dropout_rate=0.5)
        self.dense5 = DenseBlock(800, batch_normalization=True, dropout_rate=0.5)
        self.dense6 = DenseBlock(700, batch_normalization=True, dropout_rate=0)
        self.dense7 = DenseBlock(700, batch_normalization=True, dropout_rate=0)
        self.dense8 = DenseBlock(600, batch_normalization=True, dropout_rate=0)
        self.dense9 = DenseBlock(600, batch_normalization=True, dropout_rate=0)
        self.dense10 = DenseBlock(500, batch_normalization=True, dropout_rate=0)
        self.dense11 = DenseBlock(500, batch_normalization=True, spatial=True, dropout_rate=0)
        self.reshape0 = Reshape([1, 500])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400, kernel_size=4, strides=1,
                                                            batch_normalization=True, dropout_rate=0)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=350, kernel_size=4, strides=1,
                                                            batch_normalization=True, dropout_rate=0)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=300, kernel_size=4, strides=1,
                                                            batch_normalization=True, dropout_rate=0)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=250, kernel_size=4, strides=2,
                                                            batch_normalization=True, dropout_rate=0)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=200, kernel_size=4, strides=1,
                                                            batch_normalization=True, dropout_rate=0)
        self.transposed_convolution5 = Conv1DTransposeBlock(filters=150, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution6 = Conv1DTransposeBlock(filters=100, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution7 = Conv1DTransposeBlock(filters=50, kernel_size=4, strides=2,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution8 = Conv1DTransposeBlock(filters=1, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.reshape1 = Reshape([64])
        self.cropping0 = Cropping1D((1, 2))

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
        x = self.dense6(x, training=training)
        x = self.dense7(x, training=training)
        x = self.dense8(x, training=training)
        x = self.dense9(x, training=training)
        x = self.dense10(x, training=training)
        x = self.dense11(x, training=training)
        x = self.reshape0(x, training=training)
        x = self.transposed_convolution0(x, training=training)
        x = self.transposed_convolution1(x, training=training)
        x = self.transposed_convolution2(x, training=training)
        x = self.transposed_convolution3(x, training=training)
        x = self.transposed_convolution4(x, training=training)
        x = self.transposed_convolution5(x, training=training)
        x = self.transposed_convolution6(x, training=training)
        x = self.transposed_convolution7(x, training=training)
        x = self.transposed_convolution8(x, training=training)
        x = self.cropping0(x, training=training)
        x = self.reshape1(x, training=training)
        return x
    
class Eos0(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(20, batch_normalization=False, dropout_rate=0)
        self.dense1 = DenseBlock(30, batch_normalization=False, dropout_rate=0)
        self.dense2 = DenseBlock(40, batch_normalization=False, dropout_rate=0)
        self.dense3 = DenseBlock(50, batch_normalization=False, dropout_rate=0)
        self.dense4 = DenseBlock(100, batch_normalization=False, dropout_rate=0.5)
        self.dense5 = DenseBlock(100, batch_normalization=False, dropout_rate=0.5)
        self.dense6 = DenseBlock(200, batch_normalization=False, dropout_rate=0.5)
        self.dense7 = DenseBlock(200, batch_normalization=False, dropout_rate=0.5)
        self.dense8 = DenseBlock(300, batch_normalization=False, dropout_rate=0.5)
        self.dense9 = DenseBlock(300, batch_normalization=False, dropout_rate=0.5)
        self.dense10 = DenseBlock(500, batch_normalization=False, dropout_rate=0.5)
        self.dense11 = DenseBlock(500, batch_normalization=False, spatial=True, dropout_rate=0.5)
        self.reshape0 = Reshape([1, 500])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=350, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=300, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=250, kernel_size=4, strides=2,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=200, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution5 = Conv1DTransposeBlock(filters=150, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution6 = Conv1DTransposeBlock(filters=100, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution7 = Conv1DTransposeBlock(filters=50, kernel_size=4, strides=2,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution8 = Conv1DTransposeBlock(filters=1, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.reshape1 = Reshape([64])
        self.cropping0 = Cropping1D((1, 2))

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
        x = self.dense6(x, training=training)
        x = self.dense7(x, training=training)
        x = self.dense8(x, training=training)
        x = self.dense9(x, training=training)
        x = self.dense10(x, training=training)
        x = self.dense11(x, training=training)
        x = self.reshape0(x, training=training)
        x = self.transposed_convolution0(x, training=training)
        x = self.transposed_convolution1(x, training=training)
        x = self.transposed_convolution2(x, training=training)
        x = self.transposed_convolution3(x, training=training)
        x = self.transposed_convolution4(x, training=training)
        x = self.transposed_convolution5(x, training=training)
        x = self.transposed_convolution6(x, training=training)
        x = self.transposed_convolution7(x, training=training)
        x = self.transposed_convolution8(x, training=training)
        x = self.cropping0(x, training=training)
        x = self.reshape1(x, training=training)
        return x

class Eos1(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(20, batch_normalization=False, dropout_rate=0)
        self.dense1 = DenseBlock(30, batch_normalization=False, dropout_rate=0)
        self.dense2 = DenseBlock(40, batch_normalization=False, dropout_rate=0)
        self.dense3 = DenseBlock(50, batch_normalization=False, dropout_rate=0)
        self.dense4 = DenseBlock(60, batch_normalization=False, dropout_rate=0)
        self.dense5 = DenseBlock(60, batch_normalization=False, dropout_rate=0)
        self.dense6 = DenseBlock(60, batch_normalization=False, dropout_rate=0)
        self.dense7 = DenseBlock(60, batch_normalization=False, dropout_rate=0)
        self.dense8 = DenseBlock(60, batch_normalization=False, dropout_rate=0)
        self.dense9 = DenseBlock(60, batch_normalization=False, dropout_rate=0)
        self.dense10 = DenseBlock(60, batch_normalization=False, dropout_rate=0)
        self.dense11 = DenseBlock(60, batch_normalization=False, spatial=True, dropout_rate=0)
        self.reshape0 = Reshape([1, 60])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=60, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=60, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=60, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=50, kernel_size=4, strides=2,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=40, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution5 = Conv1DTransposeBlock(filters=30, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution6 = Conv1DTransposeBlock(filters=20, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution7 = Conv1DTransposeBlock(filters=10, kernel_size=4, strides=2,
                                                            batch_normalization=False, dropout_rate=0)
        self.transposed_convolution8 = Conv1DTransposeBlock(filters=1, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.reshape1 = Reshape([64])
        self.cropping0 = Cropping1D((1, 2))

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
        x = self.dense6(x, training=training)
        x = self.dense7(x, training=training)
        x = self.dense8(x, training=training)
        x = self.dense9(x, training=training)
        x = self.dense10(x, training=training)
        x = self.dense11(x, training=training)
        x = self.reshape0(x, training=training)
        x = self.transposed_convolution0(x, training=training)
        x = self.transposed_convolution1(x, training=training)
        x = self.transposed_convolution2(x, training=training)
        x = self.transposed_convolution3(x, training=training)
        x = self.transposed_convolution4(x, training=training)
        x = self.transposed_convolution5(x, training=training)
        x = self.transposed_convolution6(x, training=training)
        x = self.transposed_convolution7(x, training=training)
        x = self.transposed_convolution8(x, training=training)
        x = self.cropping0(x, training=training)
        x = self.reshape1(x, training=training)
        return x

class Eos2(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(20, batch_normalization=False, dropout_rate=0.1)
        self.dense1 = DenseBlock(30, batch_normalization=False, dropout_rate=0.1)
        self.dense2 = DenseBlock(40, batch_normalization=False, dropout_rate=0.1)
        self.dense3 = DenseBlock(50, batch_normalization=False, dropout_rate=0.1)
        self.dense4 = DenseBlock(60, batch_normalization=False, dropout_rate=0.1)
        self.dense5 = DenseBlock(60, batch_normalization=False, dropout_rate=0.1)
        self.dense6 = DenseBlock(60, batch_normalization=False, dropout_rate=0.1)
        self.dense7 = DenseBlock(60, batch_normalization=False, dropout_rate=0.1)
        self.dense8 = DenseBlock(60, batch_normalization=False, dropout_rate=0.1)
        self.dense9 = DenseBlock(60, batch_normalization=False, dropout_rate=0.1)
        self.dense10 = DenseBlock(60, batch_normalization=False, dropout_rate=0.1)
        self.dense11 = DenseBlock(60, batch_normalization=False, spatial=True, dropout_rate=0.1)
        self.reshape0 = Reshape([1, 60])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=60, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.1)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=60, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.1)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=60, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.1)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=50, kernel_size=4, strides=2,
                                                            batch_normalization=False, dropout_rate=0.1)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=40, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.1)
        self.transposed_convolution5 = Conv1DTransposeBlock(filters=30, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.1)
        self.transposed_convolution6 = Conv1DTransposeBlock(filters=20, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0.1)
        self.transposed_convolution7 = Conv1DTransposeBlock(filters=10, kernel_size=4, strides=2,
                                                            batch_normalization=False, dropout_rate=0.1)
        self.transposed_convolution8 = Conv1DTransposeBlock(filters=1, kernel_size=4, strides=1,
                                                            batch_normalization=False, dropout_rate=0)
        self.reshape1 = Reshape([64])
        self.cropping0 = Cropping1D((1, 2))

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
        x = self.dense6(x, training=training)
        x = self.dense7(x, training=training)
        x = self.dense8(x, training=training)
        x = self.dense9(x, training=training)
        x = self.dense10(x, training=training)
        x = self.dense11(x, training=training)
        x = self.reshape0(x, training=training)
        x = self.transposed_convolution0(x, training=training)
        x = self.transposed_convolution1(x, training=training)
        x = self.transposed_convolution2(x, training=training)
        x = self.transposed_convolution3(x, training=training)
        x = self.transposed_convolution4(x, training=training)
        x = self.transposed_convolution5(x, training=training)
        x = self.transposed_convolution6(x, training=training)
        x = self.transposed_convolution7(x, training=training)
        x = self.transposed_convolution8(x, training=training)
        x = self.cropping0(x, training=training)
        x = self.reshape1(x, training=training)
        return x

class Nyx9Narrow(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(4000//2, batch_normalization=False, dropout_rate=0.5)
        self.dense1 = DenseBlock(3000//2, batch_normalization=False, dropout_rate=0.5)
        self.dense2 = DenseBlock(2000//2, batch_normalization=False, dropout_rate=0.5)
        self.dense3 = DenseBlock(1000//2, batch_normalization=False, dropout_rate=0.5)
        self.dense4 = DenseBlock(750//2, batch_normalization=False)
        self.dense5 = DenseBlock(500//2, batch_normalization=False, spatial=True)
        self.reshape0 = Reshape([1, 500//2])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400//2, kernel_size=2, strides=1, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=200//2, kernel_size=3, strides=1, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=120//2, kernel_size=4, strides=1, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=60//2, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=30//2, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0)
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

class Nyx9Wide(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(4000*2, batch_normalization=False, dropout_rate=0)
        self.dense1 = DenseBlock(3000*2, batch_normalization=False, dropout_rate=0)
        self.dense2 = DenseBlock(2000*2, batch_normalization=False, dropout_rate=0)
        self.dense3 = DenseBlock(1000*2, batch_normalization=False, dropout_rate=0)
        self.dense4 = DenseBlock(750*2, batch_normalization=False, dropout_rate=0)
        self.dense5 = DenseBlock(500*2, batch_normalization=False, spatial=True, dropout_rate=0)
        self.reshape0 = Reshape([1, 500*2])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400*2, kernel_size=2, strides=1, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=200*2, kernel_size=3, strides=1, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=120*2, kernel_size=4, strides=1, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=60*2, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=30*2, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0)
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

class Nyx9Wider(Model):
    def __init__(self):
        super().__init__()
        self.dense0 = DenseBlock(4000*4, batch_normalization=False, dropout_rate=0.0)
        self.dense1 = DenseBlock(3000*4, batch_normalization=False, dropout_rate=0.0)
        self.dense2 = DenseBlock(2000*4, batch_normalization=False, dropout_rate=0.0)
        self.dense3 = DenseBlock(1000*4, batch_normalization=False, dropout_rate=0.0)
        self.dense4 = DenseBlock(750*4, batch_normalization=False, dropout_rate=0.0)
        self.dense5 = DenseBlock(500*4, batch_normalization=False, spatial=True, dropout_rate=0.0)
        # self.noise0 = GaussianNoise(0.05)
        self.reshape0 = Reshape([1, 500*4])
        self.transposed_convolution0 = Conv1DTransposeBlock(filters=400*4, kernel_size=2, strides=1, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution1 = Conv1DTransposeBlock(filters=200*4, kernel_size=3, strides=1, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution2 = Conv1DTransposeBlock(filters=120*4, kernel_size=4, strides=1, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution3 = Conv1DTransposeBlock(filters=60*4, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0)
        self.transposed_convolution4 = Conv1DTransposeBlock(filters=30*4, kernel_size=4, strides=2, batch_normalization=False, dropout_rate=0)
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
        # ones = tf.ones([1, 11])
        # noise_multiplier = self.noise0(ones, training=training)
        # x = x * noise_multiplier
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
