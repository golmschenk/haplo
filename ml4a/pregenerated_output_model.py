from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import sigmoid, reduce_mean, reduce_max, reduce_min, reduce_sum
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, Convolution1D, MaxPooling1D, AveragePooling1D, Permute, LeakyReLU, Dense, \
    Dropout, Layer
from tensorflow.keras.regularizers import L2


class DenseBlock(Layer):
    def __init__(self, output_channels: int, l2_regularization: float = 0.0001, dropout_rate: float = 0.5):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        if l2_regularization > 0:
            l2_regularizer = L2(l2_regularization)
        else:
            l2_regularizer = None
        self.dense = Dense(output_channels, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        if dropout_rate > 0:
            self.dropout_layer = Dropout(rate=dropout_rate)
        else:
            self.dropout_layer = None

    def call(self, inputs, training=False, mask=None):
        x = inputs
        x = self.dense(x)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        return x


class PregeneratedOutputModel(Model):
    def __init__(self, pregenerated_output_array: np.ndarray):
        super().__init__()
        assert pregenerated_output_array.shape == (200, 64)
        self.pregenerated_output = tf.constant(pregenerated_output_array, dtype=tf.float32)
        self.blocks: List[Layer] = []
        for _ in range(5):
            self.blocks.append(DenseBlock(200))
        self.scale_layer = Dense(200)
        self.offset_layer = Dense(200)
        self.reshape = Reshape([200, 1])

    def call(self, inputs, training=False, mask=None):
        x = inputs
        for block in self.blocks:
            x = block(x)
        scale = self.reshape(self.scale_layer(x))
        offset = self.reshape(self.offset_layer(x))
        x = (self.pregenerated_output * scale) + offset
        x = reduce_mean(x, axis=1)
        return x


class PregeneratedOutputModelNoDo(Model):
    def __init__(self, pregenerated_output_array: np.ndarray):
        super().__init__()
        assert pregenerated_output_array.shape == (200, 64)
        self.pregenerated_output = tf.constant(pregenerated_output_array, dtype=tf.float32)
        self.blocks: List[Layer] = []
        for _ in range(5):
            self.blocks.append(DenseBlock(200, dropout_rate=0.0))
        self.scale_layer = Dense(200)
        self.offset_layer = Dense(200)
        self.reshape = Reshape([200, 1])

    def call(self, inputs, training=False, mask=None):
        x = inputs
        for block in self.blocks:
            x = block(x)
        scale = self.reshape(self.scale_layer(x))
        offset = self.reshape(self.offset_layer(x))
        x = (self.pregenerated_output * scale) + offset
        x = reduce_mean(x, axis=1)
        return x
