"""
Code for custom losses for the ramjet package.
"""
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.python.framework import ops
from tensorflow.keras import backend
from tensorflow.python.ops import math_ops


class RelativeMeanSquaredErrorLoss(Loss):
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        flux_maximum = backend.max(y_true, axis=1, keepdims=True)
        flux_minimum = backend.min(y_true, axis=1, keepdims=True)
        true_normalized_0_to_1_output = (y_true - flux_minimum) / (flux_maximum - flux_minimum)
        true_normalized_negative_1_to_1_output = (true_normalized_0_to_1_output * 2) - 1
        pred_normalized_0_to_1_output = (y_pred - flux_minimum) / (flux_maximum - flux_minimum)
        pred_normalized_negative_1_to_1_output = (pred_normalized_0_to_1_output * 2) - 1
        error = true_normalized_negative_1_to_1_output - pred_normalized_negative_1_to_1_output
        squared_error = error ** 2
        mean_squared_error = backend.mean(squared_error)
        return mean_squared_error
