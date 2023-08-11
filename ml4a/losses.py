"""
Code for custom losses for the ramjet package.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.python.framework import ops
from tensorflow.keras import backend
from tensorflow.python.ops import math_ops

from ml4a.nicer_example import NicerExample

phase_amplitude_mean = tf.constant(34025.080543335825, dtype=tf.float64)
phase_amplitude_standard_deviation = tf.constant(47698.66676993027, dtype=tf.float64)
parameter_means = tf.constant(
    [-0.0008009571736463096, -0.0008946310379428422, -2.274708783534052e-05, 1.5716876559520705,
     3.1388159291733086, -0.001410436081400537, -0.0001470613574040905, -3.793528434430451e-05,
     1.5723036365564083, 3.1463088925150258, 5.509554132916939])
parameter_standard_deviations = tf.constant(
    [0.28133126679885656, 0.28100480365686287, 0.28140136435474244, 0.907001394792043, 1.811683338833852,
     0.2815981892528909, 0.281641754864262, 0.28109705707606697, 0.9062620846468298, 1.8139690831565327,
     2.886950440590801])


def unnormalize_phase_amplitudes(phase_amplitudes):
    phase_amplitudes *= phase_amplitude_standard_deviation
    phase_amplitudes += phase_amplitude_mean
    return phase_amplitudes


class RelativeMeanSquaredErrorLoss(Loss):
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return self.relative_mean_squared_error_loss(y_true, y_pred)

    @staticmethod
    def relative_mean_squared_error_loss(y_true, y_pred):
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


class PlusOneChiSquaredStatisticLoss(Loss):
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return self.plus_one_chi_squared_statistic(y_true, y_pred)

    @staticmethod
    def plus_one_chi_squared_statistic(y_true, y_pred):
        observed = unnormalize_phase_amplitudes(tf.cast(y_pred + 1.0, dtype=tf.float64))
        expected = unnormalize_phase_amplitudes(tf.cast(y_true + 1.0, dtype=tf.float64))
        chi_squared_statistic_f64 = backend.mean(backend.sum(((observed - expected) ** 2) / expected, axis=1))
        chi_squared_statistic = tf.cast(chi_squared_statistic_f64, dtype=tf.float32)
        return chi_squared_statistic

class PlusOneChiSquaredStatisticLossUnreduced(Loss):
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return self.plus_one_chi_squared_statistic(y_true, y_pred)

    @staticmethod
    def plus_one_chi_squared_statistic(y_true, y_pred):
        observed = unnormalize_phase_amplitudes(tf.cast(y_pred + 1.0, dtype=tf.float64))
        expected = unnormalize_phase_amplitudes(tf.cast(y_true + 1.0, dtype=tf.float64))
        chi_squared_statistic_f64 = backend.sum(((observed - expected) ** 2) / expected, axis=1)
        chi_squared_statistic = tf.cast(chi_squared_statistic_f64, dtype=tf.float32)
        return chi_squared_statistic


class PlusOneChiSquaredStatisticLossNoUnnormalizing(Loss):
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return self.plus_one_chi_squared_statistic(y_true, y_pred)

    @staticmethod
    def plus_one_chi_squared_statistic(y_true, y_pred):
        observed = tf.cast(y_pred + 1.0, dtype=tf.float64)
        expected = tf.cast(y_true + 1.0, dtype=tf.float64)
        chi_squared_statistic_f64 = backend.mean(backend.sum(((observed - expected) ** 2) / expected, axis=1))
        chi_squared_statistic = tf.cast(chi_squared_statistic_f64, dtype=tf.float32)
        return chi_squared_statistic


class PlusOneChiSquaredMeanDenominatorStatisticLoss(Loss):
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return self.loss(y_true, y_pred)

    @staticmethod
    def loss(y_true, y_pred):
        observed = unnormalize_phase_amplitudes(tf.cast(y_pred + 1.0, dtype=tf.float64))
        expected = unnormalize_phase_amplitudes(tf.cast(y_true + 1.0, dtype=tf.float64))
        chi_squared_statistic_f64 = (
            backend.mean(
                backend.sum(((observed - expected) ** 2), axis=1) / backend.mean(expected, axis=1)
            )
        )
        chi_squared_statistic = tf.cast(chi_squared_statistic_f64, dtype=tf.float32)
        return chi_squared_statistic
