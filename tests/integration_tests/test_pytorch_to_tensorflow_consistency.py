import numpy as np
import torch
from torch import Tensor

from haplo.models import ResidualGenerationLightCurveNetworkBlock as PtResidualGenerationLightCurveNetworkBlock
from integration_tests.pytorch_model_analysis import get_total_size_of_parameters_in_pt_model
from integration_tests.tensorflow_model import \
    ResidualGenerationLightCurveNetworkBlock as TfResidualGenerationLightCurveNetworkBlock
from integration_tests.tensorflow_model_analysis import get_total_size_of_parameters_in_tf_model


def test_consistency_in_number_of_parameters_in_residual_light_curve_generation_block():
    batch_size = 11
    length = 7
    input_channels = 3 * 4
    output_channels = 5 * 4
    tf_block = TfResidualGenerationLightCurveNetworkBlock(input_channels=input_channels,
                                                          output_channels=output_channels, pooling_size=2)
    pt_block = PtResidualGenerationLightCurveNetworkBlock(input_channels=input_channels,
                                                          output_channels=output_channels, upsampling_scale_factor=2)

    fake_tf_input = np.zeros(shape=[batch_size, length, input_channels])
    _ = tf_block.call(fake_tf_input)

    tf_dimension_decrease_layer_size = get_total_size_of_parameters_in_tf_model(tf_block.dimension_decrease_layer)
    pt_dimension_decrease_layer_size = get_total_size_of_parameters_in_pt_model(pt_block.dimension_decrease_layer)
    assert tf_dimension_decrease_layer_size == pt_dimension_decrease_layer_size

    tf_convolutional_layer_size = get_total_size_of_parameters_in_tf_model(tf_block.convolutional_layer)
    pt_convolutional_layer_size = get_total_size_of_parameters_in_pt_model(pt_block.convolutional_layer)
    assert tf_convolutional_layer_size == pt_convolutional_layer_size

    tf_dimension_increase_layer_size = get_total_size_of_parameters_in_tf_model(tf_block.dimension_increase_layer)
    pt_dimension_increase_layer_size = get_total_size_of_parameters_in_pt_model(pt_block.dimension_increase_layer)
    assert tf_dimension_increase_layer_size == pt_dimension_increase_layer_size

    tf_size = get_total_size_of_parameters_in_tf_model(tf_block)
    pt_size = get_total_size_of_parameters_in_pt_model(pt_block)
    assert tf_size == pt_size
    assert pt_size == pt_dimension_decrease_layer_size + pt_convolutional_layer_size + pt_dimension_increase_layer_size


def test_consistency_of_output_sizes_of_residual_light_curve_generation_block():
    batch_size = 11
    length = 7
    input_channels = 3 * 4
    output_channels = 5 * 4
    tf_block = TfResidualGenerationLightCurveNetworkBlock(input_channels=input_channels,
                                                          output_channels=output_channels, pooling_size=2)
    pt_block = PtResidualGenerationLightCurveNetworkBlock(input_channels=input_channels,
                                                          output_channels=output_channels, upsampling_scale_factor=2)

    fake_tf_input = np.zeros(shape=[batch_size, length, input_channels])
    fake_tf_output = tf_block.call(fake_tf_input).numpy()

    fake_pt_input = np.zeros(shape=[batch_size, input_channels, length])
    fake_pt_output = pt_block(torch.Tensor(fake_pt_input)).detach().numpy()

    assert fake_pt_output.size == fake_tf_output.size
    assert fake_pt_output.shape[0] == fake_tf_output.shape[0]
    # TF is (batch, channel, length), PT is (batch, length, channel)
    assert fake_pt_output.shape[1] == fake_tf_output.shape[2]
    assert fake_pt_output.shape[2] == fake_tf_output.shape[1]


def test_consistency_of_output_values_for_specific_weights():
    batch_size = 11
    length = 7
    input_channels = 3 * 4
    output_channels = 5 * 4
    tf_block = TfResidualGenerationLightCurveNetworkBlock(input_channels=input_channels,
                                                          output_channels=output_channels, pooling_size=2)
    pt_block = PtResidualGenerationLightCurveNetworkBlock(input_channels=input_channels,
                                                          output_channels=output_channels, upsampling_scale_factor=2)

    random_number_generator = np.random.default_rng(seed=0)
    dimension_decrease_layer_weight = random_number_generator.normal(
        size=pt_block.dimension_decrease_layer.weight.shape)
    dimension_decrease_layer_bias = random_number_generator.normal(
        size=pt_block.dimension_decrease_layer.bias.shape)
    convolutional_layer_weight = random_number_generator.normal(
        size=pt_block.convolutional_layer.weight.shape)
    convolutional_layer_bias = random_number_generator.normal(
        size=pt_block.convolutional_layer.bias.shape)
    dimension_increase_layer_weight = random_number_generator.normal(
        size=pt_block.dimension_increase_layer.weight.shape)
    dimension_increase_layer_bias = random_number_generator.normal(
        size=pt_block.dimension_increase_layer.bias.shape)

    pt_block.dimension_decrease_layer.weight.data = Tensor(dimension_decrease_layer_weight)
    pt_block.dimension_decrease_layer.bias.data = Tensor(dimension_decrease_layer_bias)
    pt_block.convolutional_layer.weight.data = Tensor(convolutional_layer_weight)
    pt_block.convolutional_layer.bias.data = Tensor(convolutional_layer_bias)
    pt_block.dimension_increase_layer.weight.data = Tensor(dimension_increase_layer_weight)
    pt_block.dimension_increase_layer.bias.data = Tensor(dimension_increase_layer_bias)

    fake_tf_input = np.zeros(shape=[batch_size, length, input_channels])
    _ = tf_block.call(fake_tf_input).numpy()

    tf_block.dimension_decrease_layer.kernel.assign(dimension_decrease_layer_weight.transpose([2, 1, 0]))
    tf_block.dimension_decrease_layer.bias.assign(dimension_decrease_layer_bias)
    tf_block.convolutional_layer.kernel.assign(convolutional_layer_weight.transpose([2, 1, 0]))
    tf_block.convolutional_layer.bias.assign(convolutional_layer_bias)
    tf_block.dimension_increase_layer.kernel.assign(dimension_increase_layer_weight.transpose([2, 1, 0]))
    tf_block.dimension_increase_layer.bias.assign(dimension_increase_layer_bias)

    pt_input = random_number_generator.normal(size=[batch_size, input_channels, length])
    tf_input = pt_input.transpose([0, 2, 1])

    pt_output = pt_block(torch.Tensor(pt_input)).detach().numpy()
    tf_output = tf_block.call(tf_input).numpy()
    tf_output_transposed = tf_output.transpose([0, 2, 1])
    assert np.allclose(pt_output, tf_output_transposed, rtol=1.e-4, atol=1.e-7)
