import numpy as np
import torch

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
