import torch

from haplo.internal.transforms.affine_normalize import default_reverse_output_affine_transform, \
    default_reverse_input_affine_transform, default_input_affine_transform, default_output_affine_transform


def test_precomputed_normalize_phase_amplitudes_values():
    x = torch.ones([1, 64], dtype=torch.float32)
    expected_y = torch.full(size=[1, 64], fill_value=-0.71331304)
    y = default_reverse_output_affine_transform(x)
    assert torch.allclose(y, expected_y)


def test_precomputed_unnormalize_phase_amplitudes_values():
    x = torch.ones([1, 64], dtype=torch.float32)
    expected_y = torch.full(size=[1, 64], fill_value=81723.75)
    y = default_output_affine_transform(x)
    assert torch.allclose(y, expected_y)


def test_precomputed_normalize_parameters_values():
    x = torch.ones([1, 11], dtype=torch.float32)
    expected_y = torch.tensor([[3.5573754, 3.5618417, 3.553724, -0.6303051, -1.1805683, 3.556168, 3.5511322, 3.557625,
                                -0.63149905, -1.1832114, -1.5620476]])
    y = default_input_affine_transform(x)
    assert torch.allclose(y, expected_y)


def test_precomputed_unnormalize_parameters_values():
    x = torch.ones([1, 11], dtype=torch.float32)
    expected_y = torch.tensor(
        [[0.2805303, 0.28011018, 0.28137863, 2.478689, 4.950499, 0.28018776, 0.28149468, 0.28105912,
          2.4785657, 4.960278, 8.396504]])
    y = default_reverse_input_affine_transform(x)
    assert torch.allclose(y, expected_y)
