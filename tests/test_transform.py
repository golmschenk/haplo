import numpy as np
import pytest

from haplo.nicer_transform import PrecomputedNormalizePhaseAmplitudes, PrecomputedUnnormalizePhaseAmplitudes, \
    PrecomputedUnnormalizeParameters, PrecomputedNormalizeParameters


def test_precomputed_normalize_phase_amplitudes_values():
    x = np.ones([1, 64], dtype=np.float32)
    expected_y = np.full(shape=[1, 64], fill_value=-0.71331304)
    transform = PrecomputedNormalizePhaseAmplitudes()
    y = transform(x)
    assert np.allclose(y, expected_y)


@pytest.mark.skip(reason='Need to refactor to make this test work.')
def test_precomputed_normalize_phase_amplitudes_does_not_mutate_original_array():
    x = np.ones([1, 64], dtype=np.float32)
    transform = PrecomputedNormalizePhaseAmplitudes()
    _ = transform(x)
    assert np.allclose(x, np.ones([1, 64], dtype=np.float32))


def test_precomputed_unnormalize_phase_amplitudes_values():
    x = np.ones([1, 64], dtype=np.float32)
    expected_y = np.full(shape=[1, 64], fill_value=81723.75)
    transform = PrecomputedUnnormalizePhaseAmplitudes()
    y = transform(x)
    assert np.allclose(y, expected_y)


@pytest.mark.skip(reason='Need to refactor to make this test work.')
def test_precomputed_unnormalize_phase_amplitudes_does_not_mutate_original_array():
    x = np.ones([1, 64], dtype=np.float32)
    transform = PrecomputedUnnormalizePhaseAmplitudes()
    _ = transform(x)
    assert np.allclose(x, np.ones([1, 64], dtype=np.float32))


def test_precomputed_normalize_parameters_values():
    x = np.ones([1, 11], dtype=np.float32)
    expected_y = np.array([[3.5573754, 3.5618417, 3.553724, -0.6303051, -1.1805683, 3.556168, 3.5511322, 3.557625,
                            -0.63149905, -1.1832114, -1.5620476]])
    transform = PrecomputedNormalizeParameters()
    y = transform(x)
    assert np.allclose(y, expected_y)


@pytest.mark.skip(reason='Need to refactor to make this test work.')
def test_precomputed_normalize_parameters_does_not_mutate_original_array():
    x = np.ones([1, 11], dtype=np.float32)
    transform = PrecomputedNormalizeParameters()
    _ = transform(x)
    assert np.allclose(x, np.ones([1, 11], dtype=np.float32))


def test_precomputed_unnormalize_parameters_values():
    x = np.ones([1, 11], dtype=np.float32)
    expected_y = np.array([[0.2805303, 0.28011018, 0.28137863, 2.478689, 4.950499, 0.28018776, 0.28149468, 0.28105912,
                            2.4785657, 4.960278, 8.396504]])
    transform = PrecomputedUnnormalizeParameters()
    y = transform(x)
    assert np.allclose(y, expected_y)


@pytest.mark.skip(reason='Need to refactor to make this test work.')
def test_precomputed_unnormalize_parameters_does_not_mutate_original_array():
    x = np.ones([1, 11], dtype=np.float32)
    transform = PrecomputedUnnormalizeParameters()
    _ = transform(x)
    assert np.allclose(x, np.ones([1, 11], dtype=np.float32))
