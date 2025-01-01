import numpy as np
import pandas as pd
import pytest

from haplo.internal.mcmc_output_file_manipulations import get_last_states_of_mcmc_states_data_frame


mcmc_states_data_frame = pd.DataFrame(
    {
        'parameter': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'iteration': [0, 0, 0, 1, 1, 1, 2, 2, 2]
    }
)


def test_get_last_states_of_mcmc_data_frame():
    latest_mcmc_states_data_frame = get_last_states_of_mcmc_states_data_frame(mcmc_states_data_frame,
                                                                              number_of_states=5)
    expected_parameters = [4, 5, 6, 7, 8]
    assert np.allclose(np.sort(expected_parameters), np.sort(latest_mcmc_states_data_frame['parameter']))


def test_get_last_states_of_mcmc_data_frame_errors_when_requesting_larger_than_total():
    with pytest.raises(ValueError):
        _ = get_last_states_of_mcmc_states_data_frame(mcmc_states_data_frame, number_of_states=10)


def test_get_last_states_of_mcmc_data_frame_when_requested_is_smaller_than_last_iteration():
    latest_mcmc_states_data_frame = get_last_states_of_mcmc_states_data_frame(mcmc_states_data_frame,
                                                                              number_of_states=2)
    expected_parameters = [7, 8]
    assert np.allclose(np.sort(expected_parameters), np.sort(latest_mcmc_states_data_frame['parameter']))
