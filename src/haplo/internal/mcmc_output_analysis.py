from __future__ import annotations

import pandas as pd
import xarray
from numpy.random import RandomState
from pathlib import Path

from haplo.internal.constantinos_kalapotharakos_format import \
    combine_constantinos_kalapotharakos_split_mcmc_output_files_to_xarray_zarr


def slice_iteration_of_mcmc_output_xarray_dataset(dataset: xarray.Dataset, start_iteration: int, end_iteration
                                                  ) -> xarray.Dataset:
    """
    Gets a slice of an MCMC output Xarray dataset along the iteration axis.

    :param dataset: The MCMC output Xarray dataset.
    :param start_iteration: The start of the slice (inclusive).
    :param end_iteration: The end of the slice (exclusive).
    :return: The Xarray dataset that is the slice of the original dataset.
    """
    sliced_dataset = dataset.sel({'iteration': slice(start_iteration, end_iteration - 1)})
    return sliced_dataset


def mcmc_output_xarray_dataset_to_pandas_data_frame(
        dataset: xarray.Dataset,
        limit_from_end: int | None = None,
        random_sample_size: int | None = None,
) -> pd.DataFrame:
    """
    Converts the MCMC output Xarray dataset to a Pandas data frame.

    :param dataset: The MCMC output Xarray dataset.
    :return: The Pandas data frame.
    """
    state_parameter_data_array = dataset['parameter'].stack({'state_index': ['iteration', 'cpu', 'chain']}).transpose()
    state_log_likelihood_data_array = dataset['log_likelihood'].stack({'state_index': ['iteration', 'cpu', 'chain']}
                                                                      ).transpose()
    if limit_from_end is not None:
        state_parameter_data_array = state_parameter_data_array.isel(
            {'state_index': slice(-limit_from_end, None)})
        state_log_likelihood_data_array = state_log_likelihood_data_array.isel(
            {'state_index': slice(-limit_from_end, None)})
    if random_sample_size is not None:
        random_state = RandomState(0)
        sample_indexes = random_state.choice(state_parameter_data_array['state_index'].size, random_sample_size,
                                             replace=False)
        state_parameter_data_array = state_parameter_data_array.isel({'state_index': sample_indexes})
        state_log_likelihood_data_array = state_log_likelihood_data_array.isel({'state_index': sample_indexes})
    parameter_data_frame = state_parameter_data_array.to_pandas()
    parameter_data_frame.rename(columns={parameter_index: f'parameter{parameter_index}'
                                         for parameter_index in parameter_data_frame.columns}, inplace=True)
    log_likelihood_data_frame = state_log_likelihood_data_array.to_pandas()
    data_frame = pd.concat([parameter_data_frame, log_likelihood_data_frame], axis=1)
    if random_sample_size is not None:
        data_frame.sort_index(inplace=True)
    return data_frame
