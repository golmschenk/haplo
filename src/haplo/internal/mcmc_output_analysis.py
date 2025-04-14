from __future__ import annotations

import numpy as np
import pandas as pd
import xarray
from numpy.random import default_rng


def slice_iteration_of_mcmc_output_xarray_dataset(
        dataset: xarray.Dataset,
        *,
        start_iteration: int,
        end_iteration: int,
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
    :param limit_from_end: Limits the number of rows from the end of the dataset.
    :param random_sample_size: Randomly samples this many elements from the dataset. If `limit_from_end` is set, that
                               limit will be applied first, then this many will be sampled from that subset.
    :return: The Pandas data frame.
    """
    state_size = dataset['log_likelihood'].size
    if limit_from_end is not None and random_sample_size is not None:
        random_generator = default_rng(seed=0)
        index_offsets = random_generator.choice(limit_from_end, random_sample_size, replace=False)
        state_indexes = np.flip((state_size - index_offsets) - 1)
    elif limit_from_end is not None:
        index_offsets = np.arange(limit_from_end)
        state_indexes = np.flip((state_size - index_offsets) - 1)
    elif random_sample_size is not None:
        random_generator = default_rng(seed=0)
        state_indexes = random_generator.choice(state_size, random_sample_size, replace=False)
    else:
        state_indexes = np.arange(state_size)

    multi_indexes = np.unravel_index(state_indexes,
                                     (dataset['iteration'].size, dataset['cpu'].size, dataset['chain'].size))
    sampled_dataset = dataset.isel({
        'iteration': xarray.DataArray(multi_indexes[0], dims='state_index'),
        'cpu': xarray.DataArray(multi_indexes[1], dims='state_index'),
        'chain': xarray.DataArray(multi_indexes[2], dims='state_index'),
    })

    parameter_data_frame = sampled_dataset['parameter'].to_pandas()
    parameter_data_frame.rename(columns={parameter_index: f'parameter{parameter_index}'
                                         for parameter_index in parameter_data_frame.columns}, inplace=True)
    log_likelihood_data_frame = sampled_dataset['log_likelihood'].to_pandas()
    data_frame = pd.concat([parameter_data_frame, log_likelihood_data_frame], axis=1)
    multi_index = pd.MultiIndex.from_arrays(
        (
            sampled_dataset['iteration'].to_numpy(),
            sampled_dataset['cpu'].to_numpy(),
            sampled_dataset['chain'].to_numpy()
        ),
        names=('iteration', 'cpu', 'chain'))
    data_frame.index = multi_index
    if random_sample_size is not None:
        data_frame.sort_index(inplace=True)
    return data_frame


def extract_windowed_median_log_likelihood_series(dataset: xarray.Dataset, window_size: int = 1000) -> pd.Series:
    """
    Extracts the windowed median log likelihood values from the dataset.

    :param dataset: The MCMC output dataset.
    :param window_size: The size of the window to take the medians over.
    :return: The resulting medians. The Pandas Series has an index that is the start of the window.
    """
    bin_edges = np.concatenate([
        np.arange(dataset['iteration'].min(), dataset['iteration'].max() + 1, window_size, dtype=np.int64),
        np.array([dataset['iteration'].max() + 1], dtype=np.int64)
    ])
    binned = dataset['log_likelihood'].groupby_bins('iteration', bins=bin_edges, right=False)
    medians = binned.quantile(q=0.5, dim=['iteration', 'cpu', 'chain']).compute().to_numpy()
    return pd.Series(index=bin_edges[:-1], data=medians)
