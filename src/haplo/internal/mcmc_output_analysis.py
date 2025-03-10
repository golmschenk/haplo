from pathlib import Path

import pandas as pd
import xarray

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


def mcmc_output_xarray_dataset_to_pandas_data_frame(dataset: xarray.Dataset) -> pd.DataFrame:
    """
    Converts the MCMC output Xarray dataset to a Pandas data frame.

    :param dataset: The MCMC output Xarray dataset.
    :return: The Pandas data frame.
    """
    parameter_data_frame = dataset['parameter'].stack({'state_index': ['iteration', 'cpu', 'chain']}
                                                      ).transpose().to_pandas()
    parameter_data_frame.rename(columns={parameter_index: f'parameter{parameter_index}'
                                         for parameter_index in parameter_data_frame.columns}, inplace=True)
    log_likelihood_data_frame = dataset['log_likelihood'].stack({'state_index': ['iteration', 'cpu', 'chain']}
                                                                ).transpose().to_pandas()
    data_frame = pd.concat([parameter_data_frame, log_likelihood_data_frame], axis=1)
    return data_frame


if __name__ == '__main__':
    dataset_path_ = Path('temp.zarr')
    combine_constantinos_kalapotharakos_split_mcmc_output_files_to_xarray_zarr(
        Path('tests/end_to_end_tests/'
             'combine_constantinos_kalapotharakos_split_mcmc_output_resources_with_complete_final_iteration'),
        dataset_path_,
        elements_per_record=13,
        overwrite=True
    )
    dataset_ = xarray.load_dataset(dataset_path_)
    data_frame_ = mcmc_output_xarray_dataset_to_pandas_data_frame(dataset_)
