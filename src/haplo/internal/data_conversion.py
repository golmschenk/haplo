from __future__ import annotations

import shutil

import itertools
import logging
import math
import mmap
import re
from pandas import DataFrame
from pathlib import Path

import numpy as np
import pandas as pd
import xarray
import zarr
from xarray import Dataset

from haplo.data_preparation import get_memory_mapped_file_contents, \
    arbitrary_constantinos_kalapotharakos_file_path_to_pandas
from haplo.logging import set_up_default_logger

logger = logging.getLogger(__name__)


def constantinos_kalapotharakos_format_file_to_zarr(input_file_path: Path, output_file_path: Path,
                                                    parameter_count: int = 11) -> None:
    """
    Converts a Constantinos Kalapotharakos format text file to a Zarr data store.

    :param input_file_path: The path to the text file.
    :param output_file_path: The path of the data store.
    :param parameter_count: The number of parameters per row.
    :return: None
    """
    with input_file_path.open() as file_handle:
        file_contents = get_memory_mapped_file_contents(file_handle)
        constantinos_kalapotharakos_file_handle_to_1d_input_1d_output_zarr(file_contents, output_file_path,
                                                                           parameter_count)


def constantinos_kalapotharakos_file_handle_to_1d_input_1d_output_zarr(file_contents: bytes | mmap.mmap,
                                                                       output_file_path: Path,
                                                                       input_size: int = 11,
                                                                       output_size: int = 64,
                                                                       zarr_chunk_axis0_size: int = 1000) -> None:
    """
    Converts the file contents of a Constantinos Kalapotharakos format text file to a Zarr data store.

    :param file_contents: Data to parse, provided in binary form as either bytes or mmap.
    :param output_file_path: Path where the resulting Zarr dataset is stored.
    :param input_size: Number of elements expected in the input array per data row.
    :param output_size: Number of elements expected in the output array per data row.
    :param zarr_chunk_axis0_size: Number of rows to store per Zarr chunk (axis 0).
    :return: None
    """
    set_up_default_logger()
    zarr_store = zarr.DirectoryStore(str(output_file_path))
    root = zarr.group(store=zarr_store, overwrite=True)
    input_array = root.create_dataset(
        'input',
        shape=(0, input_size),
        chunks=(zarr_chunk_axis0_size, input_size),
        dtype='float32',
    )
    output_array = root.create_dataset(
        'output',
        shape=(0, output_size),
        chunks=(zarr_chunk_axis0_size, output_size),
        dtype='float32',
    )

    value_iterator = re.finditer(rb"[^\s]+", file_contents)
    parameters_set = []
    phase_amplitudes_set = []
    for index in itertools.count():
        parameters = []
        try:
            parameters.append(float(next(value_iterator).group(0)))
        except StopIteration:
            break
        for _ in range(input_size - 1):
            parameters.append(float(next(value_iterator).group(0)))
        _ = float(next(value_iterator).group(0))  # Likelihood in Constantinos' output which has no meaning here.
        phase_amplitudes = []
        for _ in range(output_size):
            phase_amplitudes.append(float(next(value_iterator).group(0)))
        parameters_set.append(parameters)
        phase_amplitudes_set.append(phase_amplitudes)
        if (index + 1) % 100000 == 0:
            input_array.append(parameters_set)
            output_array.append(phase_amplitudes_set)
            logger.info(f'Processed {index + 1} rows.')
            parameters_set = []
            output_array = []
    input_array.append(parameters_set)
    output_array.append(phase_amplitudes_set)


def persist_xarray_dataset_variable_order(dataset: Dataset) -> Dataset:
    return dataset.assign_attrs(variable_order=[variable_name for variable_name in dataset.variables
                                                if variable_name not in dataset.dims])


def to_ordered_dataframe(dataset: Dataset) -> DataFrame:
    variable_order = dataset.attrs['variable_order']
    data_frame = dataset.to_dataframe()
    data_frame = data_frame[variable_order]
    return data_frame


def combine_constantinos_kalapotharakos_split_mcmc_output_files_to_xarray_zarr(
        root_directory_path: Path, combined_output_path: Path, columns_per_row: int) -> None:
    """
    Combine Constantinos Kalapotharakos format split mcmc output files into an Xarray Zarr data store.

    :param root_directory_path: The root of the split files.
    :param combined_output_path: The path of the output Zarr file.
    :param columns_per_row: The number of columns per row in the split files.
    :return: None
    """
    if combined_output_path.exists():
        shutil.rmtree(combined_output_path)
    set_up_default_logger()
    for split_index, split_data_path in enumerate(sorted(root_directory_path.glob('*.dat'))):
        logger.info(f'Processing {split_data_path}.')
        split_data_frame = arbitrary_constantinos_kalapotharakos_file_path_to_pandas(split_data_path,
                                                                                     columns_per_row=columns_per_row)
        rename_dictionary: dict[str, str] = {}
        for column_index in range(columns_per_row - 2):
            rename_dictionary[str(column_index)] = f'parameter{column_index}'
        rename_dictionary[str(columns_per_row - 2)] = f'log_likelihood'
        rename_dictionary[str(columns_per_row - 1)] = f'chain'
        split_data_frame = split_data_frame.rename(rename_dictionary, axis='columns')
        split_data_frame['chain'] = split_data_frame['chain'].astype(np.int64)
        split_data_frame_cpu_number = int(re.search('1(\d+)\.dat', split_data_path.name).group(1))
        split_data_frame['cpu'] = split_data_frame_cpu_number
        iterations = math.ceil(split_data_frame.shape[0] / 2)
        iteration_array = np.arange(iterations, dtype=np.int64)
        combined_iteration_array = np.empty((iteration_array.size * 2), dtype=iteration_array.dtype)
        combined_iteration_array[0::2] = iteration_array
        combined_iteration_array[1::2] = iteration_array
        if split_data_frame.shape[0] % 2 != 0:
            combined_iteration_array = combined_iteration_array[:-1]
        split_data_frame['iteration'] = combined_iteration_array
        split_dataset: xarray.Dataset = split_data_frame.to_xarray()
        if split_index == 0:
            split_dataset = persist_xarray_dataset_variable_order(split_dataset)
            split_dataset.to_zarr(combined_output_path)
        else:
            split_dataset = persist_xarray_dataset_variable_order(split_dataset)
            split_dataset.to_zarr(combined_output_path, append_dim='index')


def convert_from_2d_xarray_zarr_to_csv(xarray_zarr_path: Path, csv_path: Path) -> None:
    """
    Convert a table-like 2D Xarray Zarr data store to a CSV file.

    :param xarray_zarr_path: The path to the Zarr data store.
    :param csv_path: The path to the CSV file.
    :return: None
    """
    dataset = xarray.open_zarr(xarray_zarr_path)
    data_frame: pd.DataFrame = to_ordered_dataframe(dataset)
    data_frame.to_csv(csv_path, index=False)
