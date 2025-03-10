from __future__ import annotations

from enum import StrEnum
import itertools
import logging
import math
import mmap
import numpy as np
import pandas as pd
import re
import shutil
import xarray
import zarr
from pandas import DataFrame
from pathlib import Path
from xarray import Dataset

from haplo.data_preparation import get_memory_mapped_file_contents, \
    arbitrary_constantinos_kalapotharakos_file_path_to_pandas
from haplo.logging import set_up_default_logger

logger = logging.getLogger(__name__)


class DatasetVariableName(StrEnum):
    INPUT = 'input'
    OUTPUT = 'output'


def constantinos_kalapotharakos_format_file_to_xarray_zarr(
        input_path: Path,
        output_path: Path,
        input_size: int = 11,
        output_size: int = 64,
        zarr_chunk_axis0_size: int = 1000,
) -> None:
    set_up_default_logger()
    if output_path.exists():
        shutil.rmtree(output_path)
    with input_path.open() as file_handle:
        file_contents = get_memory_mapped_file_contents(file_handle)
        value_iterator = re.finditer(rb"[^\s]+", file_contents)
        parameters_set = []
        phase_amplitudes_set = []
        encoding = {
            DatasetVariableName.INPUT: {'dtype': 'float32', 'chunks': (zarr_chunk_axis0_size, input_size)},
            DatasetVariableName.OUTPUT: {'dtype': 'float32', 'chunks': (zarr_chunk_axis0_size, output_size)},
        }
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
                partial_dataset = xarray.Dataset(data_vars={
                    DatasetVariableName.INPUT: (['index', 'parameter_index'], parameters_set),
                    DatasetVariableName.OUTPUT: (['index', 'phase_index'], phase_amplitudes_set),
                })
                if not output_path.exists():
                    partial_dataset.to_zarr(output_path, encoding=encoding)
                else:
                    partial_dataset.to_zarr(output_path, append_dim='index')
                logger.info(f'Processed {index + 1} rows.')
                parameters_set = []
                phase_amplitudes_set = []
        if len(parameters_set) != 0:
            partial_dataset = xarray.Dataset(data_vars={
                DatasetVariableName.INPUT: (['index', 'parameter_index'], parameters_set),
                DatasetVariableName.OUTPUT: (['index', 'phase_index'], phase_amplitudes_set),
            })
            if not output_path.exists():
                partial_dataset.to_zarr(output_path, encoding=encoding)
            else:
                partial_dataset.to_zarr(output_path, append_dim='index')


def convert_directory_xarray_zarr_to_zip_xarray_zarr(
        input_path: Path,
        output_path: Path,
) -> None:
    if output_path.suffix != '.zip':
        raise ValueError(f'Expected a .zip extension for the output file {output_path}')
    dataset = xarray.open_zarr(input_path)
    dataset.to_zarr(output_path, mode='w')


def constantinos_kalapotharakos_format_file_to_xarray_zarr_zip(
        input_path: Path,
        output_path: Path,
        input_size: int = 11,
        output_size: int = 64,
        zarr_chunk_axis0_size: int = 1000,
) -> None:
    if output_path.suffix != '.zip':
        raise ValueError(f'Expected a .zip extension for the output file {output_path}')
    temporary_intermediate_unzipped_zarr_path = output_path.parent.joinpath(output_path.stem + '.zarr')
    if temporary_intermediate_unzipped_zarr_path.exists():
        raise ValueError(f'Tried to use temporary file {temporary_intermediate_unzipped_zarr_path}, but it already '
                         f'exists.')
    constantinos_kalapotharakos_format_file_to_xarray_zarr(
        input_path=input_path,
        output_path=temporary_intermediate_unzipped_zarr_path,
        input_size=input_size,
        output_size=output_size,
        zarr_chunk_axis0_size=zarr_chunk_axis0_size,
    )
    convert_directory_xarray_zarr_to_zip_xarray_zarr(
        input_path=temporary_intermediate_unzipped_zarr_path, output_path=output_path)
    shutil.rmtree(temporary_intermediate_unzipped_zarr_path)


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
        DatasetVariableName.INPUT,
        shape=(0, input_size),
        chunks=(zarr_chunk_axis0_size, input_size),
        dtype='float32',
    )
    output_array = root.create_dataset(
        DatasetVariableName.OUTPUT,
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
            phase_amplitudes_set = []
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
