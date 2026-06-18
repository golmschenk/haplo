from __future__ import annotations

import itertools
import logging
import re
import shutil
from enum import StrEnum
from pathlib import Path

import xarray
from zarr.storage import ZipStore

from haplo.internal.constantinos_kalapotharakos_format import get_memory_mapped_file_contents
from haplo.internal.logging import set_up_default_logger

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
        value_iterator = re.finditer(rb'[^\s]+', file_contents)
        input_set = []
        output_set = []
        encoding = {
            DatasetVariableName.INPUT: {'dtype': 'float32', 'chunks': (zarr_chunk_axis0_size, input_size)},
            DatasetVariableName.OUTPUT: {'dtype': 'float32', 'chunks': (zarr_chunk_axis0_size, output_size)},
        }
        for index in itertools.count():
            inputs = []
            try:
                inputs.append(float(next(value_iterator).group(0)))
            except StopIteration:
                break
            for _ in range(input_size - 1):
                inputs.append(float(next(value_iterator).group(0)))
            _ = float(next(value_iterator).group(0))  # Likelihood in Constantinos' output which has no meaning here.
            outputs = []
            for _ in range(output_size):
                outputs.append(float(next(value_iterator).group(0)))
            input_set.append(inputs)
            output_set.append(outputs)
            if (index + 1) % 100000 == 0:
                partial_dataset = xarray.Dataset(data_vars={
                    DatasetVariableName.INPUT: (['index', 'input'], input_set),
                    DatasetVariableName.OUTPUT: (['index', 'output'], output_set),
                })
                if not output_path.exists():
                    partial_dataset.to_zarr(output_path, encoding=encoding)
                else:
                    partial_dataset.to_zarr(output_path, append_dim='index')
                logger.info(f'Processed {index + 1} rows.')
                input_set = []
                output_set = []
        if len(input_set) != 0:
            partial_dataset = xarray.Dataset(data_vars={
                DatasetVariableName.INPUT: (['index', 'input'], input_set),
                DatasetVariableName.OUTPUT: (['index', 'output'], output_set),
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
    output_store = ZipStore(output_path, mode='w')
    dataset.to_zarr(output_store, mode='w')


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
