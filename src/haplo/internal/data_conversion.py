from __future__ import annotations

import itertools
import logging
import mmap
import re
from pathlib import Path

import zarr

from haplo.data_preparation import get_memory_mapped_file_contents
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
            logger.warning(f'Processed {index + 1} rows.')
            parameters_set = []
            output_array = []
    input_array.append(parameters_set)
    output_array.append(phase_amplitudes_set)
