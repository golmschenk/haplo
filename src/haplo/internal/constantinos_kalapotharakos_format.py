from __future__ import annotations

import itertools

import dask.array
import logging
import mmap
import numpy as np
import re
import shutil
import xarray
from pathlib import Path
from typing import Iterator, TextIO

logger = logging.getLogger(__name__)


class ConstantinosKalapotharakosFormatError(Exception):
    pass


def constantinos_kalapotharakos_format_record_generator(path: Path, elements_per_record: int
                                                        ) -> Iterator[tuple[float, ...]]:
    """
    Create a record generator for a Constantinos Kalapotharakos format file.

    :param path: The path to the file.
    :param elements_per_record: The number of elements per record.
    :return: A generator that iterates over the records.
    """
    with path.open() as file_handle:
        file_contents = get_memory_mapped_file_contents(file_handle)
        generator = constantinos_kalapotharakos_format_record_generator_from_file_contents(file_contents=file_contents,
                                                                                           elements_per_record=elements_per_record)
        for record in generator:
            yield record


def constantinos_kalapotharakos_format_record_generator_from_file_contents(
        file_contents: bytes | mmap.mmap, elements_per_record: int) -> Iterator[tuple[float, ...]]:
    """
    Create a record generator for a Constantinos Kalapotharakos format file's contents.

    :param file_contents: The file contents object.
    :param elements_per_record: The number of elements per record.
    :return: A generator that iterates over the records.
    """
    value_iterator = re.finditer(rb"[^\s]+", file_contents)
    count = 0
    while True:
        values = []
        try:
            values.append(float(next(value_iterator).group(0)))
        except StopIteration:
            break
        try:
            for _ in range(elements_per_record - 1):
                values.append(float(next(value_iterator).group(0)))
            yield tuple(values)
            if count % 100000 == 0:
                logger.info(f'Processed {count} rows.')
            count += 1
        except StopIteration:
            raise ConstantinosKalapotharakosFormatError(
                f'The Constantinos Kalapotharakos format file ran out of elements when trying to get '
                f'{elements_per_record} elements for the current record.')


def get_memory_mapped_file_contents(file_handle: TextIO) -> mmap.mmap:
    """
    Get a memory mapped version of a file handle's contents.

    :param file_handle: The file handle to memory map.
    :return: The memory map.
    """
    file_fileno = file_handle.fileno()
    file_contents = mmap.mmap(file_fileno, 0, access=mmap.ACCESS_READ)
    return file_contents


def combine_constantinos_kalapotharakos_split_mcmc_output_files_to_xarray_zarr(
        split_mcmc_output_directory: Path, combined_output_path: Path, elements_per_record: int, overwrite: bool = False
) -> None:
    """
    Combine Constantinos Kalapotharakos format split mcmc output files into an Xarray Zarr data store.

    :param split_mcmc_output_directory: The root of the split files.
    :param combined_output_path: The path of the output Zarr file.
    :param elements_per_record: The number of elements per record in the split files. Similar to columns per row, but
                                the files are not organized into rows and columns.
    :param overwrite: Overwrite existing files if they exist. Otherwise, an error will be raised if they exist.
    :return: None
    """

    def _save_batch_to_cpu_and_iteration_region(zarr_path, parameters_batch_, log_likelihood_batch_,
                                                region_start_iteration, region_end_iteration, cpu, chains_,
                                                parameter_count_, parameter_indexes_):
        flat_parameters_batch_array = np.array(parameters_batch_, dtype=np.float32)
        parameters_batch_array = flat_parameters_batch_array.reshape(
            [region_end_iteration - region_start_iteration, 1, chains_.size, parameter_count_])
        flat_log_likelihood_batch_array = np.array(log_likelihood_batch_, dtype=np.float32)
        log_likelihood_batch_array = flat_log_likelihood_batch_array.reshape(
            [region_end_iteration - region_start_iteration, 1, chains_.size])
        region_dataset = xarray.Dataset(
            coords={
                'iteration': np.arange(region_start_iteration, region_end_iteration, dtype=np.int64),
                'cpu': np.array([cpu], dtype=np.int64),
                'chain': chains_,
                'parameter_index': parameter_indexes_,
            },
            data_vars={
                'parameter': (
                    ['iteration', 'cpu', 'chain', 'parameter_index'],
                    parameters_batch_array,
                ),
                'log_likelihood': (
                    ['iteration', 'cpu', 'chain'],
                    log_likelihood_batch_array,
                ),
            },
        )
        region_dataset.to_zarr(zarr_path, region='auto')

    def _save_final_iteration_region(zarr_path, parameters_batch_, log_likelihood_batch_,
                                     iteration_, cpus_, chains_, parameter_count_, parameter_indexes_):
        flat_parameters_batch_array = np.array(parameters_batch_, dtype=np.float32)
        parameters_batch_array = flat_parameters_batch_array.reshape(
            [1, cpus_.size, chains_.size, parameter_count_])
        flat_log_likelihood_batch_array = np.array(log_likelihood_batch_, dtype=np.float32)
        log_likelihood_batch_array = flat_log_likelihood_batch_array.reshape(
            [1, cpus_.size, chains_.size])
        region_dataset = xarray.Dataset(
            coords={
                'iteration': np.array([iteration_], dtype=np.int64),
                'cpu': cpus_,
                'chain': chains_,
                'parameter_index': parameter_indexes_,
            },
            data_vars={
                'parameter': (
                    ['iteration', 'cpu', 'chain', 'parameter_index'],
                    parameters_batch_array,
                ),
                'log_likelihood': (
                    ['iteration', 'cpu', 'chain'],
                    log_likelihood_batch_array,
                ),
            },
        )
        region_dataset.to_zarr(zarr_path, append_dim='iteration')

    def _get_known_complete_iterations(split_data_file_paths, elements_per_record):
        split_data_path0 = split_data_file_paths[0]
        record_generator_ = constantinos_kalapotharakos_format_record_generator(
            split_data_path0, elements_per_record=elements_per_record)
        data_file0_iterations = 0
        is_final_iteration_known_incomplete_ = False
        read_chain0 = False
        for _ in record_generator_:
            if read_chain0:
                data_file0_iterations += 1
                read_chain0 = False
            else:
                read_chain0 = True
        if read_chain0:
            is_final_iteration_known_incomplete_ = True
        if is_final_iteration_known_incomplete_:
            max_known_complete_iteration_ = data_file0_iterations - 1
        else:
            max_known_complete_iteration_ = data_file0_iterations - 2
        return max_known_complete_iteration_, is_final_iteration_known_incomplete_

    def _create_empty_dataset_zarr(zarr_path, iterations, cpus, chains, parameter_indexes, iteration_chunk_size):
        empty_dataset = xarray.Dataset(
            coords={
                'iteration': iterations,
                'cpu': cpus,
                'chain': chains,
                'parameter_index': parameter_indexes,
            },
            data_vars={
                'parameter': (
                    ['iteration', 'cpu', 'chain', 'parameter_index'],
                    dask.array.full((iterations.size, cpus.size, chains.size, parameter_indexes.size),
                                    fill_value=np.nan, chunks=(iteration_chunk_size, -1, -1, -1), dtype=np.float32),
                ),
                'log_likelihood': (
                    ['iteration', 'cpu', 'chain'],
                    dask.array.full((iterations.size, cpus.size, chains.size),
                                    fill_value=np.nan, chunks=(iteration_chunk_size, -1, -1), dtype=np.float32),
                ),
            },
        )
        encoding = {
            'iteration': {'dtype': 'int64', 'chunks': (iteration_chunk_size,)},
            'cpu': {'dtype': 'int64', 'chunks': (-1,)},
            'chain': {'dtype': 'int64', 'chunks': (-1,)},
            'parameter_index': {'dtype': 'int64', 'chunks': (-1,)},
            'parameter': {'dtype': 'float32', 'chunks': (iteration_chunk_size, -1, -1, -1)},
            'log_likelihood': {'dtype': 'float32', 'chunks': (iteration_chunk_size, -1, -1)},
        }
        empty_dataset.to_zarr(zarr_path, compute=False, encoding=encoding)

    def _check_for_existing_files(combined_output_path_, overwrite_):
        temporary_combined_output_path_ = combined_output_path_.parent.joinpath(combined_output_path_.name + '.partial')
        if temporary_combined_output_path_.exists():
            if overwrite_:
                shutil.rmtree(temporary_combined_output_path_)
            else:
                raise FileExistsError(f'{temporary_combined_output_path_} needs to be created, but already exists. '
                                      f'Pass `overwrite=True` to overwrite.')
        if combined_output_path_.exists():
            if overwrite_:
                shutil.rmtree(combined_output_path_)
            else:
                raise FileExistsError(f'{combined_output_path_} needs to be created, but already exists. '
                                      f'Pass `overwrite=True` to overwrite.')
        return temporary_combined_output_path_

    temporary_combined_output_path = _check_for_existing_files(combined_output_path, overwrite)
    split_data_file_paths = sorted(split_mcmc_output_directory.glob('*.dat'))
    max_known_complete_iteration, is_final_iteration_known_incomplete = _get_known_complete_iterations(
        split_data_file_paths, elements_per_record)
    iterations = np.arange(max_known_complete_iteration + 1, dtype=np.int64)
    cpus = np.arange(len(split_data_file_paths), dtype=np.int64)
    chains = np.array([0, 1], dtype=np.int64)
    parameter_count = elements_per_record - 2
    parameter_indexes = np.arange(parameter_count, dtype=np.int64)
    iteration_chunk_size = 100
    _create_empty_dataset_zarr(temporary_combined_output_path, iterations, cpus, chains, parameter_indexes,
                               iteration_chunk_size)

    final_iteration_parameters_batch: list[tuple[float, ...]] = []
    final_iteration_log_likelihood_batch: list[float] = []
    for split_index, split_data_path in enumerate(split_data_file_paths):
        logger.info(f'Processing {split_data_path}.')
        record_generator = constantinos_kalapotharakos_format_record_generator(
            split_data_path, elements_per_record=elements_per_record)
        parameters_batch: list[tuple[float, ...]] = []
        log_likelihood_batch: list[float] = []
        split_data_frame_cpu_number = int(re.search(r'1(\d+)\.dat', split_data_path.name).group(1))
        if split_index != split_data_frame_cpu_number:
            raise ValueError(f'A split MCMC output file was expected but not found. '
                             f'Expected a file for CPU number {split_index}, but found {split_data_path.name}.')
        chain = 0
        iteration = 0
        batch_start_iteration = iteration
        for record_index in itertools.count():
            record = next(record_generator)
            if chain != int(record[elements_per_record - 1]):
                raise ValueError(f'The chain did not match the expected value at record index {record_index}.')
            parameters_batch.append(record[:parameter_count])
            log_likelihood_batch.append(record[parameter_count])
            if chain == 1:
                chain = 0
                iteration += 1
                if iteration > batch_start_iteration + iteration_chunk_size or iteration > max_known_complete_iteration:
                    _save_batch_to_cpu_and_iteration_region(temporary_combined_output_path, parameters_batch,
                                                            log_likelihood_batch, batch_start_iteration, iteration,
                                                            split_data_frame_cpu_number, chains, parameter_count,
                                                            parameter_indexes)
                    batch_start_iteration = iteration
                    parameters_batch = []
                    log_likelihood_batch = []
                if iteration > max_known_complete_iteration:
                    if not is_final_iteration_known_incomplete:
                        try:
                            record = next(record_generator)
                            final_iteration_parameters_batch.append(record[:parameter_count])
                            final_iteration_log_likelihood_batch.append(record[parameter_count])
                            record = next(record_generator)
                            final_iteration_parameters_batch.append(record[:parameter_count])
                            final_iteration_log_likelihood_batch.append(record[parameter_count])
                        except StopIteration:
                            is_final_iteration_known_incomplete = True
                    break
            else:
                chain = 1
    if not is_final_iteration_known_incomplete:
        _save_final_iteration_region(temporary_combined_output_path, final_iteration_parameters_batch,
                                     final_iteration_log_likelihood_batch,
                                     max_known_complete_iteration + 1, cpus, chains,
                                     parameter_count, parameter_indexes)
    temporary_combined_output_path.rename(combined_output_path)
