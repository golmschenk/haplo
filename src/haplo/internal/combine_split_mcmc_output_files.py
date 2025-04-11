import dask.array
import itertools
import logging
import numpy as np
import re
import shutil
import xarray
from multiprocessing.pool import AsyncResult, Pool
from pathlib import Path

from haplo.internal.constantinos_kalapotharakos_format import constantinos_kalapotharakos_format_record_generator

logger = logging.getLogger(__name__)


def combine_constantinos_kalapotharakos_split_mcmc_output_files_to_xarray_zarr(
        split_mcmc_output_directory: Path,
        combined_output_path: Path,
        *,
        elements_per_record: int,
        overwrite: bool = False,
        multiprocess_pool_size: int = 1,
) -> None:
    """
    Combine Constantinos Kalapotharakos format split mcmc output files into an Xarray Zarr data store.

    :param split_mcmc_output_directory: The root of the split files.
    :param combined_output_path: The path of the output Zarr file.
    :param elements_per_record: The number of elements per record in the split files. Similar to columns per row, but
                                the files are not organized into rows and columns.
    :param overwrite: Overwrite existing files if they exist. Otherwise, an error will be raised if they exist.
    :param multiprocess_pool_size: The number of processes to handle the conversion process.
    :return: None
    """
    temporary_combined_output_path0, temporary_combined_output_path1 = _check_for_existing_files(
        combined_output_path, overwrite)
    split_data_file_paths = sorted(split_mcmc_output_directory.glob('*.dat'))
    max_known_complete_iteration = _get_known_complete_iterations(split_data_file_paths, elements_per_record)
    iterations = np.arange(max_known_complete_iteration + 1, dtype=np.int64)
    cpus = np.arange(len(split_data_file_paths), dtype=np.int64)
    chains = np.array([0, 1], dtype=np.int64)
    parameter_count = elements_per_record - 2
    parameter_indexes = np.arange(parameter_count, dtype=np.int64)
    scanning_iteration_chunk_size = 1_000_000
    _create_empty_dataset_zarr(temporary_combined_output_path0, iterations, cpus, chains, parameter_indexes,
                               scanning_iteration_chunk_size)

    final_iteration_parameters_batch: list[tuple[float, ...]] = []
    final_iteration_log_likelihood_batch: list[float] = []
    split_is_final_iteration_known_incomplete_list: list[bool] = []
    with Pool(processes=multiprocess_pool_size) as pool:
        split_process_results: list[AsyncResult] = []
        for split_index, split_data_path in enumerate(split_data_file_paths):
            split_process_result = pool.apply_async(_process_split_file,
                                                    [
                                                        temporary_combined_output_path0, split_data_path, split_index,
                                                        elements_per_record, max_known_complete_iteration, chains,
                                                        parameter_count, parameter_indexes,
                                                        scanning_iteration_chunk_size
                                                    ])
            split_process_results.append(split_process_result)
        for split_process_result in split_process_results:
            (split_final_iteration_log_likelihood_batch, split_final_iteration_parameters_batch,
             split_is_final_iteration_known_incomplete) = split_process_result.get()
            final_iteration_parameters_batch.extend(split_final_iteration_parameters_batch)
            final_iteration_log_likelihood_batch.extend(split_final_iteration_log_likelihood_batch)
            split_is_final_iteration_known_incomplete_list.append(split_is_final_iteration_known_incomplete)
    _rechunk_dataset(old_zarr_path_=temporary_combined_output_path0, new_zarr_path_=temporary_combined_output_path1,
                     iterations_=iterations, cpus_=cpus, chains_=chains, parameter_indexes_=parameter_indexes,
                     new_iteration_chunk_size_=1_000)
    shutil.rmtree(temporary_combined_output_path0)
    if not any(split_is_final_iteration_known_incomplete_list):  # All false, meaning we should add the final iteration.
        _save_final_iteration_region(temporary_combined_output_path1, final_iteration_parameters_batch,
                                     final_iteration_log_likelihood_batch,
                                     max_known_complete_iteration + 1, cpus, chains,
                                     parameter_count, parameter_indexes)
    if combined_output_path.suffix == '.zip':
        dataset = xarray.open_zarr(temporary_combined_output_path1)
        temporary_combined_output_zip_path1 = temporary_combined_output_path1.parent.joinpath(
            temporary_combined_output_path1.name + '.zip')
        dataset.to_zarr(temporary_combined_output_zip_path1, mode='w')
        shutil.rmtree(temporary_combined_output_path1)
        temporary_combined_output_zip_path1.rename(combined_output_path)
    else:
        temporary_combined_output_path1.rename(combined_output_path)


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


def _get_known_complete_iterations(split_data_file_paths_, elements_per_record_):
    logger.info(f'Scanning first file to get iteration count.')
    split_data_path0 = split_data_file_paths_[0]
    record_generator_ = constantinos_kalapotharakos_format_record_generator(
        split_data_path0, elements_per_record=elements_per_record_)
    data_file0_iterations = 0
    read_chain0 = False
    for _ in record_generator_:
        if read_chain0:
            data_file0_iterations += 1
            read_chain0 = False
        else:
            read_chain0 = True
    if read_chain0:
        max_known_complete_iteration_ = data_file0_iterations - 1
    else:
        max_known_complete_iteration_ = data_file0_iterations - 2
    return max_known_complete_iteration_


def _create_empty_dataset_zarr(zarr_path_, iterations_, cpus_, chains_, parameter_indexes_, iteration_chunk_size_,
                               cpu_chunk_size_: int = 1):
    empty_dataset = xarray.Dataset(
        coords={
            'iteration': iterations_,
            'cpu': cpus_,
            'chain': chains_,
            'parameter_index': parameter_indexes_,
        },
        data_vars={
            'parameter': (
                ['iteration', 'cpu', 'chain', 'parameter_index'],
                dask.array.full((iterations_.size, cpus_.size, chains_.size, parameter_indexes_.size),
                                fill_value=np.nan, chunks=(iteration_chunk_size_, cpu_chunk_size_, -1, -1),
                                dtype=np.float32),
            ),
            'log_likelihood': (
                ['iteration', 'cpu', 'chain'],
                dask.array.full((iterations_.size, cpus_.size, chains_.size),
                                fill_value=np.nan, chunks=(iteration_chunk_size_, cpu_chunk_size_, -1),
                                dtype=np.float32),
            ),
        },
    )
    encoding = {
        'iteration': {'dtype': 'int64', 'chunks': (iteration_chunk_size_,)},
        'cpu': {'dtype': 'int64', 'chunks': (cpu_chunk_size_,)},
        'chain': {'dtype': 'int64', 'chunks': (-1,)},
        'parameter_index': {'dtype': 'int64', 'chunks': (-1,)},
        'parameter': {'dtype': 'float32', 'chunks': (iteration_chunk_size_, cpu_chunk_size_, -1, -1)},
        'log_likelihood': {'dtype': 'float32', 'chunks': (iteration_chunk_size_, cpu_chunk_size_, -1)},
    }
    empty_dataset.to_zarr(zarr_path_, compute=False, encoding=encoding)


def _rechunk_dataset(old_zarr_path_, new_zarr_path_, iterations_, cpus_, chains_, parameter_indexes_,
                     new_iteration_chunk_size_):
    logger.info('Rechunking dataset.')
    _create_empty_dataset_zarr(new_zarr_path_, iterations_, cpus_, chains_, parameter_indexes_,
                               new_iteration_chunk_size_, cpu_chunk_size_=-1)
    old_dataset = xarray.open_zarr(old_zarr_path_)
    iteration_batches = np.split(iterations_,
                                 np.arange(new_iteration_chunk_size_, len(iterations_), new_iteration_chunk_size_))
    for iteration_batch in iteration_batches:
        logger.info(f'Rechunking iteration: {iteration_batch[0]}')
        batch: xarray.Dataset = old_dataset.sel({'iteration': iteration_batch})
        batch = batch.chunk({'iteration': new_iteration_chunk_size_, 'cpu': -1})
        batch.to_zarr(new_zarr_path_, region='auto')


def _check_for_existing_files(combined_output_path_, overwrite_):
    temporary_directory_ = combined_output_path_.parent
    temporary_combined_output_path0_ = temporary_directory_.joinpath(combined_output_path_.name + '.haplo_partial0')
    if temporary_combined_output_path0_.exists():
        shutil.rmtree(temporary_combined_output_path0_)
    temporary_combined_output_path1_ = temporary_directory_.joinpath(combined_output_path_.name + '.haplo_partial1')
    if temporary_combined_output_path1_.exists():
        shutil.rmtree(temporary_combined_output_path1_)
    if combined_output_path_.exists():
        if overwrite_:
            shutil.rmtree(combined_output_path_)
        else:
            raise FileExistsError(f'{combined_output_path_} needs to be created, but already exists. '
                                  f'Pass `overwrite=True` to overwrite.')
    return temporary_combined_output_path0_, temporary_combined_output_path1_


def _process_split_file(temporary_combined_output_path0_, split_data_path_, split_index_, elements_per_record_,
                        max_known_complete_iteration_, chains_, parameter_count_, parameter_indexes_,
                        scanning_iteration_chunk_size_) -> (list[tuple[float, ...]], list[float], bool):
    logger.info(f'Processing {split_data_path_}.')
    split_final_iteration_parameters_batch_: list[tuple[float, ...]] = []
    split_final_iteration_log_likelihood_batch_: list[float] = []
    split_is_final_iteration_known_incomplete_ = False
    record_generator = constantinos_kalapotharakos_format_record_generator(
        split_data_path_, elements_per_record=elements_per_record_)
    parameters_batch: list[tuple[float, ...]] = []
    log_likelihood_batch: list[float] = []
    split_data_frame_cpu_number = int(re.search(r'1(\d+)\.dat', split_data_path_.name).group(1))
    if split_index_ != split_data_frame_cpu_number:
        raise ValueError(f'A split MCMC output file was expected but not found. '
                         f'Expected a file for CPU number {split_index_}, but found {split_data_path_.name}.')
    chain = 0
    iteration = 0
    batch_start_iteration = iteration
    for record_index in itertools.count():
        record = next(record_generator)
        if chain != int(record[elements_per_record_ - 1]):
            raise ValueError(f'The chain did not match the expected value at record index {record_index}.')
        parameters_batch.append(record[:parameter_count_])
        log_likelihood_batch.append(record[parameter_count_])
        if chain == 1:
            chain = 0
            iteration += 1
            if iteration > batch_start_iteration + scanning_iteration_chunk_size_ or iteration > max_known_complete_iteration_:
                _save_batch_to_cpu_and_iteration_region(temporary_combined_output_path0_, parameters_batch,
                                                        log_likelihood_batch, batch_start_iteration, iteration,
                                                        split_data_frame_cpu_number, chains_, parameter_count_,
                                                        parameter_indexes_)
                batch_start_iteration = iteration
                parameters_batch = []
                log_likelihood_batch = []
            if iteration > max_known_complete_iteration_:
                try:
                    record = next(record_generator)
                    split_final_iteration_parameters_batch_.append(record[:parameter_count_])
                    split_final_iteration_log_likelihood_batch_.append(record[parameter_count_])
                    record = next(record_generator)
                    split_final_iteration_parameters_batch_.append(record[:parameter_count_])
                    split_final_iteration_log_likelihood_batch_.append(record[parameter_count_])
                except StopIteration:
                    split_is_final_iteration_known_incomplete_ = True
                break
        else:
            chain = 1
    return (split_final_iteration_log_likelihood_batch_, split_final_iteration_parameters_batch_,
            split_is_final_iteration_known_incomplete_)
