import re
import mmap
from pathlib import Path
from typing import TextIO, Dict, List

import polars as pl

from haplo.data_column_name import DataColumnName
from haplo.data_paths import constantinos_kalapotharakos_format_rotated_dataset_path, rotated_dataset_path, \
    constantinos_kalapotharakos_format_unrotated_dataset_path, unrotated_dataset_path


def constantinos_kalapotharakos_file_handle_to_polars(file_contents: bytes | mmap.mmap) -> pl.DataFrame:
    value_iterator = re.finditer(rb"[^\s]+", file_contents)
    list_of_dictionaries: List[Dict] = []
    data_frame = pl.from_dicts([], schema={str(name): pl.Float32 for name in DataColumnName})
    count = 0
    while True:
        parameters = []
        try:
            parameters.append(float(next(value_iterator).group(0)))
        except StopIteration:
            break
        for _ in range(10):
            parameters.append(float(next(value_iterator).group(0)))
        _ = float(next(value_iterator).group(0))  # Likelihood in Constantinos' output which has no meaning here.
        phase_amplitudes = []
        for _ in range(64):
            phase_amplitudes.append(float(next(value_iterator).group(0)))
        row_values = parameters + phase_amplitudes
        row_dictionary = {str(name): value for name, value in zip(DataColumnName, row_values)}
        list_of_dictionaries.append(row_dictionary)
        count += 1
        if len(list_of_dictionaries) % 100000 == 0:
            print(f'{count}', flush=True)
            chunk_data_frame = pl.from_dicts(list_of_dictionaries, schema={str(name): pl.Float32 for name in DataColumnName})
            data_frame = data_frame.vstack(chunk_data_frame)
            list_of_dictionaries = []
    chunk_data_frame = pl.from_dicts(list_of_dictionaries, schema={str(name): pl.Float32 for name in DataColumnName})
    data_frame = data_frame.vstack(chunk_data_frame)
    return data_frame


def get_memory_mapped_file_contents(file_handle: TextIO) -> mmap.mmap:
    file_fileno = file_handle.fileno()
    file_contents = mmap.mmap(file_fileno, 0, access=mmap.ACCESS_READ)
    return file_contents


def constantinos_kalapotharakos_format_file_to_arrow_file(input_file_path: Path, output_file_path: Path):
    with input_file_path.open() as file_handle:
        file_contents = get_memory_mapped_file_contents(file_handle)
        data_frame = constantinos_kalapotharakos_file_handle_to_polars(file_contents)
        # data_frame = data_frame.sample(frac=1.0, seed=0)
        data_frame.write_ipc(output_file_path)


if __name__ == '__main__':
    # constantinos_kalapotharakos_format_file_to_arrow_file(
    #     Path('data/mcmc_vac_all_50m.dat'), Path('data/50m_unshuffled_rotated_parameters_and_phase_amplitudes.arrow'))
    # constantinos_kalapotharakos_format_file_to_arrow_file(
    #     constantinos_kalapotharakos_format_rotated_dataset_path, rotated_dataset_path)
    constantinos_kalapotharakos_format_file_to_arrow_file(
        Path('data/mcmc_vac_all_800k.dat'), Path('data/check.arrow'))