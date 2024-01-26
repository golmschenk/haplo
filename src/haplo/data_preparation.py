import logging
import mmap
import re
from pathlib import Path
from typing import TextIO, Dict, List

import polars as pl

from haplo.data_column_name import DataColumnName
from haplo.logging import set_up_default_logger

logger = logging.getLogger(__name__)


def constantinos_kalapotharakos_file_handle_to_sqlite(file_contents: bytes | mmap.mmap, output_file_path: Path
                                                      ):
    set_up_default_logger()
    output_file_path.unlink(missing_ok=True)
    Path(str(output_file_path) + '-shm').unlink(missing_ok=True)
    Path(str(output_file_path) + '-wal').unlink(missing_ok=True)
    Path(str(output_file_path) + '-journal').unlink(missing_ok=True)
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
            logger.info(f'Processed {count} lines.')
            chunk_data_frame = pl.from_dicts(list_of_dictionaries, schema={str(name): pl.Float32 for name in DataColumnName})
            chunk_data_frame.write_database('main', f'sqlite:///{output_file_path}', if_exists='append')
            list_of_dictionaries = []
    chunk_data_frame = pl.from_dicts(list_of_dictionaries, schema={str(name): pl.Float32 for name in DataColumnName})
    chunk_data_frame.write_database('main', f'sqlite:///{output_file_path}', if_exists='append')


def arbitrary_constantinos_kalapotharakos_file_handle_to_polars(data_path: Path, columns_per_row: int
                                                                ) -> pl.DataFrame:
    with data_path.open() as file_handle:
        file_contents = get_memory_mapped_file_contents(file_handle)
        return arbitrary_constantinos_kalapotharakos_file_contents_to_polars(file_contents, columns_per_row)


def arbitrary_constantinos_kalapotharakos_file_contents_to_polars(file_contents: bytes | mmap.mmap, columns_per_row: int):
    set_up_default_logger()
    value_iterator = re.finditer(rb"[^\s]+", file_contents)
    list_of_dictionaries: List[Dict] = []
    data_frame = pl.from_dicts([], schema={str(index): pl.Float32 for index in range(columns_per_row)})
    count = 0
    while True:
        values = []
        try:
            values.append(float(next(value_iterator).group(0)))
        except StopIteration:
            break
        for _ in range(columns_per_row - 1):
            values.append(float(next(value_iterator).group(0)))
        row_dictionary = {str(index): value for index, value in zip(range(columns_per_row), values)}
        list_of_dictionaries.append(row_dictionary)
        count += 1
        if len(list_of_dictionaries) % 100000 == 0:
            logger.info(f'Processed {count} lines.')
            chunk_data_frame = pl.from_dicts(list_of_dictionaries,
                                             schema={str(index): pl.Float32 for index in range(columns_per_row)})
            data_frame = data_frame.vstack(chunk_data_frame)
            list_of_dictionaries = []
    chunk_data_frame = pl.from_dicts(list_of_dictionaries,
                                     schema={str(index): pl.Float32 for index in range(columns_per_row)})
    data_frame = data_frame.vstack(chunk_data_frame)
    return data_frame


def get_memory_mapped_file_contents(file_handle: TextIO) -> mmap.mmap:
    file_fileno = file_handle.fileno()
    file_contents = mmap.mmap(file_fileno, 0, access=mmap.ACCESS_READ)
    return file_contents


def constantinos_kalapotharakos_format_file_to_sqlite(input_file_path: Path, output_file_path: Path) -> None:
    """
    Produces an SQLite database from a Constantinos Kalapotharakos format file. The expected input format includes
    11 parameters, 1 likelihood value (which is ignored), and 64 phase amplitude values for each entry.

    :param input_file_path: The Path to the Constantinos Kalapotharakos format file.
    :param output_file_path: The Path to the output SQLite database.
    """
    with input_file_path.open() as file_handle:
        file_contents = get_memory_mapped_file_contents(file_handle)
        constantinos_kalapotharakos_file_handle_to_sqlite(file_contents, output_file_path)


if __name__ == '__main__':
    constantinos_kalapotharakos_format_file_to_sqlite(
        Path('data/mcmc_vac_all_640m_A.dat'), Path('data/640m_rotated_parameters_and_phase_amplitudes.db'))
