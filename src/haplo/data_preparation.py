from __future__ import annotations

import math

import logging
import mmap
import numpy as np
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import polars as pl

from haplo.internal.data_conversion import get_memory_mapped_file_contents
from haplo.logging import set_up_default_logger

logger = logging.getLogger(__name__)


def arbitrary_constantinos_kalapotharakos_file_path_to_pandas(data_path: Path, columns_per_row: int,
                                                              skip_rows: int = 0, limit: int | None = None
                                                              ) -> pd.DataFrame:
    polars_data_frame = arbitrary_constantinos_kalapotharakos_file_handle_to_polars(
        data_path=data_path, columns_per_row=columns_per_row, skip_rows=skip_rows, limit=limit)
    pandas_data_frame = polars_data_frame.to_pandas()
    return pandas_data_frame


def arbitrary_constantinos_kalapotharakos_file_handle_to_polars(data_path: Path, columns_per_row: int,
                                                                skip_rows: int = 0, limit: int | None = None
                                                                ) -> pl.DataFrame:
    with data_path.open() as file_handle:
        file_contents = get_memory_mapped_file_contents(file_handle)
        return arbitrary_constantinos_kalapotharakos_file_contents_to_polars(
            file_contents, columns_per_row, skip_rows=skip_rows, limit=limit)


def combine_constantinos_kalapotharakos_split_output_files_to_csv(root_directory_path: Path, combined_output_path: Path,
                                                                  columns_per_row: int) -> None:
    split_data_frames: list[pl.DataFrame] = []
    for split_data_path in sorted(root_directory_path.glob('*.dat')):
        print(f'Processing {split_data_path}.')
        split_data_frame = arbitrary_constantinos_kalapotharakos_file_handle_to_polars(split_data_path,
                                                                                       columns_per_row=columns_per_row)
        rename_dictionary: dict[str, str] = {}
        for column_index in range(columns_per_row - 2):
            rename_dictionary[str(column_index)] = f'parameter{column_index}'
        rename_dictionary[str(columns_per_row - 2)] = f'log_likelihood'
        rename_dictionary[str(columns_per_row - 1)] = f'chain'
        split_data_frame = split_data_frame.rename(rename_dictionary)
        split_data_frame = split_data_frame.with_columns(split_data_frame["chain"].cast(pl.Int64).alias("chain"))
        split_data_frame_cpu_number = int(re.search('1(\d+)\.dat', split_data_path.name).group(1))
        split_data_frame = split_data_frame.with_columns(pl.lit(split_data_frame_cpu_number).alias('cpu'))
        iterations = math.ceil(split_data_frame.height / 2)
        iteration_array = np.arange(iterations, dtype=np.int64)
        combined_iteration_array = np.empty((iteration_array.size * 2), dtype=iteration_array.dtype)
        combined_iteration_array[0::2] = iteration_array
        combined_iteration_array[1::2] = iteration_array
        if split_data_frame.height % 2 != 0:
            combined_iteration_array = combined_iteration_array[:-1]
        split_data_frame = split_data_frame.with_columns(
            pl.Series(name='iteration', values=combined_iteration_array, dtype=pl.Int64))
        split_data_frames.append(split_data_frame)
    combined_data_frame = pl.concat(split_data_frames)
    combined_data_frame.write_csv(combined_output_path)


def arbitrary_constantinos_kalapotharakos_file_contents_to_polars(file_contents: bytes | mmap.mmap,
                                                                  columns_per_row: int, skip_rows: int = 0,
                                                                  limit: int | None = None) -> pl.DataFrame:
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
        if skip_rows > 0:
            skip_rows -= 1
            continue
        row_dictionary = {str(index): value for index, value in zip(range(columns_per_row), values)}
        list_of_dictionaries.append(row_dictionary)
        count += 1
        if limit is not None and count >= limit:
            break
        if len(list_of_dictionaries) % 100000 == 0:
            logger.info(f'Processed {count} rows.')
            chunk_data_frame = pl.from_dicts(list_of_dictionaries,
                                             schema={str(index): pl.Float32 for index in range(columns_per_row)})
            data_frame = data_frame.vstack(chunk_data_frame)
            list_of_dictionaries = []
    chunk_data_frame = pl.from_dicts(list_of_dictionaries,
                                     schema={str(index): pl.Float32 for index in range(columns_per_row)})
    data_frame = data_frame.vstack(chunk_data_frame)
    return data_frame


if __name__ == '__main__':
    combine_constantinos_kalapotharakos_split_output_files_to_csv(
        Path('/Users/golmschenk/Downloads/mcmc_vac_all_5m_cont_with_phys'),
        Path('/Users/golmschenk/Downloads/mcmc_vac_all_5m_cont_with_phys.csv'),
        columns_per_row=13,
    )
