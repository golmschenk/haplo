import re
from pathlib import Path
from typing import TextIO, Dict, List

import numpy as np
import pandas as pd

from haplo.data_column_name import DataColumnName
from haplo.data_paths import constantinos_kalapotharakos_format_dataset_path, dataset_path


def constantinos_kalapotharakos_file_handle_to_pandas(file_handle: TextIO) -> pd.DataFrame:
    value_iterator = re.finditer(r"[^\s]+", file_handle.read())
    list_of_dictionaries: List[Dict] = []
    while True:
        parameters = []
        try:
            parameters.append(float(next(value_iterator).group(0)))
        except StopIteration:
            break
        for _ in range(10):
            parameters.append(float(next(value_iterator).group(0)))
        _ = float(next(value_iterator).group(0))  # Likelihood in Constantinos' output which has not meaning here.
        phase_amplitudes = []
        for _ in range(64):
            phase_amplitudes.append(float(next(value_iterator).group(0)))
        row_values = parameters + phase_amplitudes
        row_dictionary = {str(name): value for name, value in zip(DataColumnName, row_values)}
        list_of_dictionaries.append(row_dictionary)
    data_frame = pd.DataFrame(list_of_dictionaries, dtype=np.float64)
    return data_frame


def constantinos_kalapotharakos_format_file_to_arrow_file(input_file_path: Path, output_file_path: Path):
    with input_file_path.open() as file:
        data_frame = constantinos_kalapotharakos_file_handle_to_pandas(file)
        data_frame.to_feather(output_file_path)


if __name__ == '__main__':
    constantinos_kalapotharakos_format_file_to_arrow_file(constantinos_kalapotharakos_format_dataset_path, dataset_path)
