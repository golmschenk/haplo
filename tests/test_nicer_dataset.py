from pathlib import Path

import pandas as pd
import sqlite3

from haplo.nicer_dataset import NicerDataset, split_dataset_into_count_datasets, split_dataset_into_fractional_datasets

# TODO: The test datasets should be in memory, not saved to disk.
database_path = Path(__file__ + '.temp.db')


def create_fake_data(rows: int = 3):
    database_path.unlink(missing_ok=True)
    database_path.parent.mkdir(exist_ok=True, parents=True)
    connection = sqlite3.connect(database_path)
    fake_data = []
    parameter_count = 11
    phase_amplitude_count = 64
    parameter_column_names = [f'parameter{index}' for index in range(parameter_count)]
    phase_amplitude_column_names = [f'phase_amplitude{index}' for index in range(phase_amplitude_count)]
    data_column_names = parameter_column_names + phase_amplitude_column_names
    for index in range(rows):
        fake_data.append({name: name_index + (64 * index) for name_index, name in enumerate(data_column_names)})
    data_frame = pd.DataFrame(fake_data)
    data_frame.to_sql(name='main', con=connection, index=False)


def test_getitem():
    create_fake_data()
    dataset = NicerDataset.new(database_path)
    parameters1, phase_amplitudes1 = dataset[1]
    assert parameters1[3] == 67
    assert phase_amplitudes1[3] == 78


def test_len():
    create_fake_data()
    dataset = NicerDataset.new(database_path)
    assert len(dataset) == 3


def test_len_after_factional_split():
    create_fake_data(rows=8)
    full_dataset = NicerDataset.new(database_path)
    fractional_dataset0, fractional_dataset1 = split_dataset_into_fractional_datasets(full_dataset, [0.25, 0.75])
    assert len(fractional_dataset0) == 2
    assert len(fractional_dataset1) == 6


def test_len_after_count_split():
    create_fake_data(rows=8)
    full_dataset = NicerDataset.new(database_path)
    count_dataset0, count_dataset1, count_dataset2 = split_dataset_into_count_datasets(full_dataset, [2, 5])
    assert len(count_dataset0) == 2
    assert len(count_dataset1) == 5
    assert len(count_dataset2) == 1
