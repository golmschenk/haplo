import pandas as pd

from haplo.data_column_name import DataColumnName
from haplo.nicer_dataset import NicerDataset, split_dataset_into_count_datasets, split_dataset_into_fractional_datasets


def test_getitem():
    fake_data = []
    for index in range(3):
        fake_data.append({name: name_index + (64 * index) for name_index, name in enumerate(DataColumnName)})
    data_frame = pd.DataFrame(fake_data)
    dataset = NicerDataset(data_frame)
    parameters1, phase_amplitudes1 = dataset[1]
    assert parameters1[3] == 67
    assert phase_amplitudes1[3] == 78


def test_len():
    fake_data = []
    for index in range(3):
        fake_data.append({name: name_index + (64 * index) for name_index, name in enumerate(DataColumnName)})
    data_frame = pd.DataFrame(fake_data)
    dataset = NicerDataset(data_frame)
    assert len(dataset) == 3


def test_len_after_factional_split():
    fake_data = []
    for index in range(8):
        fake_data.append({name: name_index + (64 * index) for name_index, name in enumerate(DataColumnName)})
    data_frame = pd.DataFrame(fake_data)
    full_dataset = NicerDataset(data_frame)
    fractional_dataset0, fractional_dataset1 = split_dataset_into_fractional_datasets(full_dataset, [0.25, 0.75])
    assert len(fractional_dataset0) == 2
    assert len(fractional_dataset1) == 6


def test_len_after_count_split():
    fake_data = []
    for index in range(8):
        fake_data.append({name: name_index + (64 * index) for name_index, name in enumerate(DataColumnName)})
    data_frame = pd.DataFrame(fake_data)
    full_dataset = NicerDataset(data_frame)
    count_dataset0, count_dataset1, count_dataset2 = split_dataset_into_count_datasets(full_dataset, [2, 5])
    assert len(count_dataset0) == 2
    assert len(count_dataset1) == 5
    assert len(count_dataset2) == 1
