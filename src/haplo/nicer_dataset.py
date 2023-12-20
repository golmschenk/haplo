from pathlib import Path
from typing import Optional, Callable, List

import numpy as np
import polars as pl
from sqlalchemy import create_engine
from torch.utils.data import Dataset, Subset

from haplo.data_paths import move_path_to_nvme, move_to_tmp_on_pbs


class NicerDataset(Dataset):
    def __init__(self, dataset_path: Path, parameters_transform: Optional[Callable] = None,
                 phase_amplitudes_transform: Optional[Callable] = None):
        self.dataset_path: Path = dataset_path
        self.parameters_transform: Callable = parameters_transform
        self.phase_amplitudes_transform: Callable = phase_amplitudes_transform
        self.database_uri = f'sqlite:///{self.dataset_path}?mode=ro'
        # TODO: Quick hack. Should not being doing logic in init. Move this to factory method.
        engine = create_engine(self.database_uri)
        connection = engine.connect()
        count_data_frame = pl.read_database(query='select count(1) from main', connection=connection)
        count_row = count_data_frame.row(0)
        count = count_row[0]
        self.length: int = count
        self.engine = None
        self.connection = None

    @classmethod
    def new(cls, dataset_path: Path, parameters_transform: Optional[Callable] = None,
            phase_amplitudes_transform: Optional[Callable] = None):
        instance = cls(dataset_path=dataset_path, parameters_transform=parameters_transform,
                       phase_amplitudes_transform=phase_amplitudes_transform)
        return instance

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # TODO: Horrible hack. This should happen on the initialization of each worker's dataset, not in the getitem.
        if self.engine is None:
            self.database_uri = f'sqlite:///{self.dataset_path}?mode=ro'
            self.engine = create_engine(self.database_uri)
            self.connection = self.engine.connect()
        row_index = index + 1  # The SQL database auto increments from 1, not 0.
        row = self.get_row_from_index(row_index)
        parameters = np.array(row[:11], dtype=np.float32)
        phase_amplitudes = np.array(row[11:], dtype=np.float32)
        if self.parameters_transform is not None:
            parameters = self.parameters_transform(parameters)
        if self.phase_amplitudes_transform is not None:
            phase_amplitudes = self.phase_amplitudes_transform(phase_amplitudes)
        return parameters, phase_amplitudes

    def get_row_from_index(self, row_index):
        row_data_frame = pl.read_database(query=rf'select * from main where ROWID = {row_index}',
                                          connection=self.connection)
        row = row_data_frame.row(0)
        return row


def split_into_train_validation_and_test_datasets(dataset: NicerDataset) -> (NicerDataset, NicerDataset, NicerDataset):
    length_10_percent = round(len(dataset) * 0.1)
    train_dataset = Subset(dataset, range(length_10_percent * 8))
    validation_dataset = Subset(dataset, range(length_10_percent * 8, length_10_percent * 9))
    test_dataset = Subset(dataset, range(length_10_percent * 9, len(dataset)))
    return train_dataset, validation_dataset, test_dataset


def split_dataset_into_fractional_datasets(dataset: NicerDataset, fractions: List[float]) -> List[NicerDataset]:
    assert np.isclose(np.sum(fractions), 1.0)
    fractional_datasets: List[NicerDataset] = []
    cumulative_fraction = 0
    previous_index = 0
    for fraction in fractions:
        cumulative_fraction += fraction
        if np.isclose(cumulative_fraction, 1.0):
            next_index = len(dataset)
        else:
            next_index = round(len(dataset) * cumulative_fraction)
        fractional_dataset: NicerDataset = Subset(dataset, range(previous_index, next_index))
        fractional_datasets.append(fractional_dataset)
        previous_index = next_index
    return fractional_datasets


def split_dataset_into_count_datasets(dataset: NicerDataset, counts: List[int]) -> List[NicerDataset]:
    assert np.sum(counts) < len(dataset)
    count_datasets: List[NicerDataset] = []
    next_index = 0
    previous_index = 0
    for count in counts:
        next_index += count
        count_dataset: NicerDataset = Subset(dataset, range(previous_index, next_index))
        count_datasets.append(count_dataset)
        previous_index = next_index
    count_datasets.append(Subset(dataset, range(previous_index, len(dataset))))
    return count_datasets
