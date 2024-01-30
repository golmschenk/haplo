from pathlib import Path
from typing import Optional, Callable, List

import numpy as np
import pandas as pd
import polars as pl
from sqlalchemy import create_engine
from torch.distributed import get_rank
from torch.utils.data import Dataset, Subset, get_worker_info, Sampler
from torch.multiprocessing import Manager

from haplo.data_paths import move_path_to_nvme, move_to_tmp_on_pbs


class NicerDataset(Dataset):
    def __init__(self, database_uri: str, length: int, parameters_transform: Optional[Callable] = None,
                 phase_amplitudes_transform: Optional[Callable] = None, in_memory: bool = False):
        self.database_uri = database_uri
        self.parameters_transform: Callable = parameters_transform
        self.phase_amplitudes_transform: Callable = phase_amplitudes_transform
        # TODO: Quick hack. Should not being doing logic in init. Move this to factory method.
        self.length: int = length
        self.engine = None
        self.connection = None
        self.in_memory = in_memory
        self.shared_data_frame = None

    @classmethod
    def new(cls, dataset_path: Path, parameters_transform: Optional[Callable] = None,
            phase_amplitudes_transform: Optional[Callable] = None, in_memory: bool = False,
            length: Optional[int] = None):
        database_uri = f'sqlite:///{dataset_path}?mode=ro'
        if length is None:
            engine = create_engine(database_uri)
            connection = engine.connect()
            count_data_frame = pl.read_database(query='select count(1) from main', connection=connection)
            count_row = count_data_frame.row(0)
            length = count_row[0]
        instance = cls(database_uri=database_uri, length=length, parameters_transform=parameters_transform,
                       phase_amplitudes_transform=phase_amplitudes_transform, in_memory=in_memory)
        return instance

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        initialize_connection(self)  # TODO: Probably shouldn't be necessary.
        row_index = index + 1  # The SQL database auto increments from 1, not 0.
        if self.in_memory:
            row = self.get_row_from_in_memory(row_index)
        else:
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

    def get_row_from_in_memory(self, row_index):
        row_data_frame = self.shared_data_frame.loc[row_index]
        row = row_data_frame.values
        return row

    def get_rows_from_indexes_with_row_id_column(self, row_indexes) -> pd.DataFrame:
        initialize_connection(self)
        row_indexes = list(np.asarray(row_indexes) + 1)
        indexes_sql_string = ', '.join(map(str, row_indexes))
        data_frame = pl.read_database(query=rf'select ROWID, * from main where ROWID in ({indexes_sql_string})',
                                          connection=self.connection)
        pandas_data_frame = data_frame.to_pandas()
        pandas_data_frame.set_index('rowid', inplace=True)
        return pandas_data_frame


def nicer_dataset_worker_initialization_function(worker_id: int) -> None:
    # TODO: Hacked way of getting dataset.
    worker_info = get_worker_info()
    dataset: NicerDataset = worker_info.dataset.dataset
    initialize_connection(dataset)

def initialize_connection(dataset: NicerDataset):
    if dataset.engine is None:
        dataset.database_uri = f'sqlite:///{dataset.dataset_path}?mode=ro'
        dataset.engine = create_engine(dataset.database_uri)
        dataset.connection = dataset.engine.connect()


def disconnect(dataset: NicerDataset):
    dataset.engine = None
    dataset.connection = None


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
