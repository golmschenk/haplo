from pathlib import Path
from typing import Optional, Callable, List

import numpy as np
import pandas as pd
from pyarrow import feather
from torch.utils.data import Dataset, Subset

from haplo.data_column_name import DataColumnName


class NicerDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, parameters_transform: Optional[Callable] = None,
                 phase_amplitudes_transform: Optional[Callable] = None):
        self.parameters_transform: Callable = parameters_transform
        self.phase_amplitudes_transform: Callable = phase_amplitudes_transform
        self.data_frame: pd.DataFrame = data_frame
        # if self.data_frame.shape[0] > 50_000_000:
        #     self.data_frame = self.data_frame.head(50_000_000)

    @classmethod
    def new(cls, dataset_path: Path, parameters_transform: Optional[Callable] = None,
            phase_amplitudes_transform: Optional[Callable] = None):
        data_frame = feather.read_feather(dataset_path, memory_map=True)
        instance = cls(data_frame=data_frame, parameters_transform=parameters_transform,
                       phase_amplitudes_transform=phase_amplitudes_transform)
        return instance

    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, index):
        row = self.data_frame.iloc[index]
        parameters = row.loc[[
            DataColumnName.PARAMETER0,
            DataColumnName.PARAMETER1,
            DataColumnName.PARAMETER2,
            DataColumnName.PARAMETER3,
            DataColumnName.PARAMETER4,
            DataColumnName.PARAMETER5,
            DataColumnName.PARAMETER6,
            DataColumnName.PARAMETER7,
            DataColumnName.PARAMETER8,
            DataColumnName.PARAMETER9,
            DataColumnName.PARAMETER10,
        ]].values
        phase_amplitudes = row.loc[[
            DataColumnName.PHASE_AMPLITUDE0,
            DataColumnName.PHASE_AMPLITUDE1,
            DataColumnName.PHASE_AMPLITUDE2,
            DataColumnName.PHASE_AMPLITUDE3,
            DataColumnName.PHASE_AMPLITUDE4,
            DataColumnName.PHASE_AMPLITUDE5,
            DataColumnName.PHASE_AMPLITUDE6,
            DataColumnName.PHASE_AMPLITUDE7,
            DataColumnName.PHASE_AMPLITUDE8,
            DataColumnName.PHASE_AMPLITUDE9,
            DataColumnName.PHASE_AMPLITUDE10,
            DataColumnName.PHASE_AMPLITUDE11,
            DataColumnName.PHASE_AMPLITUDE12,
            DataColumnName.PHASE_AMPLITUDE13,
            DataColumnName.PHASE_AMPLITUDE14,
            DataColumnName.PHASE_AMPLITUDE15,
            DataColumnName.PHASE_AMPLITUDE16,
            DataColumnName.PHASE_AMPLITUDE17,
            DataColumnName.PHASE_AMPLITUDE18,
            DataColumnName.PHASE_AMPLITUDE19,
            DataColumnName.PHASE_AMPLITUDE20,
            DataColumnName.PHASE_AMPLITUDE21,
            DataColumnName.PHASE_AMPLITUDE22,
            DataColumnName.PHASE_AMPLITUDE23,
            DataColumnName.PHASE_AMPLITUDE24,
            DataColumnName.PHASE_AMPLITUDE25,
            DataColumnName.PHASE_AMPLITUDE26,
            DataColumnName.PHASE_AMPLITUDE27,
            DataColumnName.PHASE_AMPLITUDE28,
            DataColumnName.PHASE_AMPLITUDE29,
            DataColumnName.PHASE_AMPLITUDE30,
            DataColumnName.PHASE_AMPLITUDE31,
            DataColumnName.PHASE_AMPLITUDE32,
            DataColumnName.PHASE_AMPLITUDE33,
            DataColumnName.PHASE_AMPLITUDE34,
            DataColumnName.PHASE_AMPLITUDE35,
            DataColumnName.PHASE_AMPLITUDE36,
            DataColumnName.PHASE_AMPLITUDE37,
            DataColumnName.PHASE_AMPLITUDE38,
            DataColumnName.PHASE_AMPLITUDE39,
            DataColumnName.PHASE_AMPLITUDE40,
            DataColumnName.PHASE_AMPLITUDE41,
            DataColumnName.PHASE_AMPLITUDE42,
            DataColumnName.PHASE_AMPLITUDE43,
            DataColumnName.PHASE_AMPLITUDE44,
            DataColumnName.PHASE_AMPLITUDE45,
            DataColumnName.PHASE_AMPLITUDE46,
            DataColumnName.PHASE_AMPLITUDE47,
            DataColumnName.PHASE_AMPLITUDE48,
            DataColumnName.PHASE_AMPLITUDE49,
            DataColumnName.PHASE_AMPLITUDE50,
            DataColumnName.PHASE_AMPLITUDE51,
            DataColumnName.PHASE_AMPLITUDE52,
            DataColumnName.PHASE_AMPLITUDE53,
            DataColumnName.PHASE_AMPLITUDE54,
            DataColumnName.PHASE_AMPLITUDE55,
            DataColumnName.PHASE_AMPLITUDE56,
            DataColumnName.PHASE_AMPLITUDE57,
            DataColumnName.PHASE_AMPLITUDE58,
            DataColumnName.PHASE_AMPLITUDE59,
            DataColumnName.PHASE_AMPLITUDE60,
            DataColumnName.PHASE_AMPLITUDE61,
            DataColumnName.PHASE_AMPLITUDE62,
            DataColumnName.PHASE_AMPLITUDE63,
        ]].values
        if self.parameters_transform is not None:
            parameters = self.parameters_transform(parameters.copy())
        if self.phase_amplitudes_transform is not None:
            phase_amplitudes = self.phase_amplitudes_transform(phase_amplitudes.copy())
        return parameters, phase_amplitudes


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
