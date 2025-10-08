from __future__ import annotations

from typing import TypeVar, List

import numpy as np
import torch
from torch.utils.data import Subset

from haplo.internal.sized_dataset import SizedDataset

_T_co = TypeVar("_T_co", covariant=True)


def split_dataset_into_count_datasets(dataset: SizedDataset[_T_co], counts: List[int]) -> List[Subset[_T_co]]:
    assert np.sum(counts) < len(dataset)
    count_datasets: List[Subset[_T_co]] = []
    next_index = 0
    previous_index = 0
    for count in counts:
        next_index += count
        indexes = torch.tensor(range(previous_index, next_index), dtype=torch.int32)
        count_dataset = Subset(dataset, indexes)
        count_datasets.append(count_dataset)
        previous_index = next_index
    indexes = torch.tensor(range(previous_index, len(dataset)), dtype=torch.int32)
    count_datasets.append(Subset(dataset, indexes))
    return count_datasets
