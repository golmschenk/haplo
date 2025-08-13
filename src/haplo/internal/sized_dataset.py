from typing import Sized

from abc import ABC
from torch.utils.data import Dataset


class SizedDataset(ABC, Dataset, Sized):
    pass
