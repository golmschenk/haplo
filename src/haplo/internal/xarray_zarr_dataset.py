import xarray
from pathlib import Path
from typing import Self
from xarray import Dataset

from haplo.internal.sized_dataset import SizedDataset


class XarrayBasedDataset(SizedDataset):
    @classmethod
    def new(cls, zarr_path: Path) -> Self:
        xarray_dataset: Dataset = xarray.open_zarr(zarr_path)
        instance = cls(xarray_dataset=xarray_dataset)
        return instance

    def __init__(self, xarray_dataset: Dataset):
        self.xarray_dataset: Dataset = xarray_dataset

    def __len__(self):
        return self.xarray_dataset['index'].size

    def __getitem__(self, index):
        input_ = self.xarray_dataset['input'][index].to_numpy()
        output = self.xarray_dataset['output'][index].to_numpy()
        return input_, output
