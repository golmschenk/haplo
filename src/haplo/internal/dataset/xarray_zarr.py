from pathlib import Path

import xarray
from typing import Self
from xarray import Dataset
from zarr.storage import ZipStore, LocalStore

from haplo.internal.sized_dataset import SizedDataset


class XarrayBasedDataset(SizedDataset):
    @classmethod
    def new(cls, zarr_path: Path) -> Self:
        if zarr_path.suffix == '.zip':
            store = ZipStore(zarr_path)
        else:
            store = LocalStore(zarr_path)
        xarray_dataset: Dataset = xarray.open_dataset(store, engine='zarr', chunks=None, cache=False)
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


class Xarray2dEnergyBinSubsampling(XarrayBasedDataset):
    def __getitem__(self, index):
        input_ = self.xarray_dataset['input'][index].to_numpy()
        output = self.xarray_dataset['output'][index, 0:400].to_numpy()
        return input_, output