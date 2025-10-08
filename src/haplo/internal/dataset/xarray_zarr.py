import xarray
from pathlib import Path
from typing import Self
from xarray import Dataset
from zarr.experimental.cache_store import CacheStore
from zarr.storage import ZipStore, LocalStore, MemoryStore

from haplo.internal.sized_dataset import SizedDataset


class XarrayBasedDataset(SizedDataset):
    @classmethod
    def new(cls, zarr_path: Path, memory_cached: bool = True) -> Self:
        if zarr_path.suffix == '.zip':
            store = ZipStore(zarr_path)
        else:
            store = LocalStore(zarr_path)
        if memory_cached:
            memory_store = MemoryStore()
            store = CacheStore(store=store, cache_store=memory_store)
        xarray_dataset: Dataset = xarray.open_zarr(store)
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
