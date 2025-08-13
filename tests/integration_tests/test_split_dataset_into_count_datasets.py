import numpy as np
import xarray

from haplo.internal.xarray_zarr_dataset import XarrayBasedDataset
from haplo.nicer_dataset import split_dataset_into_count_datasets


def test_split_dataset_into_count_datasets_with_xarray_dataset():
    xarray_dataset = xarray.Dataset({'input': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                     'output': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]},
                                    coords={'index': np.arange(10, dtype=np.int64)})
    xarray_based_dataset = XarrayBasedDataset(xarray_dataset)
    subset0, subset1, subset2 = split_dataset_into_count_datasets(xarray_based_dataset, [2, 3])
    assert len(subset0) == 2
    assert len(subset1) == 3
    assert len(subset2) == 5
    assert subset0[0][0] == 0
    assert subset1[2][1] == 40
