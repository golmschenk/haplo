# Converting MCMC output files to Xarray Zarr

The split file text format the MCMC outputs data into is a bit cumbersome. The Xarray Zarr format makes the files much smaller (20x smaller), allows for random access indexing (meaning you can grab a random subset from the middle almost instantly), makes parallelization of analysis easy, and provides several other advanced benefits.

## What does Xarray Zarr mean?

[Xarray](https://docs.xarray.dev/en/stable/) and [Zarr](https://zarr.readthedocs.io/en/stable/) are two separate things that work together. Xarray is N-dimensional arrays with labels (sorta like Pandas, but for more dimensions), but also makes parallelization easy. Xarray is the form of the data from an abstract point of view. Zarr is the on-disk data format of the data. It's a format that allows reading parts directly from the disk without needing to load the entire array, but is still compressed at the same time. Xarray can take advantage of many file formats, Zarr being one of them. Zarr can be used by several data structure libraries, Xarray being one of them. For the most part, you only need to use the Xarray side of things. Just know that the file format this data is saved in is Zarr.

## Converting the data from the split `.dat` files to Zarr

To convert the data, you will need to pass the directory of the split MCMC output `.dat` files, where you want to put the Zarr file, and how many elements there are for each record in the `.dat` files. For example, the file might contain 11 parameters, 1 log likelihood, and 1 MCMC chain value for each record, resulting in 13 elements per record. Then the conversion would be accomplished by:

```python
from pathlib import Path
from haplo.analysis import combine_constantinos_kalapotharakos_split_mcmc_output_files_to_xarray_zarr

split_mcmc_output_directory = Path('path/to/split/mcmc/directory')
zarr_path = Path('path/to/output.zarr')  # Use a better name, but still use the `.zarr` extension.
combine_constantinos_kalapotharakos_split_mcmc_output_files_to_xarray_zarr(
    split_mcmc_output_directory=split_mcmc_output_directory,
    combined_output_path=zarr_path,
    elements_per_record=13
)
```

Currently, the conversion process is done in a single process. An accelerated multiprocess version is possible. If you believe it would be particularly useful to speed this process up, please report that.