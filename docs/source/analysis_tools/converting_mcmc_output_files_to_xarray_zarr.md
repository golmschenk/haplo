# Converting MCMC output files to Xarray Zarr

The split file text format the MCMC outputs data into is a bit cumbersome. The Xarray Zarr format makes the files much smaller (20x smaller), allows for random access indexing (meaning you can grab a random subset from the middle almost instantly), does not require loading the full file into memory (allowing low memory nodes to perform full dataset analysis), makes parallelization of analysis easy, and provides several other advanced benefits.

## High-level API

There's a small high-level API that allows you to get the smaller file size and some of the quick subset extraction benefits without knowing how it works. Learning the basics of Xarray will enable a lot of extra benefits, but this high-level API doesn't require that.

First convert your dataset to the Zarr file format (more explanation below):
```python
from pathlib import Path
from haplo.analysis import combine_constantinos_kalapotharakos_split_mcmc_output_files_to_xarray_zarr
from haplo.logging import enable_logger

split_mcmc_output_directory = Path('path/to/split/mcmc/directory')
zarr_path = Path('path/to/output.zarr')  # Use a better name, but still use the `.zarr` extension.
enable_logger()  # Optional. Will add printing of some progress information.
combine_constantinos_kalapotharakos_split_mcmc_output_files_to_xarray_zarr(
    split_mcmc_output_directory=split_mcmc_output_directory,
    combined_output_path=zarr_path,
    elements_per_record=13
)
```

Once converted, you can open the Xarray dataset from the Zarr file using:
```python
import xarray

dataset = xarray.load_dataset(zarr_path)
```

You can get a range of the MCMC iterations (start is inclusive, end is exclusive):
```python
from haplo.analysis import slice_iteration_of_mcmc_output_xarray_dataset

dataset = slice_iteration_of_mcmc_output_xarray_dataset(dataset, start_iteration=100, end_iteration=200)
```

You can export your subset dataset (or full dataset) to a Pandas data frame using:
```python
from haplo.analysis import mcmc_output_xarray_dataset_to_pandas_data_frame

data_frame = mcmc_output_xarray_dataset_to_pandas_data_frame(dataset)
```
The resulting data frame values will be the parameters and log likelihoods. It will have a [Pandas MultiIndex](https://pandas.pydata.org/docs/user_guide/advanced.html#multiindex-advanced-indexing) that includes the iteration, cpu, and chain for each entry (but the values of the data frame are only the parameters and log likelihood). From here, you could save to a CSV or perform analysis using Pandas' normal methods.

The `mcmc_output_xarray_dataset_to_pandas_data_frame` function also accepts optional `limit_from_end` and `random_sample_size` arguments. Setting `limit_from_end` will make the export to Pandas export only the last N rows, where N is the value that's set. Similarly, setting `random_sample_size` to N will make the export take a random sample of N from the dataset that's passed. Note, that you can use the iteration slicing of the dataset before applying these export limits.

## What is the Xarray and Zarr? What is the structure of this data?

[Xarray](https://docs.xarray.dev/en/stable/) and [Zarr](https://zarr.readthedocs.io/en/stable/) are two separate things that work together.

Xarray is N-dimensional arrays with labels (sort of like Pandas, but for more dimensions), but also makes parallelization easy. Xarray is the form of the data from an abstract point of view. In this format, the data is stored in a `Dataset` object, which contains two `DataArray`s. One is the array that contains the parameters of the MCMC states and one is the array that contains the log likelihood of the states. The parameter array is a 4D array with the dimensions being `[iteration, cpu, chain, parameter_index]`. The log likelihood array is a 3D array with dimensions `[iteration, cpu, chain]`. These two arrays share the overlapping dimensions, so you can take slices of both arrays at the same time along those dimensions.

```{image} converting_mcmc_output_files_to_xarray_zarr.md
:width: 400px
```

Zarr is the on-disk data format of the data. It's a format that allows reading parts directly from the disk without needing to load the entire array, but is still compressed at the same time.

Xarray can take advantage of many file formats, Zarr being one of them. Zarr can be used by several data structure libraries, Xarray being one of them. For the most part, you only need to use the Xarray side of things. Just know that the file format this data is saved in is Zarr.

## Converting the data from the split `.dat` files to Zarr

To convert the data, you will need to pass the directory of the split MCMC output `.dat` files, where you want to put the Zarr file, and how many elements there are for each record in the `.dat` files. For example, the file might contain 11 parameters, 1 log likelihood, and 1 MCMC chain value for each record, resulting in 13 elements per record. Then the conversion would be accomplished by:

```python
from pathlib import Path
from haplo.analysis import combine_constantinos_kalapotharakos_split_mcmc_output_files_to_xarray_zarr
from haplo.logging import enable_logger
enable_logger()  # Optional. Will add printing of some progress information.
split_mcmc_output_directory = Path('path/to/split/mcmc/directory')
zarr_path = Path('path/to/output.zarr')  # Use a better name, but still use the `.zarr` extension.
combine_constantinos_kalapotharakos_split_mcmc_output_files_to_xarray_zarr(
    split_mcmc_output_directory=split_mcmc_output_directory,
    combined_output_path=zarr_path,
    elements_per_record=13
)
```

Currently, the conversion process is done in a single process. An accelerated multiprocess version is possible. If you believe it would be particularly useful to speed this process up, please report that.

## Xarray manipulations

You can open the Xarray dataset from the Zarr file using:
```python
import xarray

dataset = xarray.load_dataset(zarr_path)
```

At any point after manipulating the Xarray dataset (say, after reducing the subsample by slicing on the iterations), you can save the updated dataset to another Zarr file using:
```python
dataset.to_zarr(another_zarr_path)
```
This is particularly useful if you want to perform some reductions of the data on a remote server, but then want a smaller Zarr file for further local processing.

Xarray will automatically multiprocess tasks. If you wanted to get the mean log likelihood value (with the computation automatically parallelized across the available CPUs), you can use:
```python
mean_log_likelihood_value = dataset['log_likelihood'].mean().compute()
```
The `compute()` is necessary, because by default Xarray is "lazy" in that it avoids unnecessary computation by only computing values (and intermediate values) for the results you explicitly request.

The `parameter` and `log_likelihood` arrays within the dataset share the `[iteration, cpu, chain]` dimensions. So, you can get a subset of both at the same time. For example,
```python
iteration_sliced_dataset = dataset.sel({'iteration': slice(100, 200)})
```
Will get a new dataset which is the subsample of the dataset for iterations 100 through 200. Note, this follows Pandas style labeled indexing [where endpoints are inclusive](https://pandas.pydata.org/docs/user_guide/advanced.html#endpoints-are-inclusive). Position-based indexing is also possible using `isel`, where endpoints are exclusive (again following Pandas). More general indexing and selecting rules for Xarray are similar to Pandas, but details can be found [here](https://docs.xarray.dev/en/latest/user-guide/indexing.html).

Notably for the MCMC output data, more specific selection can often be useful for other analyses. For example, if you wanted to follow what an individual MCMC chain on one CPU did, you could use:
```python
specific_chain_dataset = dataset.sel({'cpu': 7, 'chain': 0})
```
