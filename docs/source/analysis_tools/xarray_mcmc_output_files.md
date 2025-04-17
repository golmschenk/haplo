# Xarray MCMC output files

The split file text format the MCMC outputs data into is a bit cumbersome. The Xarray Zarr format makes the output smaller (~7x smaller), allows for random access indexing (meaning you can grab a random subset from the middle almost instantly), does not require loading the full file into memory (allowing low memory nodes to perform full file analysis), makes parallel processing of analysis easy, and provides several other benefits in advanced use cases.

## High-level API

There's a small high-level API that allows you to get the smaller file size and some of the quick subset extraction benefits without knowing how it works. Learning the basics of Xarray will enable a lot of extra benefits, but this high-level API doesn't require that.

First convert your dataset to the Zarr file format. This part, unfortunately, still requires substantial time and should probably be run as a job on clusters (see {ref}`converting_the_data_from_the_split_dat_files_to_zarr` for more details):
```python
from pathlib import Path
from haplo.analysis import combine_constantinos_kalapotharakos_split_mcmc_output_files_to_xarray_zarr
from haplo.logging import enable_logger

split_mcmc_output_directory = Path('path/to/split/mcmc/directory')
zarr_path = Path('path/to/output.zarr.zip')  # Use a better name, but still use the `.zarr.zip` extension.
enable_logger()  # Optional. Will add printing of some progress information.
combine_constantinos_kalapotharakos_split_mcmc_output_files_to_xarray_zarr(
    split_mcmc_output_directory=split_mcmc_output_directory,
    combined_output_path=zarr_path,
    elements_per_record=13,
    multiprocess_pool_size=28,
)
```

Once converted, you can open the Xarray dataset from the Zarr file using:
```python
import xarray

dataset = xarray.open_zarr(zarr_path)
```

Xarray automatically parallelizes many tasks. This is normally useful on your own machine or a job node, as it speeds things up. However, if you just want to do a couple quick operations on a cluster login node (such as the PFE login node), the node often gets mad at you for trying to use all the available CPUs. We can explicitly tell Xarray (which uses Dask under-the-hood) to only use a single core using:
```python
from multiprocessing.pool import ThreadPool
import dask

dask.config.set(schedular='threads', pool=ThreadPool(1))  # Sets processing to a single core this for the rest of the script.
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
The resulting data frame values will be the parameters and log likelihoods. It will have a [Pandas MultiIndex](https://pandas.pydata.org/docs/user_guide/advanced.html#multiindex-advanced-indexing) that includes the iteration, cpu, and chain for each entry (but the values of the data frame are only the parameters and log likelihood). If you're not used to Pandas MultiIndexes, you can use Pandas' `reset_index` to convert these to regular columns. From here, you could save the data frame to a CSV or perform analysis using Pandas' normal methods.

The `mcmc_output_xarray_dataset_to_pandas_data_frame` function also accepts optional `limit_from_end` and `random_sample_size` arguments. Setting `limit_from_end` will make the export to Pandas export only the last N rows, where N is the value that's set. Similarly, setting `random_sample_size` to N will make the export take a random sample of N from the dataset that's passed. Note, that you can use the iteration slicing of the dataset before applying these export limits. Slicing to an iteration followed by `limit_from_end=100_000` will take only a few seconds, regardless of where you take the states from in the dataset. Because of how the data is stored, `random_sample_size=100_000` will still take longer, because it will access each chunk of the data array on-disk. When using a single core, on `nobackup` data, for a dataset with 1.3M iterations, it usually takes about 15 minutes. Most of this time is due to the slow read speed of `nobackup`, and this will probably highly depend on `nobackup` traffic that day. That said, this can still be run on a login node, without submitting a job, as it never uses substantial amounts of memory.

There is also a high-level function for getting the median log-likelihoods over windows of iterations of the data.
```python
from haplo.analysis import extract_windowed_median_log_likelihood_series

medians = extract_windowed_median_log_likelihood_series(dataset, window_size=1000)
```
This provides a Pandas Series with the indexes being the starts of the windows and the values being the median of those windows. Again, since this method accesses every part of the data array, it can take a while to run (e.g., 15 minutes with a single core when read from `nobackup`). However, it does not require substantial memory and can be run on a login node without submitting a job.

## What is Xarray and Zarr? What is the structure of this data?

[Xarray](https://docs.xarray.dev/en/stable/) and [Zarr](https://zarr.readthedocs.io/en/stable/) are two separate things that work together.

Xarray is N-dimensional arrays with labels (sort of like Pandas, but for more dimensions), but also makes parallelization easy. Xarray is the form of the data from an abstract point of view. In this format, the data is stored in a `Dataset` object, which contains two `DataArray`s. One is the array that contains the parameters of the MCMC states and one is the array that contains the log likelihood of the states. The parameter array is a 4D array with the dimensions being `[iteration, cpu, chain, parameter_index]`. The log likelihood array is a 3D array with dimensions `[iteration, cpu, chain]`. These two arrays share the overlapping dimensions, so you can take slices of both arrays at the same time along those dimensions.

```{image} mcmc_output_xarray_data_structure.png
:width: 400px
```

Zarr is the on-disk data format of the data. It's a format that allows reading parts directly from the disk without needing to load the entire array, but is still compressed at the same time.

Xarray can take advantage of many file formats, Zarr being one of them. Zarr can be used by several data structure libraries, Xarray being one of them. For the most part, you only need to use the Xarray side of things. Just know that the file format this data is saved in is Zarr.

(converting_the_data_from_the_split_dat_files_to_zarr)=
## Converting the data from the split `.dat` files to Zarr

To convert the data, you will need to pass the directory of the split MCMC output `.dat` files, where you want to put the Zarr file, and how many elements there are for each record in the `.dat` files. For example, the file might contain 11 parameters, 1 log likelihood, and 1 MCMC chain value for each record, resulting in 13 elements per record. Then the conversion would be accomplished by:

```python
from pathlib import Path
from haplo.analysis import combine_constantinos_kalapotharakos_split_mcmc_output_files_to_xarray_zarr
from haplo.logging import enable_logger
enable_logger()  # Optional. Will add printing of some progress information.
split_mcmc_output_directory = Path('path/to/split/mcmc/directory')
zarr_path = Path('path/to/output.zarr.zip')  # Use a better name, but still use the `.zarr.zip` extension.
combine_constantinos_kalapotharakos_split_mcmc_output_files_to_xarray_zarr(
    split_mcmc_output_directory=split_mcmc_output_directory,
    combined_output_path=zarr_path,
    elements_per_record=13,
    multiprocess_pool_size=28,
)
```

`multiprocess_pool_size` should be set to the number of available CPU cores for best performances. This can then be run NASA's Pleiades machines using something like:

```shell
#PBS -l select=1:ncpus=28:model=bro:mem=100GB
#PBS -l place=scatter:excl
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -k eod
#PBS -r n
#PBS -W group_list=s2853
#PBS -q long@pbspl1
job_description="combine_split_mcmc_files"
current_time=$(date "+%Y_%m_%d_%H_%M_%S")
qalter -o "${current_time}_${job_description}_${PBS_JOBID}.log" $PBS_JOBID

module use -a /swbuild/analytix/tools/modulefiles
module load miniconda3/v4
source activate haplo_env  # Use your environment here.
export OMP_NUM_THREADS=56  # 2x the number of CPUs (might not be necessary, haven't checked).

python combine_split_mcmc_files.py
```

With 1.3M iterations, this took about 10 hours.


## Working directly with the Xarray dataset

See the [full Xarray documentation](https://docs.xarray.dev/en/stable/) for everything you can do with it. Below are just a couple examples to get started.

You can open the Xarray dataset from the Zarr file using:
```python
import xarray

dataset = xarray.open_zarr(zarr_path)
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
The `compute()` is necessary, because by default Xarray is "lazy" in that it avoids unnecessary computation by only computing values (and intermediate values) for the results you explicitly request. It's also worth noting that this will by default run being computed from the disk, so the entire set of values never needs to be loaded into memory at once.

The `parameter` and `log_likelihood` arrays within the dataset share the `[iteration, cpu, chain]` dimensions. So, you can get a subset of both at the same time. For example,
```python
iteration_sliced_dataset = dataset.sel({'iteration': slice(100, 200)})
```
Will get a new dataset which is the subsample of the dataset for iterations 100 through 200. Note, this follows Pandas style labeled indexing [where endpoints are inclusive](https://pandas.pydata.org/docs/user_guide/advanced.html#endpoints-are-inclusive). Position-based indexing is also possible using `isel`, where endpoints are exclusive (again following Pandas). More general indexing and selecting rules for Xarray are similar to Pandas, but details can be found [here](https://docs.xarray.dev/en/latest/user-guide/indexing.html).

Notably for the MCMC output data, more specific selection can often be useful for other analyses. For example, if you wanted to follow what an individual MCMC chain on one CPU did, you could use:
```python
specific_chain_dataset = dataset.sel({'cpu': 7, 'chain': 0})
```

You can convert the contents of the sliced Xarray dataset to a NumPy array. Just be sure you have selected a small enough sub-portion of the full dataset that it will all fit in memory. Once it's ready, you could do something like: 
```python
numpy_array = dataset['parameter'].to_numpy()  # This will still be in the [iteration, cpu, chain, parameter_index] shape.
numpy_array = numpy_array.reshape([-1, 11])  # Assuming 11 parameters, this will give the parameter sets in an array with the first dimension being the state index.
```
