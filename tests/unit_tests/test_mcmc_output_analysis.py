import numpy as np
import pytest
import xarray

from haplo.internal.mcmc_output_analysis import slice_iteration_of_mcmc_output_xarray_dataset, \
    mcmc_output_xarray_dataset_to_pandas_data_frame, extract_windowed_median_log_likelihood_series


@pytest.fixture
def sample_dataset():
    number_of_iterations = 10
    number_of_cpus = 4
    number_of_chains = 2
    number_of_parameters = 3

    iterations = np.arange(number_of_iterations)
    cpus = np.arange(number_of_cpus)
    chains = np.arange(number_of_chains)
    parameters = np.arange(number_of_parameters)

    parameter_data = np.random.rand(number_of_iterations, number_of_cpus, number_of_chains, number_of_parameters)
    log_likelihood_data = np.random.rand(number_of_iterations, number_of_cpus, number_of_chains)

    dataset = xarray.Dataset({
        'parameter': (['iteration', 'cpu', 'chain', 'parameter_index'], parameter_data),
        'log_likelihood': (['iteration', 'cpu', 'chain'], log_likelihood_data)
    }, coords={
        'iteration': iterations,
        'cpu': cpus,
        'chain': chains,
        'parameter_index': parameters
    })

    return dataset


def test_slice_iteration_of_mcmc_output_xarray_dataset(sample_dataset):
    start_iteration = 2
    end_iteration = 5

    sliced_dataset = slice_iteration_of_mcmc_output_xarray_dataset(
        sample_dataset, start_iteration=start_iteration, end_iteration=end_iteration
    )

    assert len(sliced_dataset['iteration']) == end_iteration - start_iteration
    assert all(value in range(start_iteration, end_iteration) for value in sliced_dataset['iteration'].values)


def test_mcmc_output_xarray_dataset_to_pandas_data_frame(sample_dataset):
    data_frame = mcmc_output_xarray_dataset_to_pandas_data_frame(sample_dataset)

    parameter_columns = [f'parameter{parameter_index}' for parameter_index in sample_dataset['parameter_index'].values]
    expected_columns = parameter_columns + ['log_likelihood']
    assert list(data_frame.columns) == expected_columns

    expected_rows = sample_dataset['iteration'].size * sample_dataset['cpu'].size * sample_dataset['chain'].size
    assert data_frame.shape[0] == expected_rows

    assert (data_frame.loc[5, 2, 1]['parameter1'] ==
            sample_dataset.sel({'iteration': 5, 'cpu': 2, 'chain': 1})['parameter'][1].item())


def test_mcmc_output_xarray_dataset_to_pandas_data_frame_with_limit_from_end(sample_dataset):
    data_frame = mcmc_output_xarray_dataset_to_pandas_data_frame(sample_dataset, limit_from_end=10)

    assert data_frame.shape[0] == 10


def test_mcmc_output_xarray_dataset_to_pandas_data_frame_with_sample_size(sample_dataset):
    data_frame = mcmc_output_xarray_dataset_to_pandas_data_frame(sample_dataset, random_sample_size=10)

    assert data_frame.shape[0] == 10


def get_toy_dataset():
    iterations = np.arange(20)
    cpus = np.arange(10)
    chains = np.arange(2)
    parameter_indexes = np.arange(11)
    dataset = xarray.Dataset(
        coords={
            'iteration': iterations,
            'cpu': cpus,
            'chain': chains,
            'parameter_index': parameter_indexes,
        },
        data_vars={
            'parameter': (
                ['iteration', 'cpu', 'chain', 'parameter_index'],
                np.arange(iterations.size * cpus.size * chains.size * parameter_indexes.size, dtype=np.float32
                          ).reshape([iterations.size, cpus.size, chains.size, parameter_indexes.size]),
            ),
            'log_likelihood': (
                ['iteration', 'cpu', 'chain'],
                np.arange(iterations.size * cpus.size * chains.size, dtype=np.float32
                          ).reshape([iterations.size, cpus.size, chains.size]),
            ),
        },
    )
    return dataset


def test_mcmc_output_xarray_dataset_to_pandas_data_frame_on_explicit_dataset():
    dataset = get_toy_dataset()

    data_frame = mcmc_output_xarray_dataset_to_pandas_data_frame(dataset)

    assert data_frame.shape == (400, 12)
    assert data_frame.index.levshape == (20, 10, 2)


def test_mcmc_output_xarray_dataset_to_pandas_data_frame_with_limit_from_end_on_explicit_dataset():
    dataset = get_toy_dataset()

    data_frame = mcmc_output_xarray_dataset_to_pandas_data_frame(dataset, limit_from_end=35)

    assert set(np.unique(data_frame.index.get_level_values('iteration').tolist())) == {18, 19}


def test_mcmc_output_xarray_dataset_to_pandas_data_frame_on_sliced_dataset():
    dataset = get_toy_dataset()

    sliced_dataset = dataset.sel({'iteration': slice(5, 15)})
    data_frame = mcmc_output_xarray_dataset_to_pandas_data_frame(sliced_dataset, limit_from_end=35)

    assert set(np.unique(data_frame.index.get_level_values('iteration').tolist())) == {14, 15}


def test_mcmc_output_xarray_dataset_to_pandas_data_frame_on_split_dataset():
    dataset = get_toy_dataset()

    split_dataset = dataset.sel({'iteration': [5, 15]})
    data_frame = mcmc_output_xarray_dataset_to_pandas_data_frame(split_dataset, limit_from_end=35)

    assert set(np.unique(data_frame.index.get_level_values('iteration').tolist())) == {5, 15}


def test_extract_windowed_median_log_likelihood_series():
    dataset = get_toy_dataset()

    log_likelihood_series = extract_windowed_median_log_likelihood_series(dataset, window_size=5)

    assert np.allclose(log_likelihood_series.index, [0, 5, 10, 15])
    assert np.allclose(log_likelihood_series.values, [49.5, 149.5, 249.5, 349.5])
