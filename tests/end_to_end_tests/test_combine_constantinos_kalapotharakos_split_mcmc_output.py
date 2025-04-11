import pytest
import xarray
from pathlib import Path

import shutil

from haplo.internal.combine_split_mcmc_output_files import \
    combine_constantinos_kalapotharakos_split_mcmc_output_files_to_xarray_zarr


def test_combine_constantinos_kalapotharakos_split_mcmc_output():
    root_split_files_directory = Path(__file__).parent.joinpath(
        'combine_constantinos_kalapotharakos_split_mcmc_output_resources')
    output_path = root_split_files_directory.joinpath('test_combine_constantinos_kalapotharakos_split_mcmc_output.zarr')
    if output_path.exists():
        shutil.rmtree(output_path)
    combine_constantinos_kalapotharakos_split_mcmc_output_files_to_xarray_zarr(root_split_files_directory, output_path,
                                                                               elements_per_record=13, overwrite=True)
    xarray_dataset = xarray.open_zarr(output_path)
    assert xarray_dataset['parameter'].shape == (3, 4, 2, 11)
    assert xarray_dataset['iteration'].max() == 2
    assert xarray_dataset['parameter'][1, 1, 0, 8].compute().item() == pytest.approx(2.06011819056089)
    assert xarray_dataset['log_likelihood'][2, 3, 1].compute().item() == pytest.approx(-24990.2909981251)
    assert xarray_dataset['parameter'].encoding['chunks'] == (1000, 4, 2, 11)
    if output_path.exists():
        shutil.rmtree(output_path)


def test_combine_constantinos_kalapotharakos_split_mcmc_output_with_complete_final_iteration():
    root_split_files_directory = Path(__file__).parent.joinpath(
        'combine_constantinos_kalapotharakos_split_mcmc_output_resources_with_complete_final_iteration')
    output_path = root_split_files_directory.joinpath('test_combine_constantinos_kalapotharakos_split'
                                                      '_mcmc_output_with_complete_final_iteration.zarr')
    if output_path.exists():
        shutil.rmtree(output_path)
    combine_constantinos_kalapotharakos_split_mcmc_output_files_to_xarray_zarr(root_split_files_directory, output_path,
                                                                               elements_per_record=13, overwrite=True)
    xarray_dataset = xarray.open_zarr(output_path)
    assert xarray_dataset['parameter'].shape == (4, 4, 2, 11)
    assert xarray_dataset['iteration'].max() == 3
    assert xarray_dataset['parameter'][1, 1, 0, 8].compute().item() == pytest.approx(2.06011819056089)
    assert xarray_dataset['log_likelihood'][2, 3, 1].compute().item() == pytest.approx(-24990.2909981251)
    assert xarray_dataset['parameter'].encoding['chunks'] == (1000, 4, 2, 11)
    if output_path.exists():
        shutil.rmtree(output_path)


def test_combine_constantinos_kalapotharakos_split_mcmc_output_with_incomplete_final_iteration():
    root_split_files_directory = Path(__file__).parent.joinpath(
        'combine_constantinos_kalapotharakos_split_mcmc_output_resources_with_incomplete_final_iteration')
    output_path = root_split_files_directory.joinpath('test_combine_constantinos_kalapotharakos_split'
                                                      '_mcmc_output_with_incomplete_final_iteration.zarr')
    if output_path.exists():
        shutil.rmtree(output_path)
    combine_constantinos_kalapotharakos_split_mcmc_output_files_to_xarray_zarr(root_split_files_directory, output_path,
                                                                               elements_per_record=13, overwrite=True)
    xarray_dataset = xarray.open_zarr(output_path)
    assert xarray_dataset['parameter'].shape == (3, 4, 2, 11)
    assert xarray_dataset['iteration'].max() == 2
    assert xarray_dataset['parameter'][1, 1, 0, 8].compute().item() == pytest.approx(2.06011819056089)
    assert xarray_dataset['log_likelihood'][2, 3, 1].compute().item() == pytest.approx(-24990.2909981251)
    assert xarray_dataset['parameter'].encoding['chunks'] == (1000, 4, 2, 11)
    if output_path.exists():
        shutil.rmtree(output_path)


def test_combine_constantinos_kalapotharakos_split_mcmc_output_with_multiprocessing():
    root_split_files_directory = Path(__file__).parent.joinpath(
        'combine_constantinos_kalapotharakos_split_mcmc_output_resources')
    output_path = root_split_files_directory.joinpath(f'test_combine_constantinos_kalapotharakos'
                                                      f'_split_mcmc_output_with_multiprocessing.zarr')
    if output_path.exists():
        shutil.rmtree(output_path)
    combine_constantinos_kalapotharakos_split_mcmc_output_files_to_xarray_zarr(root_split_files_directory, output_path,
                                                                               elements_per_record=13, overwrite=True,
                                                                               multiprocess_pool_size=4)
    xarray_dataset = xarray.open_zarr(output_path)
    assert xarray_dataset['parameter'].shape == (3, 4, 2, 11)
    assert xarray_dataset['iteration'].max() == 2
    assert xarray_dataset['parameter'][1, 1, 0, 8].compute().item() == pytest.approx(2.06011819056089)
    assert xarray_dataset['log_likelihood'][2, 3, 1].compute().item() == pytest.approx(-24990.2909981251)
    assert xarray_dataset['parameter'].encoding['chunks'] == (1000, 4, 2, 11)
    if output_path.exists():
        shutil.rmtree(output_path)


def test_combine_constantinos_kalapotharakos_split_mcmc_output_with_zip():
    root_split_files_directory = Path(__file__).parent.joinpath(
        'combine_constantinos_kalapotharakos_split_mcmc_output_resources_with_zip')
    output_path = root_split_files_directory.joinpath(
        'test_combine_constantinos_kalapotharakos_split_mcmc_output.zarr.zip')
    if output_path.exists():
        output_path.unlink()
    combine_constantinos_kalapotharakos_split_mcmc_output_files_to_xarray_zarr(root_split_files_directory, output_path,
                                                                               elements_per_record=13, overwrite=True)
    xarray_dataset = xarray.open_zarr(output_path)
    assert xarray_dataset['parameter'].shape == (3, 4, 2, 11)
    assert xarray_dataset['iteration'].max() == 2
    assert xarray_dataset['parameter'][1, 1, 0, 8].compute().item() == pytest.approx(2.06011819056089)
    assert xarray_dataset['log_likelihood'][2, 3, 1].compute().item() == pytest.approx(-24990.2909981251)
    assert xarray_dataset['parameter'].encoding['chunks'] == (1000, 4, 2, 11)
    try:
        output_path.unlink()
    except PermissionError:  # In the Windows tests, this can sporadically fail. It's fine just to let it slide.
        pass
