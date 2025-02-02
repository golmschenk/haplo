import numpy as np
import shutil
import xarray

from pathlib import Path

from haplo.internal.data_conversion import constantinos_kalapotharakos_format_file_to_xarray_zarr, \
    DatasetVariableName, convert_directory_xarray_zarr_to_zip_xarray_zarr


def test_constantinos_kalapotharakos_format_file_to_xarray_zarr():
    input_path = Path(__file__).parent.joinpath(
        'test_data_conversion_resources/test_constantinos_kalapotharakos_format_file_to_xarray_zarr_input.dat')
    output_path = Path(__file__).parent.joinpath(
        'test_data_conversion_resources/test_constantinos_kalapotharakos_format_file_to_xarray_zarr_output.zarr')
    if output_path.exists():
        shutil.rmtree(output_path)
    constantinos_kalapotharakos_format_file_to_xarray_zarr(
        input_path=input_path,
        output_path=output_path,
        input_size=11,
        output_size=64,
        zarr_chunk_axis0_size=10,
    )
    xarray_dataset = xarray.open_zarr(output_path)
    assert xarray_dataset['input'].encoding['chunks'] == (10, 11)
    assert xarray_dataset['output'].encoding['chunks'] == (10, 64)
    expected_input2 = np.array(
        [-7.114849265088502E-002, -0.446535380769292, 0.273304723543533, 2.73786426394815, 3.24070261148579,
         0.292888465521632, -0.394671280424604, 0.359151156632157, 0.817593568733874, 5.08583282945785,
         1.18473436785974])
    expected_output2 = np.array(
        [40373.3744333786, 39596.4017232499, 39098.9642740749, 38916.4008620452, 39134.5362310010, 39569.0266077526,
         40254.4902517607, 41161.1573601567, 42168.3831829546, 43252.3984455773, 44283.8816825797, 45141.5396138537,
         45859.1240096732, 46304.2084888225, 46392.5521193707, 46131.1855299938, 45496.4464447583, 44541.5343021029,
         43221.8847822845, 41635.6586946216, 39768.9268424975, 37729.5876527838, 35654.2385527695, 33518.5685791590,
         31453.3284321734, 29544.2318380988, 27856.3837351974, 26496.3931805888, 25450.2960853550, 24821.9939584110,
         24617.7568518408, 24747.0687201377, 25242.9928347868, 26048.9722282203, 27124.0954536980, 28380.6333676351,
         29838.6117265666, 31362.3497959607, 32978.9479803886, 34648.3100655514, 36353.5750394435, 38038.7256478592,
         39668.5374577450, 41311.9151003579, 42889.7708873659, 44339.2376330008, 45665.7429913092, 46846.0588364893,
         47869.8583396867, 48724.6179907228, 49328.0070815555, 49779.1333959773, 49999.3834517703, 50008.0402032060,
         49805.3777445359, 49415.8166581612, 48833.9575561057, 48086.5545809052, 47226.8313410686, 46193.7888312072,
         45037.9321757776, 43819.8549896964, 42542.3644125813, 41383.5023840041])
    assert np.allclose(xarray_dataset[DatasetVariableName.INPUT][2], expected_input2)
    assert np.allclose(xarray_dataset[DatasetVariableName.OUTPUT][2], expected_output2)
    shutil.rmtree(output_path)


def test_convert_directory_xarray_zarr_to_zip_xarray_zarr():
    input_path = Path(__file__).parent.joinpath(
        'test_data_conversion_resources/test_convert_directory_xarray_zarr_to_zip_xarray_zarr_input.zarr')
    output_path = Path(__file__).parent.joinpath(
        'test_data_conversion_resources/test_convert_directory_xarray_zarr_to_zip_xarray_zarr_output.zip')
    output_path.unlink(missing_ok=True)
    convert_directory_xarray_zarr_to_zip_xarray_zarr(
        input_path=input_path,
        output_path=output_path,
    )
    xarray_dataset = xarray.open_zarr(output_path)
    assert xarray_dataset['input'].encoding['chunks'] == (10, 11)
    assert xarray_dataset['output'].encoding['chunks'] == (10, 64)
    expected_input2 = np.array(
        [-7.114849265088502E-002, -0.446535380769292, 0.273304723543533, 2.73786426394815, 3.24070261148579,
         0.292888465521632, -0.394671280424604, 0.359151156632157, 0.817593568733874, 5.08583282945785,
         1.18473436785974])
    expected_output2 = np.array(
        [40373.3744333786, 39596.4017232499, 39098.9642740749, 38916.4008620452, 39134.5362310010, 39569.0266077526,
         40254.4902517607, 41161.1573601567, 42168.3831829546, 43252.3984455773, 44283.8816825797, 45141.5396138537,
         45859.1240096732, 46304.2084888225, 46392.5521193707, 46131.1855299938, 45496.4464447583, 44541.5343021029,
         43221.8847822845, 41635.6586946216, 39768.9268424975, 37729.5876527838, 35654.2385527695, 33518.5685791590,
         31453.3284321734, 29544.2318380988, 27856.3837351974, 26496.3931805888, 25450.2960853550, 24821.9939584110,
         24617.7568518408, 24747.0687201377, 25242.9928347868, 26048.9722282203, 27124.0954536980, 28380.6333676351,
         29838.6117265666, 31362.3497959607, 32978.9479803886, 34648.3100655514, 36353.5750394435, 38038.7256478592,
         39668.5374577450, 41311.9151003579, 42889.7708873659, 44339.2376330008, 45665.7429913092, 46846.0588364893,
         47869.8583396867, 48724.6179907228, 49328.0070815555, 49779.1333959773, 49999.3834517703, 50008.0402032060,
         49805.3777445359, 49415.8166581612, 48833.9575561057, 48086.5545809052, 47226.8313410686, 46193.7888312072,
         45037.9321757776, 43819.8549896964, 42542.3644125813, 41383.5023840041])
    assert np.allclose(xarray_dataset[DatasetVariableName.INPUT][2], expected_input2)
    assert np.allclose(xarray_dataset[DatasetVariableName.OUTPUT][2], expected_output2)
    try:
        output_path.unlink()
    except PermissionError:  # In the Windows tests, this can sporadically fail. It's fine just to let it slide.
        pass
