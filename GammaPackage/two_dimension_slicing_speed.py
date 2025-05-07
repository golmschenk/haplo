import numpy as np
import torch
from bokeh.io import show
from bokeh.plotting import figure
from pathlib import Path

from GammaPackage.GammaSkymapDataset import GammaSkymapDataset
from GammaPackage.cura_2D_model import Cura2D
from haplo.export_onnx import WrappedModel
from haplo.nicer_dataset import split_dataset_into_count_datasets
from haplo.unwrap_model import unwrap_model


def main():
    dataset_path = Path('Gamma_halfplane_v3.csv')
    dataset = GammaSkymapDataset(dataset_path, randomDataAugmentation=False, loadLimit=500, normalize=False)
    test_dataset, validation_dataset, train_dataset, _ = split_dataset_into_count_datasets(
        dataset, [200, 1, 1])
    
    full_model = WrappedModel(Cura2D(input_features=11))
    saved_model_path = Path('lowest_validation_model.pt')
    full_model.load_state_dict(torch.load(str(saved_model_path), map_location=torch.device('cpu')))

    efficient_model = WrappedModel(EfficientCura2D(input_features=11))
    saved_model_path = Path('lowest_validation_model.pt')
    efficient_model.load_state_dict(torch.load(str(saved_model_path), map_location=torch.device('cpu')))
    
    test_parameters0, test_phase_amplitudes0 = test_dataset[0]
    input_array = np.expand_dims(test_parameters0, axis=0)
    with torch.no_grad():
        input_tensor = torch.tensor(input_array)
        output_tensor = full_model(input_tensor)
        output_array = output_tensor.numpy()
    predicted_test_phase_amplitudes0 = np.squeeze(output_array, axis=0)

    light_curve_comparison = figure()
    light_curve_comparison.line(x=list(range(100)), y=test_phase_amplitudes0[30].numpy(), color='mediumblue')
    light_curve_comparison.line(x=list(range(100)), y=predicted_test_phase_amplitudes0[30], color='firebrick')
    show(light_curve_comparison)
    pass


if __name__ == '__main__':
    main()
