from pathlib import Path

import numpy as np
import torch
from bokeh.io import show
from bokeh.plotting import figure

from haplo.export_onnx import WrappedModel
from haplo.models import Cura
from haplo.nicer_dataset import NicerDataset, split_dataset_into_count_datasets
from haplo.nicer_transform import PrecomputedNormalizeParameters, PrecomputedNormalizePhaseAmplitudes


def example_infer_session():
    full_dataset_path = Path('data/800k_parameters_and_phase_amplitudes.db')
    full_train_dataset = NicerDataset.new(
        dataset_path=full_dataset_path,
        parameters_transform=PrecomputedNormalizeParameters(),
        phase_amplitudes_transform=PrecomputedNormalizePhaseAmplitudes())
    test_dataset, validation_dataset, train_dataset, _ = split_dataset_into_count_datasets(
        full_train_dataset, [100_000, 100_000, 500_000])

    model = Cura()
    model = WrappedModel(model)  # The DDP module requires an extra wrapping. This emulates that.
    saved_model_path = Path('sessions/dp9gxdz3_latest_model.pt')
    model.load_state_dict(torch.load(str(saved_model_path), map_location=torch.device('cpu')))
    model.eval()

    test_parameters0, test_phase_amplitudes0 = test_dataset[0]
    input_array = np.expand_dims(test_parameters0, axis=0)
    with torch.no_grad():
        input_tensor = torch.tensor(input_array)
        output_tensor = model(input_tensor)
        output_array = output_tensor.numpy()
    predicted_test_phase_amplitudes0 = np.squeeze(output_array, axis=0)

    comparison_figure = figure(x_axis_label='phase', y_axis_label='amplitude')
    comparison_figure.line(x=list(range(64)), y=test_phase_amplitudes0, line_color='mediumblue')
    comparison_figure.line(x=list(range(64)), y=predicted_test_phase_amplitudes0, line_color='firebrick')
    show(comparison_figure)


if __name__ == '__main__':
    example_infer_session()
