import numpy as np
import os
import torch
from pathlib import Path

from haplo.export_onnx import WrappedModel
from haplo.models import SingleDenseNetwork
from haplo.nicer_dataset import NicerDataset, split_dataset_into_count_datasets
from haplo.nicer_transform import PrecomputedNormalizeParameters, PrecomputedNormalizePhaseAmplitudes


def test_infer_session():
    os.environ["WANDB_MODE"] = "disabled"
    full_dataset_path = Path(__file__).parent.joinpath('test_train_session_resources/300_parameters_and_phase_amplitudes.db')
    full_train_dataset = NicerDataset.new(
        dataset_path=full_dataset_path,
        length=300,
        parameters_transform=PrecomputedNormalizeParameters(),
        phase_amplitudes_transform=PrecomputedNormalizePhaseAmplitudes(),
        in_memory=True
    )
    test_dataset, validation_dataset, train_dataset, _ = split_dataset_into_count_datasets(
        full_train_dataset, [10, 10, 100])

    model = SingleDenseNetwork()
    model = WrappedModel(model)  # The DDP module requires an extra wrapping. This emulates that.
    saved_model_path = Path(__file__).parent.joinpath('test_infer_session_resources/test_infer_session_model.pt')
    model.load_state_dict(torch.load(str(saved_model_path), map_location=torch.device('cpu')))
    model.eval()

    test_parameters0, test_phase_amplitudes0 = test_dataset[0]
    input_array = np.expand_dims(test_parameters0, axis=0)
    with torch.no_grad():
        input_tensor = torch.tensor(input_array)
        output_tensor = model(input_tensor)
        output_array = output_tensor.numpy()
    predicted_test_phase_amplitudes0 = np.squeeze(output_array, axis=0)
    assert predicted_test_phase_amplitudes0.shape == (64,)
