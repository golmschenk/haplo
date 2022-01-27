from pathlib import Path

import numpy as np
from tensorflow.keras.models import Model

from ml4a.nicer_phase_amplitudes_to_parameters_models import Mira
from ml4a.paths import user_data_directory
from ml4a.residual_model import ResModel1InitialDenseNoDoConvEndDoublingWidererL2


def load_trained_phase_amplitudes_to_parameters_model() -> Model:
    model = Mira()
    model_path_string = 'model_states/infer_from_phase_amplitudes_to_parameters_model_state/model.ckpt'
    if Path(model_path_string).exists():
        model_path = Path(model_path_string)
    else:
        model_path = user_data_directory.joinpath(model_path_string)
    model.load_weights(
        model_path
    ).expect_partial()
    return model


def load_trained_parameters_to_phase_amplitudes_model() -> Model:
    model = ResModel1InitialDenseNoDoConvEndDoublingWidererL2()
    model_path_string='model_states/infer_from_parameters_to_phase_amplitudes_model_state/model.ckpt'
    if Path(model_path_string).exists():
        model_path = Path(model_path_string)
    else:
        model_path = user_data_directory.joinpath(model_path_string)
    model.load_weights(
        model_path
    ).expect_partial()
    return model


def split_array_into_chunks(array: np.ndarray, chunk_size: int):
    return np.split(array, np.arange(chunk_size, len(array), chunk_size))
