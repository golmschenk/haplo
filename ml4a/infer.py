from pathlib import Path

import numpy as np
from tensorflow.keras.models import Model

from ml4a.download_model_states import download_model_states
from ml4a.nicer_model import Nyx9Wider
from ml4a.nicer_phase_amplitudes_to_parameters_models import Mira
from ml4a.paths import user_data_directory
from ml4a.residual_model import Lira


def load_trained_phase_amplitudes_to_parameters_model() -> Model:
    model = Mira()
    model_path = user_data_directory.joinpath(
        'model_states/infer_from_phase_amplitudes_to_parameters_model_state/model.ckpt')
    if not model_path.parent.exists():
        download_model_states()
    model.load_weights(
        model_path
    ).expect_partial()
    return model


def load_trained_parameters_to_phase_amplitudes_model() -> Model:
    model = Nyx9Wider()
    model_path = Path( # user_data_directory.joinpath(
        'logs/Nyx9Widerer_no_do_l2_1000_cont2/best_validation_model.ckpt')
    if not model_path.parent.exists():
        download_model_states()
    model.load_weights(
        model_path
    ).expect_partial()
    return model


def split_array_into_chunks(array: np.ndarray, chunk_size: int):
    return np.split(array, np.arange(chunk_size, len(array), chunk_size))
