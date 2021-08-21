from pathlib import Path

import numpy as np
from tensorflow.keras.models import Model

from ml4a.nicer_phase_amplitudes_to_parameters_models import Mira


def load_trained_phase_amplitudes_to_parameters_model() -> Model:
    model = Mira()
    model.load_weights(Path('model_states/phase_amplitudes_to_parameters.ckpt'))
    return model

def split_array_into_chunks(array: np.ndarray, chunk_size: int):
    return np.split(array, np.arange(chunk_size, len(array), chunk_size))
