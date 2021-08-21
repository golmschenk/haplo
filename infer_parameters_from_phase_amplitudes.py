from pathlib import Path
import numpy as np

from ml4a.infer import load_trained_phase_amplitudes_to_parameters_model, split_array_into_chunks
from ml4a.nicer_example import NicerExample

input_csv_path = Path('phase_amplitudes.csv')
output_csv_path = Path('parameters.csv')

phase_amplitudes = np.loadtxt(str(input_csv_path))
model = load_trained_phase_amplitudes_to_parameters_model()
normalized_phase_amplitudes = NicerExample.normalize_phase_amplitudes(phase_amplitudes)
normalized_parameter_chunks = []
for normalized_phase_amplitudes_chunk in split_array_into_chunks(normalized_phase_amplitudes, chunk_size=100):
    normalized_parameters_chunk = model(normalized_phase_amplitudes_chunk)
    normalized_parameter_chunks.append(normalized_parameters_chunk)
normalized_parameters = np.stack(normalized_parameter_chunks, axis=0)
parameters = NicerExample.unnormalize_phase_amplitudes(normalized_parameters)
np.savetxt(parameters)
