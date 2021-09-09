import os
import argparse
from pathlib import Path
import numpy as np

from ml4a.infer import load_trained_phase_amplitudes_to_parameters_model, split_array_into_chunks
from ml4a.nicer_example import NicerExample


def infer_from_phase_amplitudes_to_parameters(input_csv_path: Path, output_csv_path: Path) -> None:
    phase_amplitudes = np.loadtxt(str(input_csv_path))
    model = load_trained_phase_amplitudes_to_parameters_model()
    normalized_phase_amplitudes = NicerExample.normalize_phase_amplitudes(phase_amplitudes)
    normalized_parameter_chunks = []
    for normalized_phase_amplitudes_chunk in split_array_into_chunks(normalized_phase_amplitudes, chunk_size=100):
        normalized_parameters_chunk = model.call(normalized_phase_amplitudes_chunk, training=False)
        normalized_parameter_chunks.append(normalized_parameters_chunk)
    normalized_parameters = np.concatenate(normalized_parameter_chunks, axis=0)
    parameters = NicerExample.unnormalize_parameters(normalized_parameters)
    np.savetxt(output_csv_path, parameters)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    parser = argparse.ArgumentParser(description='Infer parameters from phase amplitudes.')
    parser.add_argument('input_csv_path', type=Path, help='The path to the input phase amplitudes CSV.')
    parser.add_argument('output_csv_path', type=Path, help='The path to output parameters CSV to.')
    arguments = parser.parse_args()
    infer_from_phase_amplitudes_to_parameters(arguments.input_csv_path, arguments.output_csv_path)
