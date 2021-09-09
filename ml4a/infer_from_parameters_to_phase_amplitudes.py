import os
import argparse
from pathlib import Path
import numpy as np

from ml4a.infer import load_trained_parameters_to_phase_amplitudes_model, split_array_into_chunks
from ml4a.nicer_example import NicerExample


def infer_from_parameters_to_phase_amplitudes(input_csv_path: Path, output_csv_path: Path) -> None:
    parameters = np.loadtxt(str(input_csv_path))
    model = load_trained_parameters_to_phase_amplitudes_model()
    normalized_parameters = NicerExample.normalize_parameters(parameters)
    normalized_parameter_chunks = []
    for normalized_parameters_chunk in split_array_into_chunks(normalized_parameters, chunk_size=1000):
        normalized_phase_amplitudes_chunk = model.call(normalized_parameters_chunk, training=False)
        normalized_parameter_chunks.append(normalized_phase_amplitudes_chunk)
    normalized_phase_amplitudes = np.concatenate(normalized_parameter_chunks, axis=0)
    phase_amplitudes = NicerExample.unnormalize_phase_amplitudes(normalized_phase_amplitudes)
    np.savetxt(output_csv_path, phase_amplitudes)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    parser = argparse.ArgumentParser(description='Infer phase amplitudes from parameters.')
    parser.add_argument('input_csv_path', type=Path, help='The path to the input parameters CSV.')
    parser.add_argument('output_csv_path', type=Path, help='The path to output phase amplitudes CSV to.')
    arguments = parser.parse_args()
    infer_from_parameters_to_phase_amplitudes(arguments.input_csv_path, arguments.output_csv_path)
