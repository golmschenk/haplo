import warnings
from pathlib import Path
from typing import Optional

import dvc.api

from ml4a.paths import user_data_directory


def download_model_states(directory: Optional[Path] = None):
    if directory is None:
        directory = user_data_directory
    model_state_file_paths = [
        Path('model_states/infer_from_phase_amplitudes_to_parameters_model_state/model.ckpt.data-00000-of-00001'),
        Path('model_states/infer_from_phase_amplitudes_to_parameters_model_state/model.ckpt.index'),
        Path('model_states/infer_from_parameters_to_phase_amplitudes_model_state/model.ckpt.data-00000-of-00001'),
        Path('model_states/infer_from_parameters_to_phase_amplitudes_model_state/model.ckpt.index'),
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        for model_state_file_path in model_state_file_paths:
            model_path = directory.joinpath(model_state_file_path)
            with dvc.api.open(model_state_file_path, repo='https://github.com/golmschenk/ml4a', mode='rb'
                              ) as dvc_file_handle:
                file_content = dvc_file_handle.read()
            model_path.parent.mkdir(exist_ok=True, parents=True)
            with model_path.open('wb') as local_file_handle:
                local_file_handle.write(file_content)


if __name__ == '__main__':
    download_model_states()
