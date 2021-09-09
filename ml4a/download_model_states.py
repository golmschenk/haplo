import warnings
from pathlib import Path

import dvc.api

from ml4a.paths import user_data_directory


def download_model_states():
    model_state_file_paths = [
        Path('model_states/infer_from_phase_amplitudes_to_parameters_model_state/model.ckpt.data-00000-of-00001'),
        Path('model_states/infer_from_phase_amplitudes_to_parameters_model_state/model.ckpt.index'),
        Path('model_states/infer_from_parameters_to_phase_amplitudes_model_state/model.ckpt.data-00000-of-00001'),
        Path('model_states/infer_from_parameters_to_phase_amplitudes_model_state/model.ckpt.index'),
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        for model_state_file_path in model_state_file_paths:
            user_data_model_path = user_data_directory.joinpath(model_state_file_path)
            with dvc.api.open(model_state_file_path, repo='https://github.com/golmschenk/ml4a', mode='rb'
                              ) as dvc_file_handle:
                file_content = dvc_file_handle.read()
            user_data_model_path.parent.mkdir(exist_ok=True, parents=True)
            with user_data_model_path.open('wb') as local_file_handle:
                local_file_handle.write(file_content)


if __name__ == '__main__':
    download_model_states()
