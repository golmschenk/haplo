import datetime
import numpy as np
import torch
from pathlib import Path
from torch import tensor

from GammaPackage.cura_2D_model import Cura2D
from GammaPackage.efficient_cura_2D_model import SelectiveThetaComputeCura2D
from haplo.export_onnx import WrappedModel


def main():
    full_model = WrappedModel(Cura2D(input_features=11))
    saved_model_path = Path('lowest_validation_model.pt')
    full_model.load_state_dict(torch.load(str(saved_model_path), map_location=torch.device('cpu')))

    efficient_model = WrappedModel(SelectiveThetaComputeCura2D(input_features=11))
    saved_model_path = Path('lowest_validation_model.pt')
    efficient_model.load_state_dict(torch.load(str(saved_model_path), map_location=torch.device('cpu')))

    random_generator = np.random.default_rng(seed=0)
    for _ in range(100):
        parameters = random_generator.random(11, dtype=np.float32)
        input_array = np.expand_dims(parameters, axis=0)
        theta_bin = random_generator.integers(0, 64)
        theta_bin = 4
        with torch.no_grad():
            input_tensor = torch.tensor(input_array)
            full_model_output_tensor = full_model(input_tensor)
            efficient_model_output_tensor = efficient_model(input_tensor, tensor(theta_bin))
            full_model_bin_tensor = full_model_output_tensor[:, theta_bin]
            full_model_bin_array = full_model_bin_tensor.numpy()
            efficient_model_output_array = efficient_model_output_tensor.numpy()
            assert np.allclose(full_model_bin_array, efficient_model_output_array, rtol=0.02)


if __name__ == '__main__':
    main()
