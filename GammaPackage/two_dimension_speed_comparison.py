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
    start_time = datetime.datetime.now()
    for _ in range(100):
        parameters = random_generator.random(11, dtype=np.float32)
        input_array = np.expand_dims(parameters, axis=0)
        with torch.no_grad():
            input_tensor = torch.tensor(input_array)
            model_output_tensor = full_model(input_tensor)
            model_output_array = model_output_tensor.numpy()
            _ = np.squeeze(model_output_array, axis=0)
    end_time = datetime.datetime.now()
    print(f'{end_time - start_time}')

    theta_bin = 4
    random_generator = np.random.default_rng(seed=0)
    start_time = datetime.datetime.now()
    for _ in range(100):
        parameters = random_generator.random(11, dtype=np.float32)
        input_array = np.expand_dims(parameters, axis=0)
        with torch.no_grad():
            input_tensor = torch.tensor(input_array)
            model_output_tensor = efficient_model(input_tensor, tensor(theta_bin))
            model_output_array = model_output_tensor.numpy()
            _ = np.squeeze(model_output_array, axis=0)
    end_time = datetime.datetime.now()
    print(f'{end_time - start_time}')


if __name__ == '__main__':
    main()
