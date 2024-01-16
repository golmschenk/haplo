from pathlib import Path

import torch

from haplo.export_onnx import WrappedModel, export_onnx
from haplo.models import Cura


def example_export_to_onnx():
    pytorch_path = Path('sessions/your_model_name.pt')
    model = Cura()
    model = WrappedModel(model)
    model.load_state_dict(torch.load(pytorch_path, map_location='cpu'))
    export_onnx(model=model, output_onnx_model_path=Path('exported_model.onnx'))


if __name__ == '__main__':
    example_export_to_onnx()
