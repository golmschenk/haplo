from pathlib import Path

import onnx
import torch
from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed
from torch.nn import Module

from haplo.models import Cura


class WrappedModel(Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


def export_onnx(model: Module, output_onnx_model_path: Path):
    model.eval()
    fake_input = torch.randn(1, 11, requires_grad=True)
    _ = model(fake_input)  # Model must be run to trace.
    torch.onnx.export(model,
                      fake_input,
                      str(output_onnx_model_path),
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'output': {0: 'batch_size', 1: 'phase_amplitudes_length'}},
                      )
    onnx_model = onnx.load(str(output_onnx_model_path))
    make_dim_param_fixed(onnx_model.graph, 'batch_size', 1)
    make_dim_param_fixed(onnx_model.graph, 'phase_amplitudes_length', 64)
    onnx.save(onnx_model, str(output_onnx_model_path))


def export_onnx_model_from_pytorch_path(pytorch_path: Path):
    model = Cura()
    model = WrappedModel(model)
    model.load_state_dict(torch.load(pytorch_path, map_location='cpu'))
    export_onnx(model, output_onnx_model_path=Path('exported_model.onnx'))


if __name__ == '__main__':
    export_onnx_model_from_pytorch_path(Path('sessions/1.pt'))
