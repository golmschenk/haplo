from pathlib import Path

import torch
from torch.nn import Module

from haplo.models import LiraTraditionalShape8xWidthWithNoDoNoBnOldFirstLayers


class WrappedModel(Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


def export_onnx(model: Module):
    model.eval()
    fake_input = torch.randn(1, 11, requires_grad=True)
    _ = model(fake_input)  # Model must be run to trace.
    torch.onnx.export(model,
                      fake_input,
                      'exported_model.onnx',
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'output': {0: 'batch_size'}},
                      )


def export_onnx_model_from_pytorch_path(pytorch_path: Path):
    model = LiraTraditionalShape8xWidthWithNoDoNoBnOldFirstLayers()
    model = WrappedModel(model)
    model.load_state_dict(torch.load(pytorch_path, map_location='cpu'))
    export_onnx(model)


if __name__ == '__main__':
    export_onnx_model_from_pytorch_path(Path('sessions/1.pt'))
