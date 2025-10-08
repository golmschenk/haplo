from __future__ import annotations

from typing import Union, Self

import numpy as np
import numpy.typing as npt
from torch.nn import Module
from torch import Tensor
import torch


phase_amplitude_mean = 34025.080543335825
phase_amplitude_standard_deviation = 47698.66676993027
parameter_means = np.array(
    [-0.0008009571736463096, -0.0008946310379428422, -2.274708783534052e-05, 1.5716876559520705,
     3.1388159291733086, -0.001410436081400537, -0.0001470613574040905, -3.793528434430451e-05,
     1.5723036365564083, 3.1463088925150258, 5.509554132916939])
parameter_standard_deviations = np.array(
    [0.28133126679885656, 0.28100480365686287, 0.28140136435474244, 0.907001394792043, 1.811683338833852,
     0.2815981892528909, 0.281641754864262, 0.28109705707606697, 0.9062620846468298, 1.8139690831565327,
     2.886950440590801])


class AffineTransform(Module):
    def __init__(self, scale: Tensor, translation: Tensor):
        super().__init__()
        self.register_buffer('translation', translation)
        self.translation: Tensor = self.translation  # Static analysis workaround for PyTorch register_buffer.
        self.register_buffer('scale', scale)
        self.scale: Tensor = self.scale  # Static analysis workaround for PyTorch register_buffer.

    def forward(self, x: Tensor) -> Tensor:
        x = x * self.scale
        x = x + self.translation
        return x

    @classmethod
    def new(cls, scale: Union[Tensor, npt.NDArray, float], translation: Union[Tensor, npt.NDArray, float]) -> Self:
        if not isinstance(translation, Tensor):
            translation = torch.tensor(translation, dtype=torch.float32)
        if not isinstance(scale, Tensor):
            scale = torch.tensor(scale, dtype=torch.float32)
        return cls(scale=scale, translation=translation)


default_reverse_output_affine_transform = AffineTransform.new(
    scale=1 / phase_amplitude_standard_deviation,
    translation=-phase_amplitude_mean / phase_amplitude_standard_deviation
)
default_output_affine_transform = AffineTransform.new(
    scale=phase_amplitude_standard_deviation,
    translation=phase_amplitude_mean
)
default_input_affine_transform = AffineTransform.new(
    scale=1 / parameter_standard_deviations,
    translation=-parameter_means / parameter_standard_deviations
)
default_reverse_input_affine_transform = AffineTransform.new(
    scale=parameter_standard_deviations,
    translation=parameter_means
)
