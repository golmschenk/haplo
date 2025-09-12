from __future__ import annotations

from typing import Union, Self

import numpy.typing as npt
from torch.nn import Module
from torch import Tensor
import torch

from haplo.nicer_transform import phase_amplitude_mean, phase_amplitude_standard_deviation, \
    parameter_standard_deviations, parameter_means


class AffineTransform(Module):
    def __init__(self, scale: Tensor, translation: Tensor):
        super().__init__()
        self.translation: Tensor = translation
        self.scale: Tensor = scale

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
