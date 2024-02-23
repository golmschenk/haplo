import torch
from torch import Tensor
from torch.linalg import vector_norm
from torch.nn import Module

from haplo.nicer_transform import PrecomputedUnnormalizePhaseAmplitudes


class PlusOneChiSquaredStatisticMetric(Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        unnormalize_phase_amplitudes = PrecomputedUnnormalizePhaseAmplitudes()
        observed = unnormalize_phase_amplitudes(output.type(torch.float64)) + 1.0
        expected = unnormalize_phase_amplitudes(target.type(torch.float64)) + 1.0
        chi_squared_statistic_f64 = torch.mean(torch.sum(((observed - expected) ** 2) / expected, dim=1))
        chi_squared_statistic = chi_squared_statistic_f64.type(torch.float32)
        return chi_squared_statistic


class PlusOneBeforeUnnormalizationChiSquaredStatisticMetric(Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        unnormalize_phase_amplitudes = PrecomputedUnnormalizePhaseAmplitudes()
        observed = unnormalize_phase_amplitudes(output.type(torch.float64) + 1.0)
        expected = unnormalize_phase_amplitudes(target.type(torch.float64) + 1.0)
        chi_squared_statistic_f64 = torch.mean(torch.sum(((observed - expected) ** 2) / expected, dim=1))
        chi_squared_statistic = chi_squared_statistic_f64.type(torch.float32)
        return chi_squared_statistic


class SumDifferenceSquaredOverMedianExpectedSquaredMetric(Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        epsilon = 1e-10
        unnormalize_phase_amplitudes = PrecomputedUnnormalizePhaseAmplitudes()
        observed = unnormalize_phase_amplitudes(output.type(torch.float64)) + 1.0
        expected = unnormalize_phase_amplitudes(target.type(torch.float64)) + 1.0
        numerator = torch.sum(((observed - expected) ** 2), dim=1)
        median = torch.median(expected, dim=1).values
        denominator = median ** 2
        quality_indicator = numerator / denominator
        metric_f64 = torch.mean(torch.log10(quality_indicator + epsilon))
        metric = metric_f64.type(torch.float32)
        return metric

def norm_based_gradient_clip(gradient_tensor: Tensor) -> Tensor:
    return gradient_tensor / torch.maximum(torch.linalg.vector_norm(gradient_tensor), torch.tensor(1.0))
