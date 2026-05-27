import torch
from torch.nn import Module


class PlusOneChiSquaredStatisticMetric(Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        observed = output.type(torch.float64) + 1.0
        expected = target.type(torch.float64) + 1.0
        chi_squared_statistic_f64 = torch.mean(torch.sum(((observed - expected) ** 2) / expected, dim=1))
        chi_squared_statistic = chi_squared_statistic_f64.type(torch.float32)
        return chi_squared_statistic


class SumDifferenceSquaredOverMedianExpectedSquaredMetric(Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        epsilon = 1e-10
        observed = output.type(torch.float64) + 1.0
        expected = target.type(torch.float64) + 1.0
        numerator = torch.sum(((observed - expected) ** 2), dim=1)
        median = torch.median(expected, dim=1).values
        denominator = median ** 2
        quality_indicator = numerator / denominator
        metric_f64 = torch.mean(torch.log10(quality_indicator + epsilon))
        metric = metric_f64.type(torch.float32)
        return metric

# TODO: Merge 2D version with 1D version.
class PlusOneChiSquaredStatisticMetric2d(Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        observed = output.type(torch.float64) + 1.0
        expected = target.type(torch.float64) + 1.0
        chi_squared_statistic_f64 = torch.mean(torch.sum(((observed - expected) ** 2) / expected, dim=[1, 2]))
        chi_squared_statistic = chi_squared_statistic_f64.type(torch.float32)
        return chi_squared_statistic

# TODO: Merge 2D version with 1D version.
class MeanDifferenceSquaredOverMedianExpectedSquaredMetric2d(Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        epsilon = 1e-10
        observed = output.type(torch.float64) + 1.0
        expected = target.type(torch.float64) + 1.0
        numerator = torch.mean(((observed - expected) ** 2), dim=2)
        median = torch.median(expected, dim=2).values
        denominator = median ** 2
        quality_indicator = numerator / denominator
        metric_f64 = torch.mean(torch.log10(quality_indicator + epsilon))
        metric = metric_f64.type(torch.float32)
        return metric

class SumDifferencePowerOverMedianExpectedPowerMetric(Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        epsilon = 1e-10
        observed = output.type(torch.float64) + 1.0
        expected = target.type(torch.float64) + 1.0
        numerator = torch.sum((torch.abs(observed - expected) ** 1.1), dim=1)
        median = torch.median(expected, dim=1).values
        denominator = median ** 1.1
        quality_indicator = numerator / denominator
        metric_f64 = torch.mean(torch.log10(quality_indicator + epsilon))
        metric = metric_f64.type(torch.float32)
        return metric


class SumDifferenceSquaredOverMedianExpectedSquaredWithLogExpectedMetric(Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        epsilon = 1e-10
        observed = output.type(torch.float64) + 1.0
        expected = torch.log(target.type(torch.float64) + 1.0)
        numerator = torch.sum(((observed - expected) ** 2), dim=1)
        median = torch.median(expected, dim=1).values
        denominator = median ** 2
        quality_indicator = numerator / denominator
        metric_f64 = torch.mean(torch.log10(quality_indicator + epsilon))
        metric = metric_f64.type(torch.float32)
        return metric


class SumDifferenceSquaredOverMedianExpectedSquaredWithLogExpectedMetricNoLogInQi(Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        epsilon = 1e-10
        observed = output.type(torch.float64)
        expected = torch.log(target.type(torch.float64) + 2.0)
        numerator = torch.sum(((observed - expected) ** 2), dim=1)
        median = torch.median(expected, dim=1).values
        denominator = median ** 2
        quality_indicator = numerator / denominator
        metric_f64 = torch.mean(quality_indicator + epsilon)
        metric = metric_f64.type(torch.float32)
        return metric


class SumDifferenceSquaredOverMedianExpectedSquaredWithExpObservedMetric(Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        epsilon = 1e-10
        observed = torch.exp(output.type(torch.float64))
        expected = target.type(torch.float64) + 1.0
        numerator = torch.sum(((observed - expected) ** 2), dim=1)
        median = torch.median(expected, dim=1).values
        denominator = median ** 2
        quality_indicator = numerator / denominator
        metric_f64 = torch.mean(torch.log10(quality_indicator + epsilon))
        metric = metric_f64.type(torch.float32)
        return metric


class SumDifferenceSquaredOverMedianExpectedSquaredWithExpObservedMetricLse(Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        epsilon = 1e-10
        observed = torch.exp(output.type(torch.float64)) + 1.0
        expected = target.type(torch.float64) + 1.0
        numerator = torch.sum(((observed - expected) ** 2), dim=1)
        median = torch.median(expected, dim=1).values
        denominator = median ** 2
        quality_indicator = numerator / denominator
        metric_f64 = torch.mean(torch.log10(quality_indicator + epsilon))
        metric = metric_f64.type(torch.float32)
        return metric