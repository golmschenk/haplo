import numpy as np
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


class PrecomputedNormalizeParameters:
    def __call__(self, parameters):
        parameters -= parameter_means
        parameters /= parameter_standard_deviations
        return parameters


class PrecomputedNormalizePhaseAmplitudes:
    def __call__(self, phase_amplitudes):
        phase_amplitudes -= phase_amplitude_mean
        phase_amplitudes /= phase_amplitude_standard_deviation
        return phase_amplitudes


class PrecomputedUnnormalizeParameters:
    def __call__(self, parameters):
        parameters *= parameter_standard_deviations
        parameters += parameter_means
        return parameters


class PrecomputedUnnormalizePhaseAmplitudes:
    def __call__(self, phase_amplitudes):
        phase_amplitudes *= phase_amplitude_standard_deviation
        phase_amplitudes += phase_amplitude_mean
        return phase_amplitudes


class ToTensor:
    def __call__(self, array):
        return torch.tensor(array)