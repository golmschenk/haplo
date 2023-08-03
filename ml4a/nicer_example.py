from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, List
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from bokeh.io import show
from bokeh.models import Column
from bokeh.plotting import figure as Figure

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


class NicerExample:
    def __init__(self):
        self.parameters: Optional[np.ndarray] = None
        self.phase_amplitudes: Optional[np.ndarray] = None
        self.likelihood: Optional[float] = None

    @classmethod
    def new(cls, parameters: np.ndarray, phase_amplitudes: np.ndarray, likelihood: float) -> NicerExample:
        example = cls()
        example.parameters = parameters
        example.phase_amplitudes = phase_amplitudes
        example.likelihood = likelihood
        return example

    @classmethod
    def list_from_constantinos_kalapotharakos_file(cls, dataset_path: Path, limit: Optional[int] = None
                                                   ) -> List[NicerExample]:
        examples: List[cls] = []
        with dataset_path.open() as dataset_file:
            value_iterator = re.finditer(r"[^\s]+", dataset_file.read())
            while True:
                parameters = []
                try:
                    parameters.append(float(next(value_iterator).group(0)))
                except StopIteration:
                    break
                for _ in range(10):
                    parameters.append(float(next(value_iterator).group(0)))
                likelihood = float(next(value_iterator).group(0))
                phase_amplitudes = []
                for _ in range(64):
                    phase_amplitudes.append(float(next(value_iterator).group(0)))
                examples.append(cls.new(np.array(parameters), np.array(phase_amplitudes), likelihood))
                if limit is not None and len(examples) >= limit:
                    break
        return examples

    def show(self):
        parameters_figure = Figure()
        parameters_figure.line(x=range(len(self.parameters)), y=self.parameters, line_width=2)
        phase_amplitudes_figure = Figure()
        phase_amplitudes_figure.line(x=range(len(self.phase_amplitudes)), y=self.phase_amplitudes, line_width=2)
        column = Column(parameters_figure, phase_amplitudes_figure)
        show(column)

    @staticmethod
    def show_phase_amplitude_distribution(examples: List[NicerExample]):
        phase_amplitudes = np.array([], dtype=np.float64)
        for example in examples:
            phase_amplitudes = np.append(phase_amplitudes, example.phase_amplitudes)
        histogram, edges = np.histogram(phase_amplitudes, bins=50)
        figure = Figure()
        figure.quad(top=histogram, bottom=0, left=edges[:-1], right=edges[1:],
                    fill_color="navy", line_color="white", alpha=0.5)
        show(figure)

    @staticmethod
    def show_parameter_distribution(examples: List[NicerExample]):
        parameters = NicerExample.extract_parameters_array(examples)
        figure, axes = plt.subplots(dpi=400)
        axes = sns.violinplot(ax=axes, data=parameters)
        plt.show()

    @staticmethod
    def to_tensorflow_dataset(examples: List[NicerExample], parameters_labels: bool = True,
                              normalize_parameters_and_phase_amplitudes: bool = False) -> tf.data.Dataset:
        parameters = NicerExample.extract_parameters_array(examples)
        phase_amplitudes = NicerExample.extract_phase_amplitudes_array(examples)
        if normalize_parameters_and_phase_amplitudes:
            parameters = NicerExample.normalize_parameters(parameters)
            phase_amplitudes = NicerExample.normalize_phase_amplitudes(phase_amplitudes)
        if parameters_labels:
            dataset = tf.data.Dataset.from_tensor_slices((parameters, phase_amplitudes))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((phase_amplitudes, parameters))
        return dataset

    @staticmethod
    def normalize_phase_amplitudes(phase_amplitudes):
        phase_amplitudes -= phase_amplitude_mean
        phase_amplitudes /= phase_amplitude_standard_deviation
        return phase_amplitudes

    @staticmethod
    def normalize_parameters(parameters):
        parameters -= parameter_means
        parameters /= parameter_standard_deviations
        return parameters

    @staticmethod
    def unnormalize_phase_amplitudes(phase_amplitudes):
        phase_amplitudes *= phase_amplitude_standard_deviation
        phase_amplitudes += phase_amplitude_mean
        return phase_amplitudes

    @staticmethod
    def unnormalize_parameters(parameters):
        parameters *= parameter_standard_deviations
        parameters += parameter_means
        return parameters

    @staticmethod
    def extract_phase_amplitudes_array(examples):
        phase_amplitudes_list = []
        for example in examples:
            phase_amplitudes_list.append(example.phase_amplitudes)
        phase_amplitudes = np.stack(phase_amplitudes_list, axis=0)
        return phase_amplitudes

    @staticmethod
    def extract_parameters_array(examples):
        parameters_list = []
        for example in examples:
            parameters_list.append(example.parameters)
        parameters = np.stack(parameters_list, axis=0)
        return parameters

    @classmethod
    def to_prepared_tensorflow_dataset(cls, examples: List[NicerExample], batch_size: int = 2000,
                                       shuffle: bool = False, parameters_labels: bool = True,
                                       normalize_parameters_and_phase_amplitudes: bool = False) -> tf.data.Dataset:
        dataset = cls.to_tensorflow_dataset(
            examples, parameters_labels=parameters_labels,
            normalize_parameters_and_phase_amplitudes=normalize_parameters_and_phase_amplitudes)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(10)
        return dataset
