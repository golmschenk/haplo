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
from bokeh.plotting import Figure


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
    def list_from_constantinos_kalapotharakos_file(cls, dataset_path: Path) -> List[NicerExample]:
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
        parameters_list = []
        for example in examples:
            parameters_list.append(example.parameters)
        parameters = np.stack(parameters_list, axis=0)
        figure, axes = plt.subplots(dpi=400)
        axes = sns.violinplot(ax=axes, data=parameters)
        plt.show()

    @staticmethod
    def to_tensorflow_dataset(examples: List[NicerExample]) -> tf.data.Dataset:
        parameters_list = []
        for example in examples:
            parameters_list.append(example.parameters)
        parameters = np.stack(parameters_list, axis=0)
        phase_amplitudes_list = []
        for example in examples:
            phase_amplitudes_list.append(example.phase_amplitudes)
        phase_amplitudes = np.stack(phase_amplitudes_list, axis=0)
        dataset = tf.data.Dataset.from_tensor_slices((parameters, phase_amplitudes))
        return dataset

    @classmethod
    def to_prepared_tensorflow_dataset(cls, examples: List[NicerExample], batch_size: int = 100,
                                       shuffle: bool = False) -> tf.data.Dataset:
        dataset = cls.to_tensorflow_dataset(examples)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size)
        return dataset
