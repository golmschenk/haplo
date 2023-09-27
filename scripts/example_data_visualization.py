from pathlib import Path

from bokeh.io import show
from bokeh.models import Column
from bokeh.plotting import figure

from haplo.nicer_dataset import NicerDataset


def example_data_visualization():
    dataset_path = Path('data/800k_parameters_and_phase_amplitudes.db')
    dataset = NicerDataset.new(dataset_path=dataset_path)
    
    example0 = dataset[0]
    parameters0, phase_amplitudes0 = example0

    parameters_figure = figure(x_axis_label='parameter', y_axis_label='value')
    parameters_figure.vbar(x=list(range(11)), top=parameters0)

    phase_amplitudes_figure = figure(x_axis_label='phase', y_axis_label='amplitude')
    phase_amplitudes_figure.line(x=list(range(64)), y=phase_amplitudes0)
    
    column = Column(parameters_figure, phase_amplitudes_figure)
    show(column)


if __name__ == '__main__':
    example_data_visualization()