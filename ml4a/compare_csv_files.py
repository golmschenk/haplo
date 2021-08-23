from pathlib import Path

import numpy as np
from bokeh.io import show
from bokeh.models import Column
from bokeh.palettes import Category10
from bokeh.plotting import Figure


def display_comparison_of_csv_files(csv_path0: Path, csv_path1: Path, number_of_rows_to_compare: int = 30) -> None:
    figures = []
    array0 = np.loadtxt(str(csv_path0))
    array1 = np.loadtxt(str(csv_path1))
    for row_index in range(number_of_rows_to_compare):
        figure = Figure()
        figure.line(x=range(len(array0[row_index])), y=array0[row_index], color=Category10[10][0])
        figure.line(x=range(len(array1[row_index])), y=array1[row_index], color=Category10[10][3])
        figures.append(figure)
    column = Column(*figures)
    show(column)


if __name__ == '__main__':
    display_comparison_of_csv_files(
        Path('true_validation_phase_amplitudes.csv'),
        Path('predicted_validation_phase_amplitudes.csv')
    )
    # display_comparison_of_csv_files(
    #     Path('true_validation_parameters.csv'),
    #     Path('predicted_validation_parameters.csv')
    # )
