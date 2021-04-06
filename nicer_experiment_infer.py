import datetime
import random
import tensorflow as tf
from bokeh.io import show, save
from bokeh.models import Column
from bokeh.plotting import Figure
from tensorflow.python.keras import callbacks
from pathlib import Path

from nicer_example import NicerExample
from nicer_model import SimpleModel, Nyx2, Nyx10, Nyx11


def main():
    dataset_path = Path("data/mcmc_vac_all_f90.dat")
    examples = NicerExample.list_from_constantinos_kalapotharakos_file(dataset_path)
    random.Random(0).shuffle(examples)
    train_examples = examples[:8000]
    validation_examples = examples[8000:9000]
    test_examples = examples[9000:]
    test_dataset = NicerExample.to_prepared_tensorflow_dataset(test_examples, batch_size=1)

    model = Nyx11()
    trial_name = "misunderstood-moon"
    trial_directory = Path("logs").joinpath(trial_name)
    model.load_weights(trial_directory.joinpath('best_validation_model.ckpt'))
    figures = []
    for (index, test_example) in enumerate(test_dataset):
        if index >= 20:
            break
        test_input, test_output = test_example
        predicted_test_output = model.predict(test_input)
        test_output = test_output.numpy()
        figure = Figure()
        figure.line(x=range(test_output.shape[1]), y=test_output[0], line_width=2)
        figure.line(x=range(predicted_test_output.shape[1]), y=predicted_test_output[0], line_width=2, color='firebrick')
        figures.append(figure)
    column = Column(*figures)
    save(column, trial_directory.joinpath(f"infer_test_examples.html"))


if __name__ == "__main__":
    main()
