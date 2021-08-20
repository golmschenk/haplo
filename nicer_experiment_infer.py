import random
from bokeh.io import save, export_svg
from bokeh.models import Column
from bokeh.plotting import Figure
from pathlib import Path

from ml4a.nicer_example import NicerExample
from ml4a.nicer_model import Nyx9Wider, Nyx11


def main():
    dataset_path = Path("data/mcmc_vac_all_f90.dat")
    examples = NicerExample.list_from_constantinos_kalapotharakos_file(dataset_path)
    random.Random(0).shuffle(examples)
    tenth_dataset_count = int(len(examples) * 0.1)
    train_examples = examples[:-2 * tenth_dataset_count]
    validation_examples = examples[-2 * tenth_dataset_count:-tenth_dataset_count]
    test_examples = examples[-tenth_dataset_count:]
    print(f'Dataset sizes: train={len(train_examples)}, validation={len(validation_examples)},'
          f'test={len(test_examples)}')
    test_dataset = NicerExample.to_prepared_tensorflow_dataset(test_examples, batch_size=1)

    old_model = Nyx11()
    old_trial_name = "misunderstood-moon"
    old_trial_directory = Path("logs").joinpath(old_trial_name)
    old_model.load_weights(old_trial_directory.joinpath('best_validation_model.ckpt'))
    model = Nyx9Wider()
    trial_name = "dark-pond-partial"
    trial_directory = Path("logs").joinpath(trial_name)
    model.load_weights(trial_directory.joinpath('best_validation_model.ckpt'))
    figures = []
    for (index, test_example) in enumerate(test_dataset):
        if index >= 6:
            break
        test_input, test_output = test_example
        predicted_test_output = model.predict(test_input)
        old_predicted_test_output = old_model.predict(test_input)
        test_output = test_output.numpy()
        figure = Figure()
        figure.output_backend = "svg"
        figure.background_fill_color = None
        figure.border_fill_color = None
        figure.line(x=range(test_output.shape[1]), y=test_output[0], line_width=2)
        figure.line(x=range(old_predicted_test_output.shape[1]), y=old_predicted_test_output[0], line_width=2, color='darkgoldenrod')
        figure.line(x=range(predicted_test_output.shape[1]), y=predicted_test_output[0], line_width=2, color='firebrick')
        export_svg(figure, filename=f"{index}.svg")
        figures.append(figure)
    column = Column(*figures)
    save(column, trial_directory.joinpath(f"infer_test_examples.html"))


if __name__ == "__main__":
    main()
