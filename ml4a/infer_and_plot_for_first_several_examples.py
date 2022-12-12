import random
from bokeh.io import save, export_svg
from bokeh.models import Column
from bokeh.plotting import Figure
from pathlib import Path

from ml4a.nicer_example import NicerExample
from ml4a.nicer_model import Nyx9Wider, Nyx11, Nyx9Re, Nyx9ReTraditionalShape
from ml4a.residual_model import Lira, NormalizingModelWrapper, LiraTraditionalShape, \
    LiraTraditionalShapeDoubleWidthWithExtraEndLayer, LiraTraditionalShape4xWidthWithDo, \
    LiraTraditionalShape8xWidthWith0d5DoNoBn


def main():
    model0 = Nyx9Wider()
    model0_trial_name = "Nyx9Widerer_no_do_l2_1000_cont2"
    model0_trial_directory = Path("logs").joinpath(model0_trial_name)
    model0.load_weights(model0_trial_directory.joinpath('best_validation_model.ckpt'))
    model1 = LiraTraditionalShape8xWidthWith0d5DoNoBn()
    model1_trial_name = "LiraTraditionalShape8xWidthWith0d5DoNoBn_chi_squared_loss"
    model1_trial_directory = Path("logs").joinpath(model1_trial_name)
    model1.load_weights(model1_trial_directory.joinpath('best_validation_model.ckpt'))
    model2 = LiraTraditionalShapeDoubleWidthWithExtraEndLayer()
    model2_trial_name = "LiraTraditionalShapeDoubleWidthWithExtraEndLayer_normalized_loss_lr_1e-5_cont2"
    model2_trial_directory = Path("logs").joinpath(model2_trial_name)
    model2.load_weights(model2_trial_directory.joinpath('best_validation_model.ckpt'))
    figures = []

    dataset_path = Path("data/mcmc_vac_all_f90.dat")
    examples = NicerExample.list_from_constantinos_kalapotharakos_file(dataset_path)
    random.Random(0).shuffle(examples)
    tenth_dataset_count = int(len(examples) * 0.1)
    train_examples = examples[:-2 * tenth_dataset_count]
    validation_examples = examples[-2 * tenth_dataset_count:-tenth_dataset_count]
    test_examples = examples[-tenth_dataset_count:]
    print(f'Dataset sizes: train={len(train_examples)}, validation={len(validation_examples)},'
          f'test={len(test_examples)}')
    test_dataset = NicerExample.to_prepared_tensorflow_dataset(test_examples, batch_size=1, normalize_parameters_and_phase_amplitudes=True)

    for (index, test_example) in enumerate(test_dataset):
        if index >= 10:
            break
        test_input, test_output = test_example
        model0_predicted_test_output = model0.predict(NicerExample.unnormalize_parameters(test_input))
        model1_predicted_test_output = model1.predict(test_input)
        model2_predicted_test_output = model2.predict(test_input)
        test_output = test_output.numpy()
        figure = Figure()
        figure.output_backend = "svg"
        figure.background_fill_color = None
        figure.border_fill_color = None
        figure.line(x=range(test_output.shape[1]), y=NicerExample.unnormalize_phase_amplitudes(test_output[0]), line_width=2)
        figure.line(x=range(model0_predicted_test_output.shape[1]), y=model0_predicted_test_output[0], line_width=2, color='darkgoldenrod')
        figure.line(x=range(model1_predicted_test_output.shape[1]), y=NicerExample.unnormalize_phase_amplitudes(model1_predicted_test_output[0]), line_width=2, color='firebrick')
        figure.line(x=range(model2_predicted_test_output.shape[1]), y=NicerExample.unnormalize_phase_amplitudes(model2_predicted_test_output[0]), line_width=2, color='forestgreen')
        # export_svg(figure, filename=f"{index}.svg")
        figures.append(figure)
    column = Column(*figures)
    save(column, model1_trial_directory.joinpath(f"infer_test_examples.html"))


if __name__ == "__main__":
    main()
