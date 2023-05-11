import random

import numpy as np
from bokeh.io import save, export_svg
from bokeh.models import Column
from bokeh.plotting import Figure
from pathlib import Path

from ml4a.losses import PlusOneChiSquaredStatisticLoss, PlusOneChiSquaredStatisticLossNoUnnormalizing
from ml4a.nicer_example import NicerExample
from ml4a.nicer_model import Nyx9Wider, Nyx11, Nyx9Re, Nyx9ReTraditionalShape
from ml4a.residual_model import Lira, NormalizingModelWrapper, LiraTraditionalShape, \
    LiraTraditionalShapeDoubleWidthWithExtraEndLayer, LiraTraditionalShape8xWidthWith0d5DoNoBn, \
    LiraTraditionalShape8xWidthWithNoDoNoBn
import subprocess


def main():
    model_class = LiraTraditionalShape8xWidthWithNoDoNoBn
    model_trial_name = "LiraTraditionalShape8xWidthWithNoDoNoBn_chi_squared_loss_50m_dataset_small_batch_clip_norm_1_cont_from_with_do"
    model = model_class()
    model_trial_directory = Path("logs").joinpath(model_trial_name)
    model_path = model_trial_directory.joinpath('best_validation_model.ckpt')
    model.load_weights(model_path)

    dataset_path = Path("data/mcmc_vac_all_10m.dat")
    examples = NicerExample.list_from_constantinos_kalapotharakos_file(dataset_path)
    random.Random(0).shuffle(examples)
    tenth_dataset_count = int(len(examples) * 0.1)
    train_examples = examples[:-2 * tenth_dataset_count]
    partial_train_examples = train_examples[:100_000]
    validation_examples = examples[-2 * tenth_dataset_count:-tenth_dataset_count]
    test_examples = examples[-tenth_dataset_count:]
    partial_test_examples = test_examples[:100_000]
    print(f'Dataset sizes: train={len(train_examples)}, validation={len(validation_examples)}, '
          f'test={len(test_examples)}')
    partial_test_dataset = NicerExample.to_prepared_tensorflow_dataset(partial_test_examples,
                                                               normalize_parameters_and_phase_amplitudes=True)
    partial_train_dataset = NicerExample.to_prepared_tensorflow_dataset(partial_train_examples,
                                                                        normalize_parameters_and_phase_amplitudes=True)
    partial_train_parameters_array = NicerExample.extract_parameters_array(partial_train_examples)
    np.savetxt(model_trial_directory.joinpath(f'partial_train_parameters.csv'), partial_train_parameters_array,
               delimiter=',')
    partial_train_phase_amplitudes_array = NicerExample.extract_phase_amplitudes_array(partial_train_examples)
    np.savetxt(model_trial_directory.joinpath(f'partial_train_phase_amplitudes.csv'), partial_train_phase_amplitudes_array,
               delimiter=',')
    partial_test_parameters_array = NicerExample.extract_parameters_array(partial_test_examples)
    np.savetxt(model_trial_directory.joinpath(f'partial_test_parameters.csv'), partial_test_parameters_array,
               delimiter=',')
    partial_test_phase_amplitudes_array = NicerExample.extract_phase_amplitudes_array(partial_test_examples)
    np.savetxt(model_trial_directory.joinpath(f'partial_test_phase_amplitudes.csv'), partial_test_phase_amplitudes_array,
               delimiter=',')
    batch_predictions = []
    processed_count = 0
    for (index, train_example) in enumerate(partial_train_dataset):
        train_input, train_output = train_example
        model_predicted_train_output = model.predict(train_input)
        batch_unnormalized_phase_amplitudes = NicerExample.unnormalize_phase_amplitudes(model_predicted_train_output)
        batch_predictions.append(batch_unnormalized_phase_amplitudes)
        processed_count += batch_unnormalized_phase_amplitudes.shape[0]
        print(f'{processed_count} processed.')
    unnormalized_phase_amplitudes = np.concatenate(batch_predictions, axis=0)
    np.savetxt(model_trial_directory.joinpath(f'inferred_partial_train_phase_amplitudes.csv'), unnormalized_phase_amplitudes,
               delimiter=',')
    batch_predictions = []
    for (index, test_example) in enumerate(partial_test_dataset):
        test_input, test_output = test_example
        model_predicted_test_output = model.predict(test_input)
        batch_unnormalized_phase_amplitudes = NicerExample.unnormalize_phase_amplitudes(model_predicted_test_output)
        batch_predictions.append(batch_unnormalized_phase_amplitudes)
        processed_count += batch_unnormalized_phase_amplitudes.shape[0]
        print(f'{processed_count} processed.')
    unnormalized_phase_amplitudes = np.concatenate(batch_predictions, axis=0)
    np.savetxt(model_trial_directory.joinpath(f'inferred_partial_test_phase_amplitudes.csv'), unnormalized_phase_amplitudes,
               delimiter=',')
    chi_squared_mean = PlusOneChiSquaredStatisticLossNoUnnormalizing().plus_one_chi_squared_statistic(partial_test_phase_amplitudes_array, unnormalized_phase_amplitudes)
    print(f'Chi squared mean: {chi_squared_mean}')

    # model = load_trained_parameters_to_phase_amplitudes_model()
    model = model_class()
    model.load_weights(model_path).expect_partial()
    random_input = np.random.random(size=[1, 11, 1])
    _ = model.predict(random_input)
    model.save('check')
    subprocess.run(['python', '-m', 'tf2onnx.convert', '--saved-model', './check', '--opset=10', '--output', f'{model_trial_directory}/lira.onnx'])


if __name__ == "__main__":
    main()
