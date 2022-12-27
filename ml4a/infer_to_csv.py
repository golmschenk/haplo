import random

import numpy as np
from bokeh.io import save, export_svg
from bokeh.models import Column
from bokeh.plotting import Figure
from pathlib import Path

from ml4a.nicer_example import NicerExample
from ml4a.nicer_model import Nyx9Wider, Nyx11, Nyx9Re, Nyx9ReTraditionalShape
from ml4a.residual_model import Lira, NormalizingModelWrapper, LiraTraditionalShape, \
    LiraTraditionalShapeDoubleWidthWithExtraEndLayer, LiraTraditionalShape8xWidthWith0d5DoNoBn


def main():
    model = LiraTraditionalShape8xWidthWith0d5DoNoBn()
    model_trial_name = "LiraTraditionalShape8xWidthWith0d5DoNoBn_chi_squared_loss"
    model_trial_directory = Path("logs").joinpath(model_trial_name)
    model.load_weights(model_trial_directory.joinpath('best_validation_model.ckpt'))

    dataset_path = Path("data/mcmc_vac_all_f90.dat")
    examples = NicerExample.list_from_constantinos_kalapotharakos_file(dataset_path)
    random.Random(0).shuffle(examples)
    tenth_dataset_count = int(len(examples) * 0.1)
    train_examples = examples[:-2 * tenth_dataset_count]
    validation_examples = examples[-2 * tenth_dataset_count:-tenth_dataset_count]
    test_examples = examples[-tenth_dataset_count:]
    print(f'Dataset sizes: train={len(train_examples)}, validation={len(validation_examples)}, '
          f'test={len(test_examples)}')
    test_dataset = NicerExample.to_prepared_tensorflow_dataset(test_examples, batch_size=1000,
                                                               normalize_parameters_and_phase_amplitudes=True)
    batch_predictions = []
    processed_count = 0
    for (index, test_example) in enumerate(test_dataset):
        test_input, test_output = test_example
        model_predicted_test_output = model.predict(test_input)
        batch_unnormalized_phase_amplitudes = NicerExample.unnormalize_phase_amplitudes(model_predicted_test_output)
        batch_predictions.append(batch_unnormalized_phase_amplitudes)
        processed_count += batch_unnormalized_phase_amplitudes.shape[0]
        print(f'{processed_count} processed.')
    unnormalized_phase_amplitudes = np.concatenate(batch_predictions, axis=0)
    np.savetxt(model_trial_directory.joinpath(f'inferred_test_phase_amplitudes.csv'), unnormalized_phase_amplitudes,
               delimiter=',')


if __name__ == "__main__":
    main()
