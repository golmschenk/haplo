from pathlib import Path

from torch.optim import AdamW

from haplo.losses import PlusOneBeforeUnnormalizationChiSquaredStatisticMetric, \
    PlusOneChiSquaredStatisticMetric
from haplo.models import Cura
from haplo.nicer_dataset import NicerDataset, split_dataset_into_fractional_datasets
from haplo.nicer_parameters_to_phase_amplitudes_train import train_session, \
    add_norm_based_gradient_clip_to_all_parameters
from haplo.nicer_transform import PrecomputedNormalizeParameters, PrecomputedNormalizePhaseAmplitudes


def example_train_session():
    full_dataset_path = Path('data/800k_parameters_and_phase_amplitudes.db')
    full_train_dataset = NicerDataset.new(
        dataset_path=full_dataset_path,
        parameters_transform=PrecomputedNormalizeParameters(),
        phase_amplitudes_transform=PrecomputedNormalizePhaseAmplitudes())
    train_dataset, validation_dataset, test_dataset = split_dataset_into_fractional_datasets(full_train_dataset,
                                                                                             [0.8, 0.1, 0.1])
    model = Cura()
    add_norm_based_gradient_clip_to_all_parameters(model)
    loss_function = PlusOneBeforeUnnormalizationChiSquaredStatisticMetric()
    metric_functions = [PlusOneChiSquaredStatisticMetric(), PlusOneBeforeUnnormalizationChiSquaredStatisticMetric()]
    learning_rate = 1e-4
    optimizer_epsilon = 1e-7
    weight_decay = 0.0001
    batch_size_per_device = 100
    cycles_to_run = 5000
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=optimizer_epsilon)
    run_comments = f'example_run'  # Whatever you want to log in a string.
    hyperparameter_log_dictionary = {
        'model_name': type(model).__name__, 'learning_rate': learning_rate,
        'batch_size_per_device': batch_size_per_device,
        'optimizer_epsilon': optimizer_epsilon, 'weight_decay': weight_decay, 'run_comments': run_comments
    }
    train_session(train_dataset, validation_dataset, model, loss_function, metric_functions, optimizer,
                  batch_size_per_device, cycles_to_run, wandb_project='example', wandb_entity='ramjet',
                  hyperparameter_log_dictionary=hyperparameter_log_dictionary)


if __name__ == '__main__':
    example_train_session()
