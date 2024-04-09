import os
import tempfile
from pathlib import Path
from torch.optim import AdamW

from haplo.losses import SumDifferenceSquaredOverMedianExpectedSquaredMetric, PlusOneChiSquaredStatisticMetric, \
    PlusOneBeforeUnnormalizationChiSquaredStatisticMetric
from haplo.models import SingleDenseNetwork
from haplo.nicer_dataset import NicerDataset, split_dataset_into_count_datasets
from haplo.nicer_transform import PrecomputedNormalizeParameters, PrecomputedNormalizePhaseAmplitudes
from haplo.train_hyperparameter_configuration import TrainHyperparameterConfiguration
from haplo.train_logging_configuration import TrainLoggingConfiguration
from haplo.train_session import train_session
from haplo.train_system_configuration import TrainSystemConfiguration


def test_simple_train_session():
    os.environ["WANDB_MODE"] = "disabled"
    full_dataset_path = Path(__file__).parent.joinpath(
        'test_train_session_resources/300_parameters_and_phase_amplitudes.db')
    full_train_dataset = NicerDataset.new(
        dataset_path=full_dataset_path,
        length=300,
        parameters_transform=PrecomputedNormalizeParameters(),
        phase_amplitudes_transform=PrecomputedNormalizePhaseAmplitudes(),
        in_memory=True
    )
    test_dataset, validation_dataset, train_dataset, _ = split_dataset_into_count_datasets(
        full_train_dataset, [10, 10, 100])
    model = SingleDenseNetwork()
    loss_function = SumDifferenceSquaredOverMedianExpectedSquaredMetric()
    metric_functions = [PlusOneChiSquaredStatisticMetric(), PlusOneBeforeUnnormalizationChiSquaredStatisticMetric(),
                        SumDifferenceSquaredOverMedianExpectedSquaredMetric()]
    hyperparameter_configuration = TrainHyperparameterConfiguration.new(cycles=5, batch_size=50)
    system_configuration = TrainSystemConfiguration.new(preprocessing_processes_per_train_process=0)
    optimizer = AdamW(params=model.parameters(), lr=hyperparameter_configuration.learning_rate,
                      weight_decay=hyperparameter_configuration.weight_decay,
                      eps=hyperparameter_configuration.optimizer_epsilon)
    run_comments = 'run_comments_placeholder'  # Whatever you want to log in a string.
    additional_log_dictionary = {
        'model_name': type(model).__name__, 'train_dataset_size': len(train_dataset), 'run_comments': run_comments
    }
    logging_configuration = TrainLoggingConfiguration.new(
        wandb_project='test', wandb_entity='test', additional_log_dictionary=additional_log_dictionary,
        session_directory=Path(tempfile.gettempdir()))
    train_session(train_dataset=train_dataset, validation_dataset=validation_dataset, model=model,
                  loss_function=loss_function, metric_functions=metric_functions, optimizer=optimizer,
                  hyperparameter_configuration=hyperparameter_configuration, system_configuration=system_configuration,
                  logging_configuration=logging_configuration)
