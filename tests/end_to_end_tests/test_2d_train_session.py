import math
import os
import tempfile
from pathlib import Path

import numpy as np
import wandb
import xarray
from torch.optim import AdamW

from haplo.internal.dataset.xarray_zarr import XarrayBasedDataset
from haplo.internal.metrics import SumDifferenceSquaredOverMedianExpectedSquaredMetric, \
    PlusOneChiSquaredStatisticMetric, MeanDifferenceSquaredOverMedianExpectedSquaredMetric2d, \
    PlusOneChiSquaredStatisticMetric2d
from haplo.internal.models.single_dense_network import SingleDenseNetworkWith2dOutput
from haplo.models import SingleDenseNetwork
from haplo.internal.dataset.split import split_dataset_into_count_datasets
from haplo.internal.train_hyperparameter_configuration import TrainHyperparameterConfiguration
from haplo.internal.train_logging_configuration import TrainLoggingConfiguration
from haplo.internal.train_session import train_session
from haplo.internal.train_system_configuration import TrainSystemConfiguration


def test_simple_train_session():
    if wandb.run is not None:
        wandb.finish()
    os.environ['WANDB_MODE'] = 'offline'
    os.environ['WANDB_DISABLED'] = 'true'
    xarray_dataset = xarray.Dataset(
        coords={
            'index': np.arange(100),
            'parameter_index': np.arange(11, dtype=np.int32),
            'phase_bin': np.linspace(0, math.tau, num=100, endpoint=False, dtype=np.float32),
            'energy_bin': np.linspace(1, 1e10, num=100, endpoint=False, dtype=np.float32),
        },
        data_vars={
            'input': (
                ['index', 'parameter_index'],
                np.arange(100 * 11, dtype=np.float32).reshape([100, 11]),
            ),
            'output': (
                ['index', 'energy_bin', 'phase_bin'],
                np.arange(100 * 100 * 100, dtype=np.float32).reshape([100, 100, 100]),
            ),
        }
    )
    full_dataset = XarrayBasedDataset(xarray_dataset=xarray_dataset)
    test_dataset, validation_dataset, train_dataset = split_dataset_into_count_datasets(
        full_dataset, [10, 10])
    model = SingleDenseNetworkWith2dOutput()
    loss_function = MeanDifferenceSquaredOverMedianExpectedSquaredMetric2d()
    metric_functions = [PlusOneChiSquaredStatisticMetric2d(),
                        MeanDifferenceSquaredOverMedianExpectedSquaredMetric2d()]
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
