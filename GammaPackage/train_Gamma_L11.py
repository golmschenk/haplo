from pathlib import Path

from haplo.unwrap_model import unwrap_model
from torch.nn import Module
from torch.optim import AdamW

from haplo.distributed import distributed_logging
from haplo.nicer_dataset import split_dataset_into_count_datasets
from haplo.train_hyperparameter_configuration import TrainHyperparameterConfiguration
from haplo.train_logging_configuration import TrainLoggingConfiguration
from haplo.train_session import train_session
from haplo.train_system_configuration import TrainSystemConfiguration

import os

import torch

from torch import load

from cura_2D_model import Cura2D

from GammaSkymapDataset import GammaSkymapDataset
from Gamma_Normalization import PrecomputedUnnormalizePhaseAmplitudes


class GammaLossL11(Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        epsilon = 1e-10
        # unnormalize_phase_amplitudes = PrecomputedUnnormalizePhaseAmplitudes()
        # observed = unnormalize_phase_amplitudes(output.type(torch.float64)) + 1.0
        # expected = unnormalize_phase_amplitudes(target.type(torch.float64)) + 1.0
        observed = output.type(torch.float64)
        expected = target.type(torch.float64)
        # numerator = torch.sum(((observed - expected) ** 2), dim=1)
        # median = torch.median(expected, dim=1).values

        # Calculate numerator as sum of squared differences across all dimensions except batch dimension
        numerator = torch.sum((observed - expected) ** 2, dim=(1, 2))

        quality_indicator = numerator

        metric_f64 = torch.mean(torch.log10(quality_indicator + epsilon))
        metric = metric_f64.type(torch.float32)

        return metric



@distributed_logging
def example_train_session():
    os.environ["WANDB_MODE"] = "offline"


    output_file_path =  "/nobackupp27/tlechien/Neural/data/Gamma_halfplane_v3.csv"

    # To continue training:
    # savedmodelPath = Path("/nobackupp27/tlechien/Neural/sessions/2024_11_02_16_20_24_train_Gamma_L11/lowest_validation_model.pt")
    # optimizerPath =  Path("/nobackupp27/tlechien/Neural/sessions/2024_11_02_16_20_24_train_Gamma_L11/lowest_validation_optimizer.pt")


    dataset = GammaSkymapDataset(output_file_path,randomDataAugmentation=True,normalize=False) # Load in the data


    test_dataset, validation_dataset, train_dataset, _ = split_dataset_into_count_datasets(
        dataset, [int(len(dataset)/10), int(len(dataset)/10), int(8*len(dataset)/10 - 1)])

    print(len(test_dataset))
    print(len(validation_dataset))
    print(len(train_dataset))


    model = Cura2D(input_features=11)

    loss_function = GammaLossL11()
    metric_functions = [GammaLossL11()]

    hyperparameter_configuration = TrainHyperparameterConfiguration.new()
    hyperparameter_configuration.batch_size = 10
    system_configuration = TrainSystemConfiguration.new(preprocessing_processes_per_train_process=2)
    optimizer = AdamW(params=model.parameters(), lr=hyperparameter_configuration.learning_rate,
                      weight_decay=hyperparameter_configuration.weight_decay,
                      eps=hyperparameter_configuration.optimizer_epsilon)


    # Finetune: start from existing trained model
    # model.load_state_dict(unwrap_model(load(savedmodelPath, map_location='cpu')))
    # optimizer.load_state_dict(unwrap_model(load(optimizerPath, map_location='cpu')))


    run_comments = f'Example run.'  # Whatever you want to log in a string.
    additional_log_dictionary = {
        'model_name': type(model).__name__, 'train_dataset_size': len(train_dataset), 'run_comments': run_comments
    }
    logging_configuration = TrainLoggingConfiguration.new(
        wandb_project='thibault', wandb_entity='ramjet', additional_log_dictionary=additional_log_dictionary
    )
    train_session(train_dataset=train_dataset, validation_dataset=validation_dataset, model=model,
                  loss_function=loss_function, metric_functions=metric_functions, optimizer=optimizer,
                  hyperparameter_configuration=hyperparameter_configuration, system_configuration=system_configuration,
                  logging_configuration=logging_configuration)


if __name__ == '__main__':
    example_train_session()
