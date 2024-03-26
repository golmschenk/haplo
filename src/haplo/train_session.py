import getpass
import logging
import math
import os
import socket
from pathlib import Path
from typing import Callable, List

import numpy as np
import stringcase
import torch
import wandb as wandb
from torch import Tensor, tensor
from torch.distributed import init_process_group, destroy_process_group, ReduceOp
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.types import Device
from torch.utils.data import DataLoader, DistributedSampler, Dataset

from haplo.distributed import ddp_setup
from haplo.logging import set_up_default_logger
from haplo.losses import norm_based_gradient_clip
from haplo.nicer_dataset import nicer_dataset_worker_initialization_function, disconnect, move_sqlite_subset_to_new_file
from haplo.rank_constant_distributed_sampler import RankConstantDistributedSampler
from haplo.train_hyperparameter_configuration import TrainHyperparameterConfiguration
from haplo.train_logging_configuration import TrainLoggingConfiguration
from haplo.train_system_configuration import TrainSystemConfiguration
from haplo.unwrap_model import unwrap_model
from haplo.wandb_liaison import wandb_init, wandb_log, wandb_commit, \
    wandb_log_dictionary, wandb_log_data_class, wandb_save_manual_config_file

logger = logging.getLogger(__name__)
non_blocking = True  # TODO: Should probably be elsewhere.
pin_memory = True  # TODO: Should probably be elsewhere.


def train_session(train_dataset: Dataset, validation_dataset: Dataset, model: Module, loss_function: Module,
                  metric_functions: List[Module], optimizer: Optimizer,
                  hyperparameter_configuration: TrainHyperparameterConfiguration,
                  system_configuration: TrainSystemConfiguration,
                  logging_configuration: TrainLoggingConfiguration):
    torch.multiprocessing.set_start_method('spawn')
    ddp_setup(system_configuration)
    set_up_default_logger()
    logger.info(f'Host: {socket.gethostname()}')
    logger.info('Starting training...')
    logging_configuration.session_directory.mkdir(exist_ok=True, parents=True)
    local_rank, process_rank, world_size = get_distributed_world_information()
    wandb_init(process_rank=process_rank, project=logging_configuration.wandb_project,
               entity=logging_configuration.wandb_entity, settings=wandb.Settings(start_method='fork'))
    wandb_log_data_class(hyperparameter_configuration, process_rank=process_rank)
    wandb_log_data_class(system_configuration, process_rank=process_rank)
    wandb_log_dictionary(logging_configuration.additional_log_dictionary, process_rank=process_rank)
    log_distributed_settings(hyperparameter_configuration, system_configuration, process_rank)
    logger.info(wandb.config)
    wandb_save_manual_config_file(process_rank)

    network_device = get_device(local_rank)
    loss_function = loss_function.to(network_device)
    metric_functions = [metric_function.to(network_device) for metric_function in metric_functions]

    model = distribute_model_across_devices(model, optimizer, network_device)

    logger.info(f'{process_rank}: Loading dataset...')
    optimizer.zero_grad()
    train_dataloader, validation_dataloader = create_data_loaders(train_dataset, validation_dataset,
                                                                  hyperparameter_configuration.batch_size,
                                                                  system_configuration)
    sessions_directory = Path('sessions')
    sessions_directory.mkdir(parents=True, exist_ok=True)

    train_loop(model, train_dataloader, validation_dataloader, optimizer, loss_function, metric_functions,
               hyperparameter_configuration.cycles, network_device, process_rank, world_size,
               logging_configuration)

    destroy_process_group()


def log_distributed_settings(hyperparameter_configuration: TrainHyperparameterConfiguration,
                             system_configuration: TrainSystemConfiguration, process_rank: int) -> None:
    training_processes = int(os.environ.get('WORLD_SIZE'))
    training_processes_per_node = int(os.environ.get('LOCAL_WORLD_SIZE'))
    wandb_log_dictionary(
        {
            'training_processes': training_processes,
            'training_processes_per_node': training_processes_per_node,
            'nodes': training_processes // training_processes_per_node,
            'preprocessing_processes': training_processes * system_configuration.preprocessing_processes_per_train_process,
            'global_batch_size': training_processes * hyperparameter_configuration.batch_size
        },
        process_rank=process_rank)


def distribute_model_across_devices(model, optimizer, device):
    model = model.to(device, non_blocking=non_blocking)
    optimizer_to_device(optimizer, device)
    if torch.cuda.is_available():
        logger.info(f'Number of CUDA devices found on host: {torch.cuda.device_count()}')
        logger.info(f'Device: {device}')
        torch.cuda.set_device(device)
        model = DistributedDataParallel(model, device_ids=[device])
    else:
        logger.info(f'Device: cpu')
        model = DistributedDataParallel(model)
    return model


def train_loop(model, train_dataloader, validation_dataloader, optimizer, loss_function, metric_functions,
               cycles_to_run, network_device, process_rank, world_size,
               logging_configuration: TrainLoggingConfiguration):
    lowest_validation_cycle_loss = tensor(math.inf)
    logger.info(f'{process_rank}: Starting training loop...')
    for cycle in range(cycles_to_run):
        logger.info(f"Epoch {cycle} -------------------------------")
        train_phase(train_dataloader, model, loss_function, optimizer, network_device=network_device,
                    cycle=cycle, metric_functions=metric_functions, process_rank=process_rank,
                    world_size=world_size)
        validation_cycle_loss = validation_phase(validation_dataloader, model, loss_function,
                                                 network_device=network_device,
                                                 cycle=cycle,
                                                 metric_functions=metric_functions,
                                                 process_rank=process_rank, world_size=world_size)
        save_state(model, optimizer, logging_configuration, state_name_prefix='latest', process_rank=process_rank)
        if (
                logging_configuration.model_save_cycle_frequency is not None and
                cycle != 0 and
                cycle % logging_configuration.model_save_cycle_frequency == 0
        ):
            save_state(model, optimizer, logging_configuration, state_name_prefix=f'cycle_{cycle}',
                       process_rank=process_rank)
        if validation_cycle_loss < lowest_validation_cycle_loss:
            lowest_validation_cycle_loss = validation_cycle_loss
            save_state(model, optimizer, logging_configuration, state_name_prefix='lowest_validation',
                       process_rank=process_rank)
        wandb_log('epoch', cycle, process_rank=process_rank)
        wandb_log('cycle', cycle, process_rank=process_rank)
        wandb_commit(process_rank=process_rank)
    logger.info("Done!")


def get_device(local_rank):
    if torch.cuda.is_available():
        network_device = torch.device(f'cuda:{local_rank}')
    else:
        network_device = torch.device('cpu')
    return network_device


def get_distributed_world_information():
    process_rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    return local_rank, process_rank, world_size


def create_data_loaders(train_dataset, validation_dataset, batch_size_per_device,
                        system_configuration: TrainSystemConfiguration):
    if system_configuration.preprocessing_processes_per_train_process > 0:
        prefetch_factor = 10
        persistent_workers = True
    else:
        prefetch_factor = None
        persistent_workers = False
    # TODO: Hacked way of passing samplers to init functions.
    train_sampler = RankConstantDistributedSampler(train_dataset)
    validation_sampler = RankConstantDistributedSampler(validation_dataset)
    if train_dataset.dataset.in_memory:
        # TODO: Hacked way of moving dataset to tmp.
        logger.info(f'Start of in memory branch...')
        local_rank, process_rank, world_size = get_distributed_world_information()
        train_indexes = list(iter(train_sampler))
        validation_indexes = list(iter(validation_sampler))
        train_subset_indexes = np.array(train_dataset.indices)[train_indexes].tolist()
        validation_subset_indexes = np.array(validation_dataset.indices)[validation_indexes].tolist()
        indexes = train_subset_indexes + validation_subset_indexes
        logger.info(f'Number of indexes on process: {len(indexes)}')
        # TODO: This hardcoded path changes depending on system.
        local_directory = Path(f'/tmp/{getpass.getuser()}')
        local_directory.mkdir(parents=True, exist_ok=True)
        local_database_path = local_directory.joinpath(f'qusi_{process_rank}.db')
        local_database_path.unlink(missing_ok=True)
        move_sqlite_subset_to_new_file(train_dataset.dataset.database_path, local_database_path, indexes)
        database_path = local_database_path
        database_uri = f'sqlite:///{database_path}?mode=ro'
        train_dataset.dataset.database_path = database_path
        validation_dataset.dataset.database_path = database_path
        train_dataset.dataset.database_uri = database_uri
        validation_dataset.dataset.database_uri = database_uri
        disconnect(train_dataset.dataset)
        disconnect(validation_dataset.dataset)
    logger.info(f'Creating train data loader...')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_per_device,
                                  num_workers=system_configuration.preprocessing_processes_per_train_process,
                                  pin_memory=pin_memory, persistent_workers=persistent_workers,
                                  prefetch_factor=prefetch_factor, shuffle=False,
                                  sampler=train_sampler,
                                  worker_init_fn=nicer_dataset_worker_initialization_function)
    logger.info(f'Creating validation data loader...')
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size_per_device,
                                       num_workers=system_configuration.preprocessing_processes_per_train_process,
                                       pin_memory=pin_memory, persistent_workers=persistent_workers,
                                       prefetch_factor=prefetch_factor, shuffle=False,
                                       sampler=validation_sampler,
                                       worker_init_fn=nicer_dataset_worker_initialization_function)
    logger.info(f'Data loaders created.')
    return train_dataloader, validation_dataloader


def save_state(model: Module, optimizer: Optimizer, logging_configuration: TrainLoggingConfiguration,
               state_name_prefix: str, process_rank: int):
    if process_rank == 0:
        torch.save(model.state_dict(),
                   logging_configuration.session_directory.joinpath(f'{state_name_prefix}_model.pt'))
        torch.save(optimizer.state_dict(), logging_configuration.session_directory.joinpath(
            f'{state_name_prefix}_optimizer.pt'))


def load_latest_state(model: Module, optimizer: Optimizer, session_directory: Path):
    model.load_state_dict(unwrap_model(torch.load(session_directory.joinpath('latest_model.pt'), map_location='cpu')))
    optimizer.load_state_dict(
        unwrap_model(torch.load(session_directory.joinpath('latest_optimizer.pt'), map_location='cpu')))


def train_phase(dataloader: DataLoader, model: Module, loss_function: Callable[[Tensor, Tensor], Tensor],
                optimizer: Optimizer, network_device: Device, cycle: int,
                metric_functions: List[Callable[[Tensor, Tensor], Tensor]], process_rank: int, world_size: int):
    model.train()
    total_cycle_loss = tensor(0, dtype=torch.float32, device='cpu', requires_grad=False)
    metric_totals = torch.zeros(size=[len(metric_functions)], device='cpu', requires_grad=False)
    assert (isinstance(dataloader.sampler, DistributedSampler) or
            isinstance(dataloader.sampler, RankConstantDistributedSampler))
    dataloader.sampler.set_epoch(cycle)
    batch_count = 0
    for batch, (parameters, light_curves) in enumerate(dataloader):
        parameters = parameters.to(network_device, non_blocking=non_blocking)
        light_curves = light_curves.to(network_device, non_blocking=non_blocking)
        predicted_light_curves = model(parameters)
        loss, total_cycle_loss = record_metrics(predicted_light_curves, light_curves, loss_function, total_cycle_loss,
                                                metric_functions, metric_totals)
        optimizer.zero_grad()
        loss_on_loss_device = loss.to(network_device, non_blocking=non_blocking)
        loss_on_loss_device.backward()
        apply_norm_based_gradient_clip_to_all_parameters(model)
        optimizer.step()

        if batch % 100 == 0:
            current = (batch + 1) * len(parameters)
            logger.info(f"loss: {loss.item():>7f}  [{current:>5d}/{len(dataloader.sampler):>5d}]")
        batch_count += 1
    log_metrics(total_cycle_loss, metric_functions, metric_totals, '', batch_count, world_size, process_rank)


def record_metrics(predicted_light_curves, light_curves, loss_function, total_cycle_loss, metric_functions,
                   metric_totals):
    loss = loss_function(predicted_light_curves, light_curves)
    total_cycle_loss += loss.item()
    for metric_function_index, metric_function in enumerate(metric_functions):
        batch_metric_value = metric_function(predicted_light_curves.detach(), light_curves)
        metric_totals[metric_function_index] += batch_metric_value.item()
    return loss, total_cycle_loss


def validation_phase(dataloader: DataLoader, model: Module, loss_function: Callable[[Tensor, Tensor], Tensor],
                     network_device: Device, cycle: int,
                     metric_functions: List[Callable[[Tensor, Tensor], Tensor]], process_rank: int, world_size: int
                     ) -> float:
    total_cycle_loss = tensor(0, dtype=torch.float32, device='cpu', requires_grad=False)
    metric_totals = torch.zeros(size=[len(metric_functions)], device='cpu', requires_grad=False)
    model.eval()
    assert (isinstance(dataloader.sampler, DistributedSampler) or
            isinstance(dataloader.sampler, RankConstantDistributedSampler))
    dataloader.sampler.set_epoch(cycle)
    with torch.no_grad():
        batch_count = 0
        for parameters, light_curves in dataloader:
            parameters = parameters.to(network_device, non_blocking=non_blocking)
            light_curves = light_curves.to(network_device, non_blocking=non_blocking)
            predicted_light_curves = model(parameters)
            loss, total_cycle_loss = record_metrics(predicted_light_curves, light_curves, loss_function,
                                                    total_cycle_loss,
                                                    metric_functions, metric_totals)
            batch_count += 1

    cycle_loss = log_metrics(total_cycle_loss, metric_functions, metric_totals, 'val_', batch_count, world_size,
                             process_rank)
    return cycle_loss


def log_metrics(total_cycle_loss: Tensor, metric_functions: List[Callable[[Tensor, Tensor], Tensor]],
                metric_totals: Tensor, prefix: str, number_of_batches: int, world_size: int, process_rank: int
                ) -> float:
    cycle_loss = total_cycle_loss / number_of_batches
    torch.distributed.reduce(cycle_loss, dst=0, op=ReduceOp.SUM)
    cycle_loss /= world_size
    wandb_log(f'{prefix}loss', cycle_loss, process_rank=process_rank)
    cycle_metric_values = metric_totals / number_of_batches
    torch.distributed.reduce(cycle_metric_values, dst=0, op=ReduceOp.SUM)
    cycle_metric_values /= world_size
    for metric_function_index, metric_function in enumerate(metric_functions):
        cycle_metric_value = cycle_metric_values[metric_function_index]
        wandb_log(f'{prefix}{get_metric_name(metric_function)}', cycle_metric_value,
                  process_rank=process_rank)
    return cycle_loss


def get_metric_name(metric_function):
    metric_name = type(metric_function).__name__
    metric_name = stringcase.snakecase(metric_name)
    metric_name = metric_name.replace('_metric', '').replace('_loss', '')
    return metric_name


def add_norm_based_gradient_clip_to_all_parameters(model):
    for parameter in model.parameters():
        parameter.register_hook(norm_based_gradient_clip)


def apply_norm_based_gradient_clip_to_all_parameters(model):
    for parameter in model.parameters():
        clip_grad_norm_(parameter, 1.0)


def optimizer_to_device(optimizer, device):
    for param in optimizer.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device, non_blocking=non_blocking)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device, non_blocking=non_blocking)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device, non_blocking=non_blocking)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device, non_blocking=non_blocking)
