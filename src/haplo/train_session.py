import math
import os
from pathlib import Path
from typing import Callable, List, Dict, Any

import stringcase
import torch
import wandb as wandb
from torch import Tensor, tensor
from torch.distributed import init_process_group, destroy_process_group, Backend, ReduceOp
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.types import Device
from torch.utils.data import DataLoader, DistributedSampler, Dataset

from haplo.train_logging_configuration import TrainLoggingConfiguration
from haplo.losses import norm_based_gradient_clip
from haplo.train_system_configuration import TrainSystemConfiguration
from haplo.train_hyperparameter_configuration import TrainHyperparameterConfiguration
from haplo.wandb_liaison import wandb_init, wandb_log, wandb_commit, \
    wandb_log_dictionary, wandb_log_data_class


def train_session(train_dataset: Dataset, validation_dataset: Dataset, model: Module, loss_function: Module,
                  metric_functions: List[Module], optimizer: Optimizer,
                  hyperparameter_configuration: TrainHyperparameterConfiguration,
                  system_configuration: TrainSystemConfiguration,
                  logging_configuration: TrainLoggingConfiguration):
    print('Starting training...')
    print('Starting process spawning...')
    torch.multiprocessing.set_start_method('spawn')
    print('Starting DDP setup...')
    ddp_setup()
    local_rank, process_rank, world_size = get_distributed_world_information()
    print(f'{process_rank}: Starting wandb...')
    wandb_init(process_rank=process_rank, project=logging_configuration.wandb_project,
               entity=logging_configuration.wandb_entity, settings=wandb.Settings(start_method='fork'))
    wandb_log_data_class(hyperparameter_configuration, process_rank=process_rank)
    wandb_log_data_class(system_configuration, process_rank=process_rank)
    wandb_log_dictionary(logging_configuration.additional_log_dictionary, process_rank=process_rank)

    loss_device, network_device = get_devices(local_rank)

    model = distribute_model_across_devices(model, network_device, local_rank)

    print(f'{process_rank}: Loading dataset...')
    train_dataloader, validation_dataloader = create_data_loaders(train_dataset, validation_dataset,
                                                                  hyperparameter_configuration.batch_size,
                                                                  system_configuration)

    sessions_directory = Path('sessions')
    sessions_directory.mkdir(parents=True, exist_ok=True)

    train_loop(model, train_dataloader, validation_dataloader, optimizer, loss_function, metric_functions,
               hyperparameter_configuration.cycles, network_device, loss_device, process_rank, world_size)

    destroy_process_group()


def distribute_model_across_devices(model, device, local_rank):
    model = model.to(device, non_blocking=True)
    if torch.cuda.is_available():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    else:
        model = DistributedDataParallel(model)
    return model


def ddp_setup():
    if torch.cuda.is_available():
        distributed_back_end = Backend.NCCL
    else:
        distributed_back_end = Backend.GLOO
    distributed_back_end = Backend.GLOO
    if 'RANK' not in os.environ:
        # The script was not called with `torchrun` and environment variables need to be set manually.
        os.environ['RANK'] = str(0)
        os.environ['LOCAL_RANK'] = str(0)
        os.environ['WORLD_SIZE'] = str(1)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "35728"
    init_process_group(backend=distributed_back_end)


def train_loop(model, train_dataloader, validation_dataloader, optimizer, loss_function, metric_functions,
               cycles_to_run, network_device, loss_device, process_rank, world_size):
    lowest_validation_cycle_loss = tensor(math.inf)
    print(f'{process_rank}: Starting training loop...')
    for cycle in range(cycles_to_run):
        print(f"Epoch {cycle}\n-------------------------------")
        train_phase(train_dataloader, model, loss_function, optimizer, network_device=network_device,
                    loss_device=loss_device, cycle=cycle, metric_functions=metric_functions, process_rank=process_rank,
                    world_size=world_size)
        validation_cycle_loss = validation_phase(validation_dataloader, model, loss_function,
                                                 network_device=network_device,
                                                 loss_device=loss_device, cycle=cycle,
                                                 metric_functions=metric_functions,
                                                 process_rank=process_rank, world_size=world_size)
        save_model(model, suffix='latest_model', process_rank=process_rank)
        if validation_cycle_loss < lowest_validation_cycle_loss:
            lowest_validation_cycle_loss = validation_cycle_loss
            save_model(model, suffix='lowest_validation_model', process_rank=process_rank)
        wandb_log('epoch', cycle, process_rank=process_rank)
        wandb_log('cycle', cycle, process_rank=process_rank)
        wandb_commit(process_rank=process_rank)
    print("Done!")

    destroy_process_group()


def get_devices(local_rank):
    if torch.cuda.is_available():
        network_device = torch.device(f'cuda:{local_rank}')
        loss_device = network_device
    else:
        network_device = torch.device('cpu')
        loss_device = network_device
    return loss_device, network_device


def get_distributed_world_information():
    process_rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    return local_rank, process_rank, world_size


def create_data_loaders(train_dataset, validation_dataset, batch_size_per_device,
                        system_configuration: TrainSystemConfiguration):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_per_device,
                                  num_workers=system_configuration.preprocessing_processes_per_train_process,
                                  pin_memory=True, persistent_workers=True, prefetch_factor=10, shuffle=False,
                                  sampler=DistributedSampler(train_dataset))
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size_per_device,
                                       num_workers=system_configuration.preprocessing_processes_per_train_process,
                                       pin_memory=True, persistent_workers=True, prefetch_factor=10, shuffle=False,
                                       sampler=DistributedSampler(validation_dataset))
    return train_dataloader, validation_dataloader


def save_model(model: Module, suffix: str, process_rank: int):
    if process_rank == 0:
        model_name = wandb.run.name
        if model_name == '':
            model_name = wandb.run.id
        torch.save(model.state_dict(), Path(f'sessions/{model_name}_{suffix}.pt'))


def train_phase(dataloader: DataLoader, model: Module, loss_function: Callable[[Tensor, Tensor], Tensor],
                optimizer: Optimizer, network_device: Device, loss_device: Device, cycle: int,
                metric_functions: List[Callable[[Tensor, Tensor], Tensor]], process_rank: int, world_size: int):
    model.train()
    total_cycle_loss = tensor(0, dtype=torch.float32, device='cpu')
    metric_totals = torch.zeros(size=[len(metric_functions)], device='cpu')
    assert isinstance(dataloader.sampler, DistributedSampler)
    dataloader.sampler.set_epoch(cycle)
    batch_count = 0
    for batch, (parameters, light_curves) in enumerate(dataloader):
        parameters = parameters.to(network_device, non_blocking=True)
        light_curves = light_curves.to(loss_device, non_blocking=True)
        predicted_light_curves = model(parameters)
        loss, total_cycle_loss = record_metrics(predicted_light_curves, light_curves, loss_function, total_cycle_loss,
                                                metric_functions, metric_totals, loss_device)
        optimizer.zero_grad()
        loss.to(network_device, non_blocking=True).backward()
        optimizer.step()

        if batch % 1 == 0:
            current = (batch + 1) * len(parameters)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{len(dataloader.sampler):>5d}]", flush=True)
        batch_count += 1
    log_metrics(total_cycle_loss, metric_functions, metric_totals, '', batch_count, world_size, process_rank)


def record_metrics(predicted_light_curves, light_curves, loss_function, total_cycle_loss, metric_functions,
                   metric_totals, loss_device):
    loss = loss_function(predicted_light_curves.to(loss_device, non_blocking=True), light_curves)
    total_cycle_loss += loss.to('cpu', non_blocking=True)
    for metric_function_index, metric_function in enumerate(metric_functions):
        batch_metric_value = metric_function(predicted_light_curves.to(loss_device, non_blocking=True),
                                             light_curves)
        metric_totals[metric_function_index] += batch_metric_value.to('cpu', non_blocking=True)
    return loss, total_cycle_loss


def validation_phase(dataloader: DataLoader, model: Module, loss_function: Callable[[Tensor, Tensor], Tensor],
                     network_device: Device, loss_device: Device, cycle: int,
                     metric_functions: List[Callable[[Tensor, Tensor], Tensor]], process_rank: int, world_size: int
                     ) -> float:
    total_cycle_loss = tensor(0, dtype=torch.float32, device='cpu')
    metric_totals = torch.zeros(size=[len(metric_functions)], device='cpu')
    model.eval()
    assert isinstance(dataloader.sampler, DistributedSampler)
    dataloader.sampler.set_epoch(cycle)
    with torch.no_grad():
        batch_count = 0
        for parameters, light_curves in dataloader:
            parameters = parameters.to(network_device, non_blocking=True)
            light_curves = light_curves.to(loss_device, non_blocking=True)
            predicted_light_curves = model(parameters)
            loss, total_cycle_loss = record_metrics(predicted_light_curves, light_curves, loss_function, total_cycle_loss,
                                                    metric_functions, metric_totals, loss_device)
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
