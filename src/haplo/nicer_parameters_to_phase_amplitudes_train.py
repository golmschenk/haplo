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
from torch.optim import AdamW, Optimizer
from torch.types import Device
from torch.utils.data import DataLoader, DistributedSampler, Dataset

from haplo.losses import PlusOneChiSquaredStatisticMetric, PlusOneBeforeUnnormalizationChiSquaredStatisticMetric, \
    norm_based_gradient_clip
from haplo.models import Cura
from haplo.nicer_dataset import NicerDataset, split_dataset_into_count_datasets, split_dataset_into_fractional_datasets
from haplo.nicer_transform import PrecomputedNormalizeParameters, PrecomputedNormalizePhaseAmplitudes
from haplo.wandb_liaison import wandb_init, wandb_log, wandb_commit, \
    wandb_log_hyperparameter_dictionary


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


def default_train_session():
    train_dataset_path = Path('data/50m_rotated_parameters_and_phase_amplitudes.db')
    full_train_dataset = NicerDataset.new(
        dataset_path=train_dataset_path,
        parameters_transform=PrecomputedNormalizeParameters(),
        phase_amplitudes_transform=PrecomputedNormalizePhaseAmplitudes())
    test_dataset, validation_dataset, train_dataset = split_dataset_into_fractional_datasets(full_train_dataset,
                                                                                             [0.1, 0.1, 0.8])
    model = Cura()
    add_norm_based_gradient_clip_to_all_parameters(model)
    loss_function = PlusOneBeforeUnnormalizationChiSquaredStatisticMetric()
    metric_functions = [PlusOneChiSquaredStatisticMetric(), PlusOneBeforeUnnormalizationChiSquaredStatisticMetric()]
    learning_rate = 1e-4
    optimizer_epsilon = 1e-7
    weight_decay = 0.0001
    optimizer = AdamW(params=model.parameters(), weight_decay=weight_decay, lr=learning_rate, eps=optimizer_epsilon)
    batch_size_per_device = 100
    cycles_to_run = 5000
    model_name = type(model).__name__
    run_comments = f"pt"
    wandb_log_dictionary = {
        'model_name': model_name, 'learning_rate': learning_rate, 'batch_size_per_device': batch_size_per_device,
        'train_dataset_size': len(train_dataset), 'optimizer_epsilon': optimizer_epsilon, 'weight_decay': weight_decay,
        'run_comments': run_comments
    }
    train_session(train_dataset, validation_dataset, model, loss_function, metric_functions, optimizer,
                  batch_size_per_device, cycles_to_run, wandb_project='haplo', wandb_entity='ramjet',
                  wandb_log_dictionary=wandb_log_dictionary)


def train_session(train_dataset: Dataset, validation_dataset: Dataset, model: Module, loss_function: Module,
                  metric_functions: List[Module], optimizer: Optimizer, batch_size_per_device: int, cycles_to_run: int,
                  wandb_project: str, wandb_entity: str,
                  wandb_log_dictionary: Dict[str, Any] | None = None):
    if wandb_log_dictionary is None:
        wandb_log_dictionary = {}
    print('Starting training...')
    print('Starting process spawning...')
    torch.multiprocessing.set_start_method('spawn')
    print('Starting DDP setup...')
    ddp_setup()
    process_rank = int(os.environ['RANK'])
    print(f'{process_rank}: Starting wandb...')
    wandb_init(process_rank=process_rank, project=wandb_project, entity=wandb_entity,
               settings=wandb.Settings(start_method='fork'))
    wandb_log_hyperparameter_dictionary(wandb_log_dictionary, process_rank=process_rank)

    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    if torch.cuda.is_available():
        network_device = torch.device(f'cuda:{local_rank}')
        loss_device = network_device
    else:
        network_device = torch.device('cpu')
        loss_device = network_device

    print(f'{process_rank}: Moving model to device...')
    model = model.to(network_device, non_blocking=True)
    if torch.cuda.is_available():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    else:
        model = DistributedDataParallel(model)

    print(f'{process_rank}: Loading dataset...')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_per_device, num_workers=10, pin_memory=True,
                                  persistent_workers=True, prefetch_factor=10, shuffle=False,
                                  sampler=DistributedSampler(train_dataset))
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size_per_device, num_workers=10,
                                       pin_memory=True, persistent_workers=True, prefetch_factor=10, shuffle=False,
                                       sampler=DistributedSampler(validation_dataset))
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


def save_model(model: Module, suffix: str, process_rank: int):
    if process_rank == 0:
        model_name = wandb.run.name
        if model_name == '':
            model_name = wandb.run.id
        torch.save(model.state_dict(), Path(f'sessions/{model_name}_{suffix}.pt'))


def train_phase(dataloader: DataLoader, model: Module, loss_function: Callable[[Tensor, Tensor], Tensor],
                optimizer: Optimizer, network_device: Device, loss_device: Device, cycle: int,
                metric_functions: List[Callable[[Tensor, Tensor], Tensor]], process_rank: int, world_size: int):
    number_of_batches = len(dataloader)
    model.train()
    total_cycle_loss = tensor(0, dtype=torch.float32)
    metric_totals = torch.zeros(size=[len(metric_functions)])
    assert isinstance(dataloader.sampler, DistributedSampler)
    dataloader.sampler.set_epoch(cycle)
    for batch, (parameters, light_curves) in enumerate(dataloader):
        parameters = parameters.to(network_device, non_blocking=True)
        light_curves = light_curves.to(loss_device, non_blocking=True)
        predicted_light_curves = model(parameters)
        loss = loss_function(predicted_light_curves.to(loss_device, non_blocking=True), light_curves).to(network_device,
                                                                                                         non_blocking=True)
        for metric_function_index, metric_function in enumerate(metric_functions):
            batch_metric_value = metric_function(predicted_light_curves.to(loss_device, non_blocking=True),
                                                 light_curves)
            metric_totals[metric_function_index] += batch_metric_value.to('cpu', non_blocking=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_cycle_loss += loss.to('cpu', non_blocking=True)
        if batch % 1 == 0:
            current = (batch + 1) * len(parameters)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{len(dataloader.sampler):>5d}]", flush=True)
    log_metrics(total_cycle_loss, metric_functions, metric_totals, '', number_of_batches, world_size, process_rank)


def validation_phase(dataloader: DataLoader, model: Module, loss_function: Callable[[Tensor, Tensor], Tensor],
                     network_device: Device, loss_device: Device, cycle: int,
                     metric_functions: List[Callable[[Tensor, Tensor], Tensor]], process_rank: int, world_size: int
                     ) -> float:
    number_of_batches = len(dataloader)
    total_cycle_loss = tensor(0, dtype=torch.float32)
    metric_totals = torch.zeros(size=[len(metric_functions)])
    model.eval()
    assert isinstance(dataloader.sampler, DistributedSampler)
    dataloader.sampler.set_epoch(cycle)
    with torch.no_grad():
        for parameters, light_curves in dataloader:
            parameters = parameters.to(network_device, non_blocking=True)
            light_curves = light_curves.to(loss_device, non_blocking=True)
            predicted_light_curves = model(parameters)
            total_cycle_loss += loss_function(predicted_light_curves.to(loss_device, non_blocking=True), light_curves
                                              ).to('cpu', non_blocking=True)
            for metric_function_index, metric_function in enumerate(metric_functions):
                batch_metric_value = metric_function(predicted_light_curves.to(loss_device, non_blocking=True),
                                                     light_curves)
                metric_totals[metric_function_index] += batch_metric_value.to('cpu', non_blocking=True)

    cycle_loss = log_metrics(total_cycle_loss, metric_functions, metric_totals, 'val_', number_of_batches, world_size,
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
    for metric_function_index, metric_function in enumerate(metric_functions):
        cycle_metric_value = cycle_metric_values[metric_function_index]
        torch.distributed.reduce(cycle_metric_value, dst=0, op=ReduceOp.SUM)
        cycle_metric_value /= world_size
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


if __name__ == '__main__':
    default_train_session()
