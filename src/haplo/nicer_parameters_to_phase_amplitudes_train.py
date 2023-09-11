import multiprocessing
import os
from pathlib import Path
from typing import Callable, Tuple, List

import numpy as np
import stringcase
import torch
import wandb as wandb
from torch.distributed import init_process_group, destroy_process_group, Backend
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW, Optimizer
from torch.types import Device
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch import multiprocessing, Tensor

from haplo.data_paths import unrotated_dataset_path, move_path_to_nvme
from haplo.losses import PlusOneChiSquaredStatisticMetric, PlusOneBeforeUnnormalizationChiSquaredStatisticMetric, \
    norm_based_gradient_clip
from haplo.models import LiraTraditionalShape8xWidthWithNoDoNoBnOldFirstLayers
from haplo.nicer_dataset import NicerDataset, split_dataset_into_fractional_datasets
from haplo.nicer_transform import PrecomputedNormalizeParameters, PrecomputedNormalizePhaseAmplitudes
from haplo.wandb_liaison import wandb_set_run_name, wandb_init, wandb_log, wandb_commit


def ddp_setup():
    if torch.cuda.is_available():
        distributed_back_end = Backend.NCCL
    else:
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
    train_dataset_path = Path('data/50m_rotated_parameters_and_phase_amplitudes.arrow')
    full_train_dataset = NicerDataset.new(
        dataset_path=train_dataset_path,
        parameters_transform=PrecomputedNormalizeParameters(),
        phase_amplitudes_transform=PrecomputedNormalizePhaseAmplitudes())
    train_dataset, validation_dataset, test_dataset = split_dataset_into_fractional_datasets(full_train_dataset,
                                                                                             [0.8, 0.1, 0.1])
    model = LiraTraditionalShape8xWidthWithNoDoNoBnOldFirstLayers()
    for parameter in model.parameters():
        parameter.register_hook(norm_based_gradient_clip)
    loss_function = PlusOneBeforeUnnormalizationChiSquaredStatisticMetric()
    metric_functions = [PlusOneChiSquaredStatisticMetric(), PlusOneBeforeUnnormalizationChiSquaredStatisticMetric()]
    learning_rate = 1e-4
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001, eps=1e-7)
    batch_size_per_device = 100
    cycles_to_run = 5000
    model_name = type(model).__name__
    run_name = f"{model_name}_old_chi_squared_loss_shuffled_50m_dataloader_shuffled_bs_{batch_size_per_device}" \
               f"_copy_on_transform_train_and_val_from_same_corrected2_val_calc_adamw_grad_norm_clip_1_node" \
               f"_spawn_w3"
    train_session(train_dataset, validation_dataset, model, loss_function, metric_functions, optimizer,
                  batch_size_per_device, cycles_to_run, run_name)


def train_session(train_dataset: Dataset, validation_dataset: Dataset, model: Module, loss_function: Module,
                  metric_functions: List[Module], optimizer: Optimizer, batch_size_per_device: int, cycles_to_run: int,
                  run_name: str):
    torch.multiprocessing.set_start_method('spawn')
    ddp_setup()
    process_rank = int(os.environ['RANK'])
    wandb_init(process_rank=process_rank, project='haplo', entity='ramjet',
               settings=wandb.Settings(start_method='fork'))
    cpu_count = multiprocessing.cpu_count()
    gpu_count = torch.cuda.device_count()
    print(f'GPUs: {gpu_count}')

    if torch.cuda.is_available():
        local_rank = int(os.environ['LOCAL_RANK'])
        network_device = torch.device(f'cuda:{local_rank}')
        loss_device = network_device
    else:
        network_device = torch.device('cpu')
        loss_device = network_device

    model = model.to(network_device, non_blocking=True)
    if torch.cuda.is_available():
        local_rank = int(os.environ['LOCAL_RANK'])
        model = DistributedDataParallel(model, device_ids=[local_rank])
    else:
        model = DistributedDataParallel(model)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_per_device, num_workers=3, pin_memory=True,
                                  persistent_workers=True, prefetch_factor=10, shuffle=False,
                                  sampler=DistributedSampler(train_dataset))
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size_per_device, num_workers=3,
                                       pin_memory=True, persistent_workers=True, prefetch_factor=10, shuffle=False,
                                       sampler=DistributedSampler(validation_dataset))

    wandb_set_run_name(run_name, process_rank=process_rank)

    for cycle in range(cycles_to_run):
        print(f"Epoch {cycle}\n-------------------------------")
        train_loop(train_dataloader, model, loss_function, optimizer, network_device=network_device,
                   loss_device=loss_device, cycle=cycle, metric_functions=metric_functions, process_rank=process_rank)
        loop_test(validation_dataloader, model, loss_function, network_device=network_device, loss_device=loss_device,
                  cycle=cycle, metric_functions=metric_functions, process_rank=process_rank)
        save_model(model, process_rank=process_rank)
        wandb_log('epoch', cycle, process_rank=process_rank)
        wandb_log('cycle', cycle, process_rank=process_rank)
        wandb_commit(process_rank=process_rank)
    print("Done!")

    destroy_process_group()


def save_model(model: Module, process_rank: int):
    if process_rank == 0:
        torch.save(model.state_dict(), Path(f'sessions/{wandb.run.id}_latest_model.pt'))


def train_loop(dataloader: DataLoader, model: Module, loss_function: Callable[[Tensor, Tensor], Tensor],
               optimizer: Optimizer, network_device: Device, loss_device: Device, cycle: int,
               metric_functions: List[Callable[[Tensor, Tensor], Tensor]], process_rank: int):
    number_of_batches = len(dataloader)
    model.train()
    total_cycle_loss = 0
    metric_totals = np.zeros(shape=[len(metric_functions)])
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
                                                 light_curves).item()
            metric_totals[metric_function_index] += batch_metric_value
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss_value, current = loss.item(), (batch + 1) * len(parameters)
            total_cycle_loss += loss_value
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{len(dataloader.sampler):>5d}]", flush=True)
    wandb_log('loss', total_cycle_loss / number_of_batches, process_rank=process_rank)
    cycle_metric_values = metric_totals / number_of_batches
    for metric_function_index, metric_function in enumerate(metric_functions):
        wandb_log(f'{get_metric_name(metric_function)}', cycle_metric_values[metric_function_index],
                  process_rank=process_rank)


def get_metric_name(metric_function):
    metric_name = type(metric_function).__name__
    metric_name = stringcase.snakecase(metric_name)
    metric_name = metric_name.replace('_metric', '').replace('_loss', '')
    return metric_name


def loop_test(dataloader: DataLoader, model: Module, loss_function: Callable[[Tensor, Tensor], Tensor],
              network_device: Device, loss_device: Device, cycle: int,
              metric_functions: List[Callable[[Tensor, Tensor], Tensor]], process_rank: int):
    number_of_batches = len(dataloader)
    total_cycle_loss = 0
    metric_totals = np.zeros(shape=[len(metric_functions)])
    model.eval()
    assert isinstance(dataloader.sampler, DistributedSampler)
    dataloader.sampler.set_epoch(cycle)
    with torch.no_grad():
        for parameters, light_curves in dataloader:
            parameters = parameters.to(network_device, non_blocking=True)
            light_curves = light_curves.to(loss_device, non_blocking=True)
            predicted_light_curves = model(parameters)
            total_cycle_loss += loss_function(predicted_light_curves.to(loss_device, non_blocking=True), light_curves
                                              ).to(network_device, non_blocking=True).item()
            for metric_function_index, metric_function in enumerate(metric_functions):
                batch_metric_value = metric_function(predicted_light_curves.to(loss_device, non_blocking=True),
                                                     light_curves).item()
                metric_totals[metric_function_index] += batch_metric_value

    cycle_loss = total_cycle_loss / number_of_batches
    print(f"Test Error: \nAvg loss: {cycle_loss:>8f} \n", flush=True)
    wandb_log('val_plus_one_chi_squared_statistic', cycle_loss, process_rank=process_rank)
    cycle_metric_values = metric_totals / number_of_batches
    for metric_function_index, metric_function in enumerate(metric_functions):
        wandb_log(f'val_{get_metric_name(metric_function)}', cycle_metric_values[metric_function_index],
                  process_rank=process_rank)


if __name__ == '__main__':
    default_train_session()
