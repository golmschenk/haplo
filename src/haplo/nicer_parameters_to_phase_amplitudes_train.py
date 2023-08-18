import multiprocessing
import os
from pathlib import Path

import numpy as np
import stringcase
import torch
import wandb as wandb
from torch.distributed import init_process_group, destroy_process_group
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torch import multiprocessing

from haplo.data_paths import rotated_dataset_path, unrotated_dataset_path, move_path_to_nvme
from haplo.losses import PlusOneChiSquaredStatisticMetric, PlusOneBeforeUnnormalizationChiSquaredStatisticMetric, \
    norm_based_gradient_clip
from haplo.models import LiraTraditionalShape8xWidthWithNoDoNoBn, LiraTraditionalShape8xWidthWithNoDoNoBnOldFirstLayers, \
    LiraTraditionalShape8xWidthWith0d5DoNoBnOldFirstLayers
from haplo.nicer_dataset import NicerDataset, split_dataset_into_count_datasets, split_dataset_into_fractional_datasets
from haplo.nicer_transform import PrecomputedNormalizeParameters, PrecomputedNormalizePhaseAmplitudes


def ddp_setup(rank: int, world_size: int, distributed_back_end: str):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = os.environ["SLURMD_NODENAME"]
    os.environ["MASTER_PORT"] = "57392"
    init_process_group(backend=distributed_back_end, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def train_session(process_rank, process_world_size, distributed_back_end):
    ddp_setup(process_rank, process_world_size, distributed_back_end=distributed_back_end)
    wandb.init(project='haplo', entity='ramjet', settings=wandb.Settings(start_method='fork'))
    cpu_count = multiprocessing.cpu_count()
    gpu_count = torch.cuda.device_count()
    print(f'GPUs: {gpu_count}')

    if torch.cuda.is_available():
        network_device = torch.device(f'cuda:{process_rank}')
        loss_device = network_device
    else:
        network_device = torch.device('cpu')
        loss_device = network_device

    train_dataset_path = Path('data/50m_rotated_parameters_and_phase_amplitudes.arrow')
    evaluation_dataset_path = unrotated_dataset_path
    train_dataset_path_moved = move_path_to_nvme(train_dataset_path)
    evaluation_dataset_path_moved = move_path_to_nvme(evaluation_dataset_path)
    full_train_dataset = NicerDataset.new(
        dataset_path=train_dataset_path,
        parameters_transform=PrecomputedNormalizeParameters(),
        phase_amplitudes_transform=PrecomputedNormalizePhaseAmplitudes())
    train_dataset, validation_dataset, test_dataset = split_dataset_into_fractional_datasets(full_train_dataset,
                                                                                             [0.8, 0.1, 0.1])
    # evaluation_dataset = NicerDataset.new(
    #     dataset_path=evaluation_dataset_path_moved,
    #     parameters_transform=PrecomputedNormalizeParameters(),
    #     phase_amplitudes_transform=PrecomputedNormalizePhaseAmplitudes())
    # validation_dataset, test_dataset = split_dataset_into_fractional_datasets(evaluation_dataset, [0.5, 0.5])

    batch_size_per_gpu = 100
    learning_rate = 1e-4
    if gpu_count == 0:
        batch_size = batch_size_per_gpu
    else:
        batch_size = batch_size_per_gpu * gpu_count
    epochs = 5000

    model = LiraTraditionalShape8xWidthWithNoDoNoBnOldFirstLayers()
    # def init_weights(m):
    #     if hasattr(m, 'weights'):
    #         torch.nn.init.normal_(m.weights, std=0.00001)
    # model.apply(init_weights)
    model_name = type(model).__name__
    model = model.to(network_device, non_blocking=True)
    if torch.cuda.is_available():
        model = DistributedDataParallel(model, device_ids=[process_rank])
    else:
        model = DistributedDataParallel(model)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, pin_memory=True,
                                  persistent_workers=True, prefetch_factor=10, shuffle=False,
                                  sampler=DistributedSampler(train_dataset))
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=2,
                                       pin_memory=True, persistent_workers=True, prefetch_factor=10, shuffle=False,
                                       sampler=DistributedSampler(validation_dataset))

    for parameter in model.parameters():
        parameter.register_hook(norm_based_gradient_clip)
    loss_function = PlusOneBeforeUnnormalizationChiSquaredStatisticMetric()
    metric_functions = [PlusOneChiSquaredStatisticMetric(), PlusOneBeforeUnnormalizationChiSquaredStatisticMetric()]
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001, eps=1e-7)

    wandb.run.notes = f"{model_name}_old_chi_squared_loss_shuffled_50m_dataloader_shuffled_bs_{batch_size}_copy_on_transform_train_and_val_from_same_corrected2_val_calc_adamw_grad_norm_clip"

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_function, optimizer, network_device=network_device,
                   loss_device=loss_device, epoch=epoch, metric_functions=metric_functions)
        loop_test(validation_dataloader, model, loss_function, network_device=network_device,
                  loss_device=loss_device,
                  epoch=epoch, metric_functions=metric_functions)

        torch.save(model.state_dict(), Path(f'sessions/{wandb.run.id}_latest_model.pt'))
        wandb.log({'epoch': epoch}, commit=False)
        wandb.log({'cycle': epoch}, commit=False)
        wandb.log({}, commit=True)
    print("Done!")

    destroy_process_group()


def train_loop(dataloader, model_, loss_fn, optimizer, network_device, loss_device, epoch, metric_functions):
    number_of_batches = len(dataloader)
    model_.train()
    total_cycle_loss = 0
    metric_totals = np.zeros(shape=[len(metric_functions)])
    dataloader.sampler.set_epoch(epoch)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(network_device, non_blocking=True)
        y = y.to(loss_device, non_blocking=True)
        pred = model_(X)
        loss = loss_fn(pred.to(loss_device, non_blocking=True), y).to(network_device, non_blocking=True)
        for metric_function_index, metric_function in enumerate(metric_functions):
            batch_metric_value = metric_function(pred.to(loss_device, non_blocking=True), y).item()
            metric_totals[metric_function_index] += batch_metric_value
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model_.parameters(), max_norm=100.0)
        optimizer.step()

        if batch % 1 == 0:
            loss_value, current = loss.item(), (batch + 1) * len(X)
            total_cycle_loss += loss_value
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]", flush=True)
    wandb.log({'loss': total_cycle_loss / number_of_batches}, commit=False)
    cycle_metric_values = metric_totals / number_of_batches
    for metric_function_index, metric_function in enumerate(metric_functions):
        wandb.log({f'{get_metric_name(metric_function)}': cycle_metric_values[metric_function_index]}, commit=False)


def get_metric_name(metric_function):
    metric_name = type(metric_function).__name__
    metric_name = stringcase.snakecase(metric_name)
    metric_name = metric_name.replace('_metric', '').replace('_loss', '')
    return metric_name


def loop_test(dataloader, model_: Module, loss_fn, network_device, loss_device, epoch, metric_functions):
    number_of_batches = len(dataloader)
    test_loss, correct = 0, 0
    metric_totals = np.zeros(shape=[len(metric_functions)])
    model_.eval()
    dataloader.sampler.set_epoch(epoch)
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(network_device, non_blocking=True)
            y = y.to(loss_device, non_blocking=True)
            pred = model_(X)
            test_loss += loss_fn(pred.to(loss_device, non_blocking=True), y).to(network_device,
                                                                                non_blocking=True).item()
            for metric_function_index, metric_function in enumerate(metric_functions):
                batch_metric_value = metric_function(pred.to(loss_device, non_blocking=True), y).item()
                metric_totals[metric_function_index] += batch_metric_value

    test_loss /= number_of_batches
    print(f"Test Error: \nAvg loss: {test_loss:>8f} \n", flush=True)
    wandb.log({'val_plus_one_chi_squared_statistic': test_loss}, commit=False)
    cycle_metric_values = metric_totals / number_of_batches
    for metric_function_index, metric_function in enumerate(metric_functions):
        wandb.log({f'val_{get_metric_name(metric_function)}': cycle_metric_values[metric_function_index]}, commit=False)


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    distributed_back_end = 'nccl'
    if world_size == 0:
        world_size = 1
        distributed_back_end = 'gloo'
    multiprocessing.spawn(train_session, args=(world_size, distributed_back_end), nprocs=world_size, join=True)
    pass