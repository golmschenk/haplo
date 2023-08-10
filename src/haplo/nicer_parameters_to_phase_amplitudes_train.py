from pathlib import Path

import torch

import multiprocessing

import wandb as wandb
from torch.nn import Module, DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader

from haplo.data_paths import rotated_dataset_path, unrotated_dataset_path, move_path_to_nvme
from haplo.losses import PlusOneChiSquaredStatisticLoss, PlusOneBeforeUnnormalizationChiSquaredStatisticLoss
from haplo.models import LiraTraditionalShape8xWidthWithNoDoNoBn, LiraTraditionalShape8xWidthWithNoDoNoBnOldFirstLayers, \
    LiraTraditionalShape8xWidthWith0d5DoNoBnOldFirstLayers
from haplo.nicer_dataset import NicerDataset, split_dataset_into_count_datasets, split_dataset_into_fractional_datasets
from haplo.nicer_transform import PrecomputedNormalizeParameters, PrecomputedNormalizePhaseAmplitudes


def train_session():
    torch.multiprocessing.set_start_method('spawn')
    wandb.init(project='haplo', entity='ramjet', settings=wandb.Settings(start_method='fork'))
    cpu_count = multiprocessing.cpu_count()
    gpu_count = torch.cuda.device_count()
    print(f'GPUs: {gpu_count}')

    if torch.cuda.is_available():
        network_device = torch.device('cuda')
        loss_device = network_device
    else:
        network_device = torch.device('cpu')
        loss_device = network_device

    train_dataset_path = Path('data/50m_unshuffled_rotated_parameters_and_phase_amplitudes.arrow')
    evaluation_dataset_path = unrotated_dataset_path
    train_dataset_path_moved = move_path_to_nvme(train_dataset_path)
    evaluation_dataset_path_moved = move_path_to_nvme(evaluation_dataset_path)
    full_train_dataset = NicerDataset.new(
        dataset_path=train_dataset_path_moved,
        parameters_transform=PrecomputedNormalizeParameters(),
        phase_amplitudes_transform=PrecomputedNormalizePhaseAmplitudes())
    train_dataset, validation_dataset, test_dataset = split_dataset_into_fractional_datasets(full_train_dataset, [0.8, 0.1, 0.1])
    # evaluation_dataset = NicerDataset.new(
    #     dataset_path=evaluation_dataset_path_moved,
    #     parameters_transform=PrecomputedNormalizeParameters(),
    #     phase_amplitudes_transform=PrecomputedNormalizePhaseAmplitudes())
    # validation_dataset, test_dataset = split_dataset_into_fractional_datasets(evaluation_dataset, [0.5, 0.5])

    batch_size_per_gpu = 100
    learning_rate = 1e-3
    if gpu_count == 0:
        batch_size = batch_size_per_gpu
    else:
        batch_size = batch_size_per_gpu * gpu_count
    epochs = 5000

    model = LiraTraditionalShape8xWidthWith0d5DoNoBnOldFirstLayers()
    # def init_weights(m):
    #     if hasattr(m, 'weights'):
    #         torch.nn.init.normal_(m.weights, std=0.00001)
    # model.apply(init_weights)
    model_name = type(model).__name__
    if gpu_count > 1:
        model = DataParallel(model)
    model = model.to(network_device, non_blocking=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=6, pin_memory=True,
                                  persistent_workers=True, prefetch_factor=10)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=6,
                                       pin_memory=True, persistent_workers=True, prefetch_factor=10)

    # clip_value = 1.0
    # for parameter in model.parameters():
    #     parameter.register_hook(lambda gradient: torch.clamp(gradient, -clip_value, clip_value))
    loss_function = PlusOneBeforeUnnormalizationChiSquaredStatisticLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

    wandb.run.notes = f"{model_name}_chi_squared_loss_50m_bs_{batch_size}_unshuffled_copy_on_transform_train_and_val_from_50m_corrected2_val_calc_lr_1e-3"

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_function, optimizer, network_device=network_device,
                   loss_device=loss_device, epoch=epoch)
        loop_test(validation_dataloader, model, loss_function, network_device=network_device,
                  loss_device=loss_device,
                  epoch=epoch)
        
        torch.save(model.state_dict(), Path(f'sessions/{wandb.run.id}_latest_model.pt'))
        wandb.log({'epoch': epoch}, commit=False)
        wandb.log({'cycle': epoch}, commit=False)
        wandb.log({}, commit=True)
    print("Done!")


def train_loop(dataloader, model_, loss_fn, optimizer, network_device, loss_device, epoch):
    number_of_observations = len(dataloader)
    model_.train()
    total_cycle_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(network_device, non_blocking=True)
        y = y.to(loss_device, non_blocking=True)
        pred = model_(X)
        loss = loss_fn(pred.to(loss_device, non_blocking=True), y).to(network_device, non_blocking=True)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model_.parameters(), max_norm=1.0)
        optimizer.step()

        if batch % 1 == 0:
            loss_value, current = loss.item(), (batch + 1) * len(X)
            total_cycle_loss += loss_value
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]", flush=True)
    wandb.log({'plus_one_chi_squared_statistic': total_cycle_loss / number_of_observations}, commit=False)


def loop_test(dataloader, model_: Module, loss_fn, network_device, loss_device, epoch):
    number_of_observations = len(dataloader)
    test_loss, correct = 0, 0
    model_.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(network_device, non_blocking=True)
            y = y.to(loss_device, non_blocking=True)
            pred = model_(X)
            test_loss += loss_fn(pred.to(loss_device, non_blocking=True), y).to(network_device,
                                                                                non_blocking=True).item()

    test_loss /= number_of_observations
    print(f"Test Error: \nAvg loss: {test_loss:>8f} \n", flush=True)
    wandb.log({'val_plus_one_chi_squared_statistic': test_loss}, commit=False)


if __name__ == '__main__':
    train_session()
