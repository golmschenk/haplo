import multiprocessing

import torch
import wandb as wandb
from torch.nn import Module, DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader

from haplo.data_paths import rotated_dataset_path, unrotated_dataset_path, move_path_to_nvme
from haplo.losses import PlusOneChiSquaredStatisticLoss
from haplo.models import LiraTraditionalShape8xWidthWithNoDoNoBn
from haplo.nicer_dataset import NicerDataset, split_dataset_into_fractional_datasets
from haplo.nicer_transform import PrecomputedNormalizeParameters, PrecomputedNormalizePhaseAmplitudes


def train_session():
    wandb.init(project='haplo', entity='ramjet', settings=wandb.Settings(start_method='fork'))
    cpu_count = multiprocessing.cpu_count()
    gpu_count = torch.cuda.device_count()
    print(f'GPUs: {gpu_count}')

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    rotated_dataset_path_moved = move_path_to_nvme(rotated_dataset_path)
    unrotated_dataset_path_moved = move_path_to_nvme(unrotated_dataset_path)
    train_dataset = NicerDataset(
        dataset_path=rotated_dataset_path_moved,
        parameters_transform=PrecomputedNormalizeParameters(),
        phase_amplitudes_transform=PrecomputedNormalizePhaseAmplitudes())
    evaluation_dataset = NicerDataset(
        dataset_path=unrotated_dataset_path_moved,
        parameters_transform=PrecomputedNormalizeParameters(),
        phase_amplitudes_transform=PrecomputedNormalizePhaseAmplitudes())
    validation_dataset, test_dataset = split_dataset_into_fractional_datasets(evaluation_dataset, [0.5, 0.5])

    learning_rate = 1e-5
    if gpu_count == 0:
        batch_size = 1000
    else:
        batch_size = 1000 * gpu_count
    epochs = 5000

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=cpu_count)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=cpu_count)

    model = LiraTraditionalShape8xWidthWithNoDoNoBn()
    if gpu_count > 1:
        model = DataParallel(model)
    model = model.to(device)
    clip_value = 1.0
    for parameter in model.parameters():
        parameter.register_hook(lambda gradient: torch.clamp(gradient, -clip_value, clip_value))
    loss_function = PlusOneChiSquaredStatisticLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    wandb.run.notes = f"pt_{type(model).__name__}_chi_squared_loss_64m_clip_lr_1e-5"

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_function, optimizer, device=device, epoch=t)
        loop_test(validation_dataloader, model, loss_function, device=device, epoch=t)
        torch.save(model.state_dict(), 'latest_model.pt')
    print("Done!")


def train_loop(dataloader, model_, loss_fn, optimizer, device, epoch):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y
        pred = model_(X)
        loss = loss_fn(pred.to('cpu'), y).to(device)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss_value, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]", flush=True)
    wandb.log({'plus_one_chi_squared_statistic': loss.item()}, step=epoch)
    wandb.log({'epoch': epoch}, step=epoch)


def loop_test(dataloader, model_: Module, loss_fn, device, epoch):
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y
            pred = model_(X)
            test_loss += loss_fn(pred.to('cpu'), y).to(device).item()

    test_loss /= num_batches
    print(f"Test Error: \nAvg loss: {test_loss:>8f} \n", flush=True)
    wandb.log({'val_plus_one_chi_squared_statistic': test_loss}, step=epoch)


if __name__ == '__main__':
    train_session()
