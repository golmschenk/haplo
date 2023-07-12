import torch
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader

from haplo.data_paths import rotated_dataset_path, unrotated_dataset_path
from haplo.losses import PlusOneChiSquaredStatisticLoss
from haplo.models import LiraTraditionalShape8xWidthWithNoDoNoBn
from haplo.nicer_dataset import NicerDataset, split_dataset_into_fractional_datasets
from haplo.nicer_transform import PrecomputedNormalizeParameters, PrecomputedNormalizePhaseAmplitudes


def train_session():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_dataset = NicerDataset(
        dataset_path=rotated_dataset_path,
        parameters_transform=PrecomputedNormalizeParameters(),
        phase_amplitudes_transform=PrecomputedNormalizePhaseAmplitudes())
    evaluation_dataset = NicerDataset(
        dataset_path=unrotated_dataset_path,
        parameters_transform=PrecomputedNormalizeParameters(),
        phase_amplitudes_transform=PrecomputedNormalizePhaseAmplitudes())
    validation_dataset, test_dataset = split_dataset_into_fractional_datasets(evaluation_dataset, [0.5, 0.5])

    learning_rate = 1e-4
    batch_size = 1000
    epochs = 5000

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    model = LiraTraditionalShape8xWidthWithNoDoNoBn()
    model = model.to(device)
    loss_function = PlusOneChiSquaredStatisticLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_function, optimizer, device=device)
        loop_test(validation_dataloader, model, loss_function, device=device)
        torch.save(model.state_dict(), 'latest_model.pt')
    print("Done!")


def train_loop(dataloader, model_, loss_fn, optimizer, device):
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
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def loop_test(dataloader, model_: Module, loss_fn, device):
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y
            pred = model_(X)
            test_loss += loss_fn(pred.to('cpu'), y).to(device).item()

    test_loss /= num_batches
    print(f"Test Error: \nAvg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    train_session()
