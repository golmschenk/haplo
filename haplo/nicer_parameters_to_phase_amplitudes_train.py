import random

import numpy as np
import torch
from bokeh.io import show
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from bokeh.plotting import figure as Figure

from haplo.losses import PlusOneChiSquaredStatisticLoss
from haplo.models import LiraTraditionalShape8xWidthWith0d5DoNoBn
from haplo.nicer_dataset import NicerDataset, split_into_train_validation_and_test_datasets
from haplo.nicer_transform import PrecomputedNormalizeParameters, PrecomputedNormalizePhaseAmplitudes


def train_session():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    full_dataset = NicerDataset(parameters_transform=PrecomputedNormalizeParameters(),
                                phase_amplitudes_transform=PrecomputedNormalizePhaseAmplitudes())
    train_dataset, validation_dataset, test_dataset = split_into_train_validation_and_test_datasets(full_dataset)
    parameters0, phase_amplitudes0 = train_dataset[0]

    learning_rate = 1e-4
    batch_size = 1000
    epochs = 5000

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)


    model = LiraTraditionalShape8xWidthWith0d5DoNoBn()
    model = model.to(device)
    loss_function = PlusOneChiSquaredStatisticLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_function, optimizer, device=device)
        loop_test(validation_dataloader, model, loss_function, device=device)
        with torch.no_grad():
            X_, y_ = test_dataset[random.randrange(100)]
            pred_y_ = np.squeeze(model(torch.tensor(X_).to(device)).to('cpu').numpy())
            figure = Figure()
            figure.line(x=np.arange(64), y=y_, line_color='mediumblue')
            figure.line(x=np.arange(64), y=pred_y_, line_color='firebrick')
            show(figure)
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
    size = len(dataloader.dataset)
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
