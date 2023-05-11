import torch
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader

from haplo.losses import PlusOneChiSquaredStatisticLoss
from haplo.models import LiraTraditionalShape8xWidthWith0d5DoNoBn
from haplo.nicer_dataset import NicerDataset, split_into_train_validation_and_test_datasets
from haplo.nicer_transform import PrecomputedNormalizeParameters, PrecomputedNormalizePhaseAmplitudes

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
test_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

model = LiraTraditionalShape8xWidthWith0d5DoNoBn()
model = model.to(device)
loss_function = PlusOneChiSquaredStatisticLoss()
loss_function = loss_function.to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model_, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model_(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def loop_test(dataloader, model_: Module, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model_(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error: \nAvg loss: {test_loss:>8f} \n")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_function, optimizer)
    loop_test(test_dataloader, model, loss_function)
print("Done!")
