import torch
from torch.nn import Module, DataParallel
from torch.utils.data import DataLoader

from haplo.data_paths import rotated_dataset_path, unrotated_dataset_path
from haplo.losses import PlusOneChiSquaredStatisticMetric
from haplo.models import LiraTraditionalShape8xWidthWithNoDoNoBn
from haplo.nicer_dataset import NicerDataset, split_dataset_into_fractional_datasets
from haplo.nicer_transform import PrecomputedNormalizeParameters, PrecomputedNormalizePhaseAmplitudes


def infer_session():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    evaluation_dataset = NicerDataset.new(
        dataset_path=unrotated_dataset_path,
        parameters_transform=PrecomputedNormalizeParameters(),
        phase_amplitudes_transform=PrecomputedNormalizePhaseAmplitudes())
    validation_dataset, test_dataset = split_dataset_into_fractional_datasets(evaluation_dataset, [0.5, 0.5])

    batch_size = 1000

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = DataParallel(LiraTraditionalShape8xWidthWithNoDoNoBn())
    model = model.to(device)
    model.load_state_dict(torch.load('latest_model.pt'))
    model.eval()
    loss_function = PlusOneChiSquaredStatisticMetric()

    loop_test(test_dataloader, model, loss_function, device=device)


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
