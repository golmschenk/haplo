import logging
from pathlib import Path

import torch
from bokeh.io import show
from bokeh.models import Column
from bokeh.plotting import figure as Figure
from torch import Tensor
from torch.nn import Module, DataParallel
from torch.utils.data import DataLoader

from haplo.logging import set_up_default_logger
from haplo.losses import PlusOneChiSquaredStatisticMetric
from haplo.models import Cura
from haplo.nicer_dataset import NicerDataset, split_dataset_into_fractional_datasets
from haplo.nicer_transform import PrecomputedNormalizeParameters, PrecomputedNormalizePhaseAmplitudes

logger = logging.getLogger(__name__)


def infer_session(dataset_path: Path):
    set_up_default_logger()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    evaluation_dataset = NicerDataset.new(
        dataset_path=dataset_path,
        parameters_transform=PrecomputedNormalizeParameters(),
        phase_amplitudes_transform=PrecomputedNormalizePhaseAmplitudes())
    validation_dataset, test_dataset = split_dataset_into_fractional_datasets(evaluation_dataset, [0.5, 0.5])

    batch_size = 100

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = DataParallel(Cura())
    model = model.to(device)
    model.load_state_dict(torch.load('sessions/rural-totem-623_lowest_validation_model.pt', map_location=device))
    model.eval()
    loss_function = PlusOneChiSquaredStatisticMetric()

    phase_test(test_dataloader, model, loss_function, device=device)


def phase_test(dataloader, model_: Module, loss_fn, device):
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch_index, (inputs_tensor, targets) in enumerate(dataloader):
            logger.info(f'Processed {batch_index * dataloader.batch_size} examples.')
            inputs_tensor = inputs_tensor.to(device)
            targets = targets
            predicted_targets = model_(inputs_tensor)
            test_loss += loss_fn(predicted_targets.to('cpu'), targets).to(device).item()
            pass

    test_loss /= num_batches
    logger.info(f"Test Error: \nAvg loss: {test_loss:>8f} \n")


def quick_view(predicted_light_curve: Tensor, light_curve: Tensor) -> Figure:
    figure = Figure()
    predicted_light_curve_array = predicted_light_curve.to('cpu').numpy()
    light_curve_array = light_curve.to('cpu').numpy()
    figure.line(x=list(range(len(predicted_light_curve_array))), y=predicted_light_curve_array, line_color='firebrick')
    figure.line(x=list(range(len(light_curve_array))), y=light_curve_array, line_color='mediumblue')
    return figure


def show_all(x_list, y_list):
    column_contents = []
    for x, y in zip(x_list, y_list):
        column_contents.append(quick_view(x, y))
    column = Column(*column_contents)
    show(column)




if __name__ == '__main__':
    dataset_path = Path('data/800k_parameters_and_phase_amplitudes.db')
    infer_session(dataset_path)
