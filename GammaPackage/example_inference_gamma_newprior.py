from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from torch.nn import Module
from torch.optim import AdamW

from haplo.distributed import distributed_logging
from haplo.nicer_dataset import split_dataset_into_count_datasets

import os
import torch
from cura_2D_model import Cura2D

from GammaSkymapDataset import GammaSkymapDataset

import astropy.io.fits as fits

# Define the input directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file_name = 'J0030+0451_3PC_data.fits'
thisFile = os.path.join(script_dir, data_file_name)

with fits.open(thisFile) as f:
    dataphmin = f["GAMMA_LC"].data["Ph_Min"]
    dataphmax = f["GAMMA_LC"].data["Ph_Max"]

middle = (dataphmin + dataphmax) / 2.0
def convert2Dskymapto1Dlightcurve(skymap,theta_input):
    from scipy.interpolate import RegularGridInterpolator

    x_grid = np.linspace(0.0314, 6.2518, 100) / (2 * np.pi)

    y_grid = np.linspace(-0.99, 0.99, 100)

    # Create the interpolator
    interpolator = RegularGridInterpolator((y_grid, x_grid), skymap)

    def interpolate_z_for_lightcurve(x_list, y_value):
        #print(x_list,y_value)
        points = np.array([(-y_value,x ) for x in x_list])
        z_values = interpolator(points)
        return z_values

    phase_list = middle[:200]  # List of x values

    #theta_input = np.cos(np.pi / 4)  # Example constant y value, convert theta to y

    theta_input = np.cos(theta_input)

    # avoid extrapolating (grid is dense enough that it's not a problem):
    if theta_input < y_grid[0]:
        theta_input = y_grid[0]

    if theta_input > y_grid[-1]:
        theta_input = y_grid[-1]

    z_predicted = interpolate_z_for_lightcurve(phase_list[3:-3], theta_input)

    # Values to append to avoid extrapolating
    prepend_values = np.array(3 * [z_predicted[0]])
    append_values = np.array(3 * [z_predicted[-1]])

    # Append values to the start and end
    interp_zvalues =  np.concatenate((prepend_values, z_predicted, append_values))

    return interp_zvalues


@distributed_logging
def example_infer_session():
    os.environ["WANDB_MODE"] = "offline"

    # No normalization, 4x aug, v3
    saved_model_path = Path(os.path.join(script_dir,"lowest_validation_model.pt"))


    dataset_file_path = os.path.join(script_dir, "Gamma_halfplane_v3.csv")

    dataset = GammaSkymapDataset(dataset_file_path,randomDataAugmentation=False,loadLimit=500,normalize=False) # Load in the data

    test_dataset, validation_dataset, train_dataset, _ = split_dataset_into_count_datasets(
        dataset, [200,1,1])

    # TODO: can replace this with haplo 'unwrap model'
    # Load the state_dict from the file
    state_dict = torch.load(saved_model_path, map_location=torch.device('cpu'))

    # Remove the 'module.' prefix from the state_dict keys
    new_state_dict = {}
    for key in state_dict.keys():
        # Remove the 'module.' prefix if it exists
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = state_dict[key]

    model = Cura2D(input_features=11)

    model.load_state_dict(new_state_dict)

    model.eval()

    counter = 0
    while True:
        fig, axes = plt.subplots(3, 5, figsize=(15, 8))

        for i in range(5):
            test_parameters0, test_phase_amplitudes0 = test_dataset[i+counter]

            input_array = np.expand_dims(test_parameters0, axis=0)
            with torch.no_grad():
                input_tensor = torch.tensor(input_array)
                output_tensor = model(input_tensor)
                output_array = output_tensor.numpy()
            predicted_test_phase_amplitudes0 = np.squeeze(output_array, axis=0)

            counter += 1

            # Process the skymaps:

            ground_truth = np.flipud((np.array(test_phase_amplitudes0).reshape(100, 100).transpose()))
            predicted = np.flipud((np.array(predicted_test_phase_amplitudes0).reshape(100, 100).transpose()))

            # Display ground truth image
            im1 = axes[0,i].imshow(ground_truth, cmap='viridis')

            # Display predicted image
            im2 = axes[1,i].imshow(predicted, cmap='viridis')

            # Show a light curve example
            # Keep trying random thetas until we find one that gives a bright enough light curve
            while True:
                theta = np.random.uniform(0, np.pi/2)
                ground_truth_lightcurve = convert2Dskymapto1Dlightcurve(ground_truth, theta)

                # Check if light curve is above 10% of skymap max
                if np.max(ground_truth_lightcurve) > 0.10 * np.max(ground_truth):
                    break

            predicted_lightcurve = convert2Dskymapto1Dlightcurve(predicted, theta)

            ground_truth_lightcurve = (ground_truth_lightcurve - np.min(ground_truth_lightcurve)) / (np.max(ground_truth_lightcurve) - np.min(ground_truth_lightcurve))
            predicted_lightcurve = (predicted_lightcurve - np.min(predicted_lightcurve)) / (np.max(predicted_lightcurve) - np.min(predicted_lightcurve))

            # ground_truth_lightcurve = (ground_truth_lightcurve - np.min(ground_truth)) / (np.max(ground_truth) - np.min(ground_truth))
            # predicted_lightcurve = (predicted_lightcurve - np.min(predicted)) / (np.max(predicted) - np.min(predicted))

            # Plot light curves in the bottom two rows
            axes[2,i].plot(ground_truth_lightcurve, color="tab:blue", label="Ground truth")
            axes[2,i].set_title(f"Random light curve (θ={theta:.2f})")
            axes[2,i].plot(predicted_lightcurve, color="tab:orange", label="Predicted")
            axes[2,i].legend()

        axes[0,2].set_title("Ground truth")
        axes[1,2].set_title("Predicted")

        # Adjust the subplot spacing for the new layout
        fig.subplots_adjust(left=0.04, right=0.95, top=0.95, bottom=0.04, hspace=0.4, wspace=0.3)

        plt.show()




if __name__ == '__main__':
    example_infer_session()
