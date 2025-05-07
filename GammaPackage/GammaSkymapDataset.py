import csv
import random

import numpy as np
import torch
from torch.utils.data import Dataset

import Gamma_Normalization


class GammaSkymapDataset(Dataset):
    def __init__(self, csv_file, randomDataAugmentation=True, loadLimit=None, normalize=False):
        self.randomDataAugmentation = randomDataAugmentation
        self.normalize = normalize
        self.data = []
        counter = 0
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.data.append([float(value) for value in row])

                counter += 1
                if loadLimit != None:
                    if counter >= loadLimit:
                        break

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     sample = self.data[idx]
    #     inputs = sample[:11]  # Assuming last column is the output
    #     outputs = np.resize(sample[11:], (100, 100))  # Adjust as needed for multiple outputs
    #     return torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)

    def getMeanandStddev(self):
        # Initialize accumulators for sum and sum of squares
        pixel_sum = 0.0
        pixel_sum_sq = 0.0
        total_pixels = 0

        # Iterate through the dataset
        for i in range(len(self)):
            _, outputs = self[i]
            outputs = outputs.numpy()  # Convert to numpy array for easier processing
            pixel_sum += np.sum(outputs)
            pixel_sum_sq += np.sum(outputs ** 2)
            total_pixels += outputs.size  # Count the number of pixels

        # Calculate mean and stddev
        mean = pixel_sum / total_pixels
        stddev = np.sqrt((pixel_sum_sq / total_pixels) - (mean ** 2))

        return mean, stddev

    def __getitem__(self, idx):
        sample = self.data[idx]
        input = sample[:11]
        output = np.resize(sample[11:], (100, 100)) / 1e17

        if self.randomDataAugmentation:
            ndataact = len(output[0])  # Assuming the length of modellc is ndataact

            # # 1x data augmentation:
            # selected_roll = random.choice([0, 0])

            # # 2x data augmentation:
            # selected_roll = random.choice([0, 50])

            # # 3x data augmentation:
            # selected_roll = random.choice([0, 33, 66])

            # 4x data augmentation:
            selected_roll = random.choice([0, 25, 50, 75])

            # # 5x data augmentation:
            # selected_roll = random.choice([0, 20, 40, 60, 80])

            # # 8x data augmentation:
            # selected_roll = random.choice([0, 12, 25, 37, 50, 62, 75, 87])

            # # 10x data augmentation:
            # selected_roll = random.choice([0, 10, 20,30,40, 50, 60,70,80,90])

            # # 100x data augmentation:
            # selected_roll = random.randint(0, ndataact)

            prms1art = input.copy()
            theta = float(selected_roll) / ndataact * 2.0 * np.pi
            csth = np.cos(theta)
            snth = np.sin(theta)

            prms1art[0] = input[0] * csth - input[1] * snth
            prms1art[1] = input[0] * snth + input[1] * csth
            prms1art[4] = np.mod(input[4] + theta, 2.0 * np.pi)

            prms1art[5] = input[5] * csth - input[6] * snth
            prms1art[6] = input[5] * snth + input[6] * csth
            prms1art[9] = np.mod(input[9] + theta, 2.0 * np.pi)

            input = prms1art

            # Convert to correct output format (phase space)
            output = np.flipud(output.transpose())

            # Roll in negative direction to match the direction of the original Fortran code
            output = np.array([np.roll(lightcurve, -selected_roll) for lightcurve in output])

            # Go back to training format
            output = ((np.flipud(output)).transpose()).copy()


        if self.normalize:
            input = Gamma_Normalization.PrecomputedNormalizeParameters()(input)
            # output = Gamma_Normalization.PrecomputedNormalizePhaseAmplitudes()(output)

        return torch.tensor(input, dtype=torch.float32), torch.tensor(output, dtype=torch.float32)
