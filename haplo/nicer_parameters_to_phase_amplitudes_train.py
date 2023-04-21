from haplo.nicer_dataset import NicerDataset, split_into_train_validation_and_test_datasets
from haplo.nicer_transform import PrecomputedNormalizeParameters, PrecomputedNormalizePhaseAmplitudes

full_dataset0 = NicerDataset()
full_dataset = NicerDataset(parameters_transform=PrecomputedNormalizeParameters(),
                            phase_amplitudes_transform=PrecomputedNormalizePhaseAmplitudes())
train_dataset, validation_dataset, test_dataset = split_into_train_validation_and_test_datasets(full_dataset)
parameters0, phase_amplitudes0 = train_dataset[0]
pass