import pandas as pd

from haplo.data_column_name import DataColumnName
from haplo.nicer_dataset import NicerDataset


class TestNicerDataset:
    def test_getitem(self):
        fake_data = []
        for index in range(3):
            fake_data.append({name: name_index + (64 * index) for name_index, name in enumerate(DataColumnName)})
        data_frame = pd.DataFrame(fake_data)
        dataset = NicerDataset(data_frame)
        parameters1, phase_amplitudes1 = dataset[1]
        assert parameters1[3] == 67
        assert phase_amplitudes1[3] == 78

    def test_len(self):
        fake_data = []
        for index in range(3):
            fake_data.append({name: name_index + (64 * index) for name_index, name in enumerate(DataColumnName)})
        data_frame = pd.DataFrame(fake_data)
        dataset = NicerDataset(data_frame)
        assert len(dataset) == 3
