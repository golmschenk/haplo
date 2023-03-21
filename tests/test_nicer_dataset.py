from unittest.mock import patch

import pandas as pd

from haplo.data_column_name import DataColumnName
from haplo.nicer_dataset import NicerDataset


class TestNicerDataset:
    @patch('haplo.nicer_dataset.pd.read_feather')
    def test_getitem(self, stub_read_feather):
        fake_data = []
        for index in range(3):
            fake_data.append({name: name_index + (64 * index) for name_index, name in enumerate(DataColumnName)})
        stub_read_feather.return_value = pd.DataFrame(fake_data)
        dataset = NicerDataset()
        parameters1, phase_amplitudes1 = dataset[1]
        assert parameters1[3] == 67
        assert phase_amplitudes1[3] == 78

    @patch('haplo.nicer_dataset.pd.read_feather')
    def test_len(self, stub_read_feather):
        fake_data = []
        for index in range(3):
            fake_data.append({name: name_index + (64 * index) for name_index, name in enumerate(DataColumnName)})
        stub_read_feather.return_value = pd.DataFrame(fake_data)
        dataset = NicerDataset()
        assert len(dataset) == 3
