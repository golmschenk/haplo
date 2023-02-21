import pandas as pd
from torch.utils.data import Dataset

from haplo.data_column_name import DataColumnName
from haplo.data_paths import dataset_path


class NicerDataset(Dataset):
    def __init__(self):
        self.data_frame: pd.DataFrame = pd.read_feather(dataset_path)

    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, index):
        row = self.data_frame.iloc[index]
        parameters = row.loc[[
            DataColumnName.PARAMETER0,
            DataColumnName.PARAMETER1,
            DataColumnName.PARAMETER2,
            DataColumnName.PARAMETER3,
            DataColumnName.PARAMETER4,
            DataColumnName.PARAMETER5,
            DataColumnName.PARAMETER6,
            DataColumnName.PARAMETER7,
            DataColumnName.PARAMETER8,
            DataColumnName.PARAMETER9,
            DataColumnName.PARAMETER10,
        ]]
        phase_amplitudes = row.loc[[
            DataColumnName.PHASE_AMPLITUDE0,
            DataColumnName.PHASE_AMPLITUDE1,
            DataColumnName.PHASE_AMPLITUDE2,
            DataColumnName.PHASE_AMPLITUDE3,
            DataColumnName.PHASE_AMPLITUDE4,
            DataColumnName.PHASE_AMPLITUDE5,
            DataColumnName.PHASE_AMPLITUDE6,
            DataColumnName.PHASE_AMPLITUDE7,
            DataColumnName.PHASE_AMPLITUDE8,
            DataColumnName.PHASE_AMPLITUDE9,
            DataColumnName.PHASE_AMPLITUDE10,
            DataColumnName.PHASE_AMPLITUDE11,
            DataColumnName.PHASE_AMPLITUDE12,
            DataColumnName.PHASE_AMPLITUDE13,
            DataColumnName.PHASE_AMPLITUDE14,
            DataColumnName.PHASE_AMPLITUDE15,
            DataColumnName.PHASE_AMPLITUDE16,
            DataColumnName.PHASE_AMPLITUDE17,
            DataColumnName.PHASE_AMPLITUDE18,
            DataColumnName.PHASE_AMPLITUDE19,
            DataColumnName.PHASE_AMPLITUDE20,
            DataColumnName.PHASE_AMPLITUDE21,
            DataColumnName.PHASE_AMPLITUDE22,
            DataColumnName.PHASE_AMPLITUDE23,
            DataColumnName.PHASE_AMPLITUDE24,
            DataColumnName.PHASE_AMPLITUDE25,
            DataColumnName.PHASE_AMPLITUDE26,
            DataColumnName.PHASE_AMPLITUDE27,
            DataColumnName.PHASE_AMPLITUDE28,
            DataColumnName.PHASE_AMPLITUDE29,
            DataColumnName.PHASE_AMPLITUDE30,
            DataColumnName.PHASE_AMPLITUDE31,
            DataColumnName.PHASE_AMPLITUDE32,
            DataColumnName.PHASE_AMPLITUDE33,
            DataColumnName.PHASE_AMPLITUDE34,
            DataColumnName.PHASE_AMPLITUDE35,
            DataColumnName.PHASE_AMPLITUDE36,
            DataColumnName.PHASE_AMPLITUDE37,
            DataColumnName.PHASE_AMPLITUDE38,
            DataColumnName.PHASE_AMPLITUDE39,
            DataColumnName.PHASE_AMPLITUDE40,
            DataColumnName.PHASE_AMPLITUDE41,
            DataColumnName.PHASE_AMPLITUDE42,
            DataColumnName.PHASE_AMPLITUDE43,
            DataColumnName.PHASE_AMPLITUDE44,
            DataColumnName.PHASE_AMPLITUDE45,
            DataColumnName.PHASE_AMPLITUDE46,
            DataColumnName.PHASE_AMPLITUDE47,
            DataColumnName.PHASE_AMPLITUDE48,
            DataColumnName.PHASE_AMPLITUDE49,
            DataColumnName.PHASE_AMPLITUDE50,
            DataColumnName.PHASE_AMPLITUDE51,
            DataColumnName.PHASE_AMPLITUDE52,
            DataColumnName.PHASE_AMPLITUDE53,
            DataColumnName.PHASE_AMPLITUDE54,
            DataColumnName.PHASE_AMPLITUDE55,
            DataColumnName.PHASE_AMPLITUDE56,
            DataColumnName.PHASE_AMPLITUDE57,
            DataColumnName.PHASE_AMPLITUDE58,
            DataColumnName.PHASE_AMPLITUDE59,
            DataColumnName.PHASE_AMPLITUDE60,
            DataColumnName.PHASE_AMPLITUDE61,
            DataColumnName.PHASE_AMPLITUDE62,
            DataColumnName.PHASE_AMPLITUDE63,
        ]]
        return parameters, phase_amplitudes