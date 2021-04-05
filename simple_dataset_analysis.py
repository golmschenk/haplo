from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd


class ColumnName(Enum):
    CUT_OFF_ENERGY__MEV = "Cut off energy (MeV)"
    SURFACE_MAGNETIC_FIELD__G = "Surface magnetic field (G)"
    SPIN_DOWN_POWER__ERG_PER_S = "Spin-down power (erg/s)"
    TOTAL_GAMMA_RAY_LUMINOSITY__ERG_PER_S = "Total gamma-ray luminosity (erg/s)"

def load_dataset() -> pd.DataFrame:
    dataset_path = Path("data/MLdata88.dat")
    column_names = [ColumnName.CUT_OFF_ENERGY__MEV.value,
                    ColumnName.SURFACE_MAGNETIC_FIELD__G.value,
                    ColumnName.SPIN_DOWN_POWER__ERG_PER_S.value,
                    ColumnName.TOTAL_GAMMA_RAY_LUMINOSITY__ERG_PER_S.value]
    dataset_data_frame = pd.read_csv(dataset_path, delim_whitespace=True, skipinitialspace=True,
                                     names=column_names)
    for column_name in column_names:
        dataset_data_frame[f"Log {column_name}"] = np.log10(dataset_data_frame[column_name])
    return dataset_data_frame

if __name__ == '__main__':
    dataset_data_frame = load_dataset()
