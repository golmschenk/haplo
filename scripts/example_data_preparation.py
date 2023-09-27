from pathlib import Path

from haplo.data_preparation import constantinos_kalapotharakos_format_file_to_sqlite


def example_data_preparation():
    constantinos_kalapotharakos_format_file_to_sqlite(
        Path('data/mcmc_vac_all_800k.dat'), Path('data/800k_parameters_and_phase_amplitudes.db'))


if __name__ == '__main__':
    example_data_preparation()
