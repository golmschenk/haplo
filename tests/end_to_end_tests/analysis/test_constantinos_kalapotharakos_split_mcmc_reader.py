from pathlib import Path

import pytest

from haplo.internal.combine_split_mcmc_output_files import get_chain_count_and_known_complete_iterations


def test_constantinos_kalapotharakos_split_mcmc_reader_determines_correct_number_of_chains():
    chain_count0, _ = get_chain_count_and_known_complete_iterations(
        Path(__file__).parent.joinpath('constantinos_kalapotharakos_split_mcmc_reader_resources/mcmc_vac_100000.dat'),
        elements_per_record=13)
    assert chain_count0 == 2
    chain_count1, _ = get_chain_count_and_known_complete_iterations(
        Path(__file__).parent.joinpath('constantinos_kalapotharakos_split_mcmc_reader_resources/mcmc_vac_100001.dat'),
        elements_per_record=13)
    assert chain_count1 == 2
    chain_count2, _ = get_chain_count_and_known_complete_iterations(
        Path(__file__).parent.joinpath('constantinos_kalapotharakos_split_mcmc_reader_resources/mcmc_vac_100002.dat'),
        elements_per_record=13)
    assert chain_count2 == 2
    chain_count3, _ = get_chain_count_and_known_complete_iterations(
        Path(__file__).parent.joinpath('constantinos_kalapotharakos_split_mcmc_reader_resources/mcmc_vac_100003.dat'),
        elements_per_record=13)
    assert chain_count3 == 3
    chain_count4, _ = get_chain_count_and_known_complete_iterations(
        Path(__file__).parent.joinpath('constantinos_kalapotharakos_split_mcmc_reader_resources/mcmc_vac_100004.dat'),
        elements_per_record=13)
    assert chain_count4 == 1


def test_constantinos_kalapotharakos_split_mcmc_reader_determines_correct_number_of_iterations():
    _, known_complete_iteration_index0 = get_chain_count_and_known_complete_iterations(
        Path(__file__).parent.joinpath('constantinos_kalapotharakos_split_mcmc_reader_resources/mcmc_vac_100000.dat'),
        elements_per_record=13)
    assert known_complete_iteration_index0 == 3
    _, known_complete_iteration_index1 = get_chain_count_and_known_complete_iterations(
        Path(__file__).parent.joinpath('constantinos_kalapotharakos_split_mcmc_reader_resources/mcmc_vac_100001.dat'),
        elements_per_record=13)
    assert known_complete_iteration_index1 == 3
    _, known_complete_iteration_index2 = get_chain_count_and_known_complete_iterations(
        Path(__file__).parent.joinpath('constantinos_kalapotharakos_split_mcmc_reader_resources/mcmc_vac_100002.dat'),
        elements_per_record=13)
    assert known_complete_iteration_index2 == 2
    _, known_complete_iteration_index3 = get_chain_count_and_known_complete_iterations(
        Path(__file__).parent.joinpath('constantinos_kalapotharakos_split_mcmc_reader_resources/mcmc_vac_100003.dat'),
        elements_per_record=13)
    assert known_complete_iteration_index3 == 1
    _, known_complete_iteration_index4 = get_chain_count_and_known_complete_iterations(
        Path(__file__).parent.joinpath('constantinos_kalapotharakos_split_mcmc_reader_resources/mcmc_vac_100004.dat'),
        elements_per_record=13)
    assert known_complete_iteration_index4 == 3


def test_constantinos_kalapotharakos_split_mcmc_reader_errors_when_chains_stop_incrementing_correctly():
    with pytest.raises(ValueError):
        _ = get_chain_count_and_known_complete_iterations(
            Path(__file__).parent.joinpath('constantinos_kalapotharakos_split_mcmc_reader_resources/mcmc_vac_100005.dat'),
            elements_per_record=13)
