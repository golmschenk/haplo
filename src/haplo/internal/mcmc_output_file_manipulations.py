from __future__ import annotations

import pandas as pd
from numpy.random import RandomState


def get_last_states_of_mcmc_states_data_frame(mcmc_states_data_frame: pd.DataFrame, number_of_states: int,
                                              iteration_column_name: str = 'iteration',
                                              maximum_iteration_to_include: int | None = None
                                              ) -> pd.DataFrame:
    random_state = RandomState(0)
    minimum_iteration = mcmc_states_data_frame[iteration_column_name].min()
    if maximum_iteration_to_include is None:
        maximum_iteration = mcmc_states_data_frame[iteration_column_name].max()
    else:
        maximum_iteration = maximum_iteration_to_include
    current_iteration = maximum_iteration
    latest_states_data_frame: pd.DataFrame = mcmc_states_data_frame[
        mcmc_states_data_frame[iteration_column_name] == current_iteration]
    if latest_states_data_frame.shape[0] >= number_of_states:
        latest_states_data_frame = latest_states_data_frame.sample(number_of_states, random_state=random_state)
        latest_states_data_frame.sort_values(iteration_column_name)
        return latest_states_data_frame
    while current_iteration > minimum_iteration:
        current_iteration -= 1
        current_iteration_states_data_frame: pd.DataFrame = mcmc_states_data_frame[
            mcmc_states_data_frame[iteration_column_name] == current_iteration]
        if latest_states_data_frame.shape[0] + current_iteration_states_data_frame.shape[0] >= number_of_states:
            current_iteration_states_data_frame = current_iteration_states_data_frame.sample(
                number_of_states - latest_states_data_frame.shape[0], random_state=random_state)
            latest_states_data_frame = pd.concat([current_iteration_states_data_frame, latest_states_data_frame])
            latest_states_data_frame = latest_states_data_frame.sort_values(iteration_column_name)
            return latest_states_data_frame
        else:
            latest_states_data_frame = pd.concat([current_iteration_states_data_frame, latest_states_data_frame])
    raise ValueError(f'Exhausted MCMC state data frame while attempting to obtain f{number_of_states}.')
