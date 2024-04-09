import re

from pathlib import Path

from haplo.train_logging_configuration import TrainLoggingConfiguration


def test_passing_no_session_directory_to_new_creates_one_based_on_the_datetime():
    configuration = TrainLoggingConfiguration.new()
    match = re.search(r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}', configuration.session_directory.name)
    assert match is not None


def test_passing_specific_session_directory_to_new():
    session_directory = Path('a')
    configuration = TrainLoggingConfiguration.new(session_directory=session_directory)
    assert configuration.session_directory == Path('a')
