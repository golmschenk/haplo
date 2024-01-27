import datetime
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class TrainLoggingConfiguration:
    """
    Configuration settings for the logging of a train session.

    :ivar wandb_project: The wandb project to log to.
    :ivar wandb_entity: The wandb entity to log to.
    :ivar additional_log_dictionary: The dictionary of additional values to log.
    """
    wandb_project: str
    wandb_entity: str
    additional_log_dictionary: Dict[str, Any]
    session_directory: Path

    @classmethod
    def new(cls,
            wandb_project: str = 'ramjet',
            wandb_entity: str = 'example',
            additional_log_dictionary: Dict[str, Any] | None = None,
            session_directory: Path | None = None):
        if additional_log_dictionary is None:
            additional_log_dictionary = {}
        session_directory_environment_variable = os.environ.get('HAPLO_SESSION_DIRECTORY')
        if session_directory_environment_variable is not None and session_directory is not None:
            raise ValueError(f'Passing a `session_directory` is not allowed when the environment variable '
                             f'`HAPLO_SESSION_DIRECTORY` is set.')
        session_directory = session_directory_environment_variable
        if session_directory is None:
            datetime_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            session_directory = Path(f'sessions/{datetime_string}')
        if isinstance(session_directory, str):
            session_directory = Path(session_directory)
        return cls(wandb_project=wandb_project,
                   wandb_entity=wandb_entity,
                   additional_log_dictionary=additional_log_dictionary,
                   session_directory=session_directory)
