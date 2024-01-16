from dataclasses import dataclass
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

    @classmethod
    def new(cls,
            wandb_project: str = 'ramjet',
            wandb_entity: str = 'example',
            additional_log_dictionary: Dict[str, Any] | None = None):
        if additional_log_dictionary is None:
            additional_log_dictionary = {}
        return cls(wandb_project=wandb_project,
                   wandb_entity=wandb_entity,
                   additional_log_dictionary=additional_log_dictionary)
