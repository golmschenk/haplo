from dataclasses import fields, dataclass
from typing import Any, Dict

import wandb as wandb


def wandb_log(name: str, value: Any, process_rank: int):
    if process_rank == 0:  # Only log for the first process.
        wandb.log({name: value}, commit=False)


def wandb_commit(process_rank: int):
    if process_rank == 0:  # Only log for the first process.
        wandb.log({}, commit=True)


def wandb_set_run_name(run_name: str, process_rank: int):
    if process_rank == 0:  # Only log for the first process.
        wandb.run.notes = run_name


def wandb_init(process_rank: int, **kwargs):
    if process_rank == 0:  # Only log for the first process.
        wandb.init(**kwargs)


def wandb_log_hyperparameter(name: str, value: Any, process_rank: int):
    if process_rank == 0:  # Only log for the first process.
        wandb.config[name] = value


def wandb_log_dictionary(log_dictionary: Dict[str, Any], process_rank: int):
    for key, value in log_dictionary.items():
        wandb_log_hyperparameter(key, value, process_rank)

def wandb_log_data_class(data_class: dataclass, process_rank: int):
    for field in fields(data_class):
        wandb_log_hyperparameter(field.name, getattr(data_class, field.name), process_rank)
