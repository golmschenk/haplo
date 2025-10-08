import logging
from dataclasses import fields, dataclass
from pathlib import Path
from typing import Any, Dict

import wandb as wandb
import yaml

logger = logging.getLogger(__name__)


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
        logger.info(f'{process_rank}: Starting wandb...')
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


def wandb_save_file(path: Path, process_rank: int):
    if process_rank == 0:  # Only log for the first process.
        wandb.save(str(path))


def wandb_save_manual_config_file(process_rank):
    if process_rank == 0:
        manual_config_path = Path(wandb.run.dir).joinpath('manual_config.yaml')
        with manual_config_path.open('w') as manual_config_file:
            yaml.dump(wandb.config, manual_config_file)
        wandb_save_file(manual_config_path, process_rank=process_rank)
