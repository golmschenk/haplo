from typing import Any

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
