import inspect
import os
import subprocess
from pathlib import Path

from torch.distributed import init_process_group

from haplo.train_system_configuration import TrainSystemConfiguration


def ddp_setup(system_configuration: TrainSystemConfiguration):
    distributed_back_end = system_configuration.distributed_back_end
    if 'RANK' not in os.environ:
        # The script was not called with `torchrun` and environment variables need to be set manually.
        os.environ['RANK'] = str(0)
        os.environ['LOCAL_RANK'] = str(0)
        os.environ['WORLD_SIZE'] = str(1)
        os.environ['LOCAL_WORLD_SIZE'] = str(1)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "35728"
    init_process_group(backend=distributed_back_end)


def distributed_logging(decorated_function):
    if 'HAPLO_DISTRIBUTED_LOGGING_ENABLED' in os.environ:
        return decorated_function
    if 'RANK' not in os.environ:
        return decorated_function
    if 'HAPLO_SESSION_DIRECTORY' not in os.environ:
        return decorated_function
    else:
        def function_in_subprocess():
            decorated_file = inspect.getfile(decorated_function)
            subprocess_environment = os.environ.copy()
            subprocess_environment['HAPLO_DISTRIBUTED_LOGGING_ENABLED'] = '1'
            session_directory = Path(os.environ['HAPLO_SESSION_DIRECTORY'])
            session_directory.mkdir(parents=True, exist_ok=True)
            with session_directory.joinpath(f'rank_{os.environ["RANK"]}_group_rank_{os.environ["GROUP_RANK"]}.log'
                                            ).open('a') as output_file:
                subprocess.run(['python', decorated_file], env=subprocess_environment,
                               stdout=output_file, stderr=subprocess.STDOUT)

        return function_in_subprocess
