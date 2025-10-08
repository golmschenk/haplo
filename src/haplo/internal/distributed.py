import _thread
import datetime
import inspect
import os
import re
import subprocess
import threading
from pathlib import Path

from torch.distributed import init_process_group

from haplo.internal.train_system_configuration import TrainSystemConfiguration


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


def schedule_self_process_interrupt_signal_before_pbs_end_time(
        time_before_pbs_end_time: datetime.timedelta = datetime.timedelta(minutes=1)) -> None:
    if os.environ.get('PBS_JOBID') is None:
        return

    def kill_process():
        logger.info(f'Sending self terminate signal before PBS wall time limit is reached. '
                    f'Current time is {datetime.datetime.now(tz=datetime.timezone.utc)} UTC.')
        _thread.interrupt_main()

    completed_process = subprocess.run(['qstat', '-f', os.environ['PBS_JOBID']], capture_output=True, text=True)
    process_output = completed_process.stdout
    pbs_start_timestamp_match = re.search(r'\n[^\S\r\n]*stime[^\S\r\n]*=[^\S\r\n]*(\d+)[^\S\r\n]+\(', process_output)
    pbs_start_timestamp = int(pbs_start_timestamp_match.group(1))
    pbs_start_time = datetime.datetime.fromtimestamp(pbs_start_timestamp, tz=datetime.timezone.utc)
    pbs_wall_time_match = re.search(
        r'\n[^\S\r\n]*Resource_List\.walltime[^\S\r\n]*=[^\S\r\n]*(\d+):(\d+):(\d+)\n',
        process_output)
    pbs_wall_time_hours = int(pbs_wall_time_match.group(1))
    pbs_wall_time_minutes = int(pbs_wall_time_match.group(2))
    pbs_wall_time_seconds = int(pbs_wall_time_match.group(3))
    pbs_wall_time_delta = datetime.timedelta(hours=pbs_wall_time_hours, minutes=pbs_wall_time_minutes,
                                             seconds=pbs_wall_time_seconds)
    pbs_end_time = pbs_start_time + pbs_wall_time_delta
    process_kill_datetime = pbs_end_time - time_before_pbs_end_time
    delay = process_kill_datetime - datetime.datetime.now(tz=datetime.timezone.utc)
    logger.info(f'PBS end time: {pbs_end_time} UTC.')
    logger.info(f'Self kill scheduled end time: {process_kill_datetime} UTC.')
    threading.Timer(delay.total_seconds(), kill_process).start()
