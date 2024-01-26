import _thread
import datetime
import logging
import os
import re
import subprocess
import threading

logger = logging.getLogger(__name__)


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
