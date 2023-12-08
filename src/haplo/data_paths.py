import os
import re
import shutil
import socket
from pathlib import Path

from filelock import FileLock

def move_path_to_nvme(path: Path) -> Path:
    match = re.match(r"gpu\d{3}", socket.gethostname())
    if match is not None:
        nvme_path = Path("/lscratch/golmsche").joinpath(path)
        if not nvme_path.exists():
            nvme_path.parent.mkdir(exist_ok=True, parents=True)
            nvme_lock_path = nvme_path.parent.joinpath(nvme_path.name + '.lock')
            lock = FileLock(str(nvme_lock_path))
            with lock.acquire():
                if not nvme_path.exists():
                    nvme_tmp_path = nvme_path.parent.joinpath(nvme_path.name + '.tmp')
                    shutil.copy(path, nvme_tmp_path)
                    nvme_tmp_path.rename(nvme_path)
        return nvme_path
    else:
        return path


def move_to_tmp_on_pbs(path: Path) -> Path:
    if 'PBS_JOBID' in os.environ:
        new_path = Path("/tmp").joinpath(path)
        if not new_path.exists():
            new_path.parent.mkdir(exist_ok=True, parents=True)
            nvme_lock_path = new_path.parent.joinpath(new_path.name + '.lock')
            lock = FileLock(str(nvme_lock_path))
            with lock.acquire():
                if not new_path.exists():
                    nvme_tmp_path = new_path.parent.joinpath(new_path.name + '.tmp')
                    shutil.copy(path, nvme_tmp_path)
                    nvme_tmp_path.rename(new_path)
        return new_path
    else:
        return path

rotated_dataset_path = Path('data/640m_rotated_parameters_and_phase_amplitudes.arrow')
constantinos_kalapotharakos_format_rotated_dataset_path = Path('data/mcmc_vac_all_640m_A.dat')
unrotated_dataset_path = Path('data/800k_parameters_and_phase_amplitudes.arrow')
constantinos_kalapotharakos_format_unrotated_dataset_path = Path('data/mcmc_vac_all_800k.dat')