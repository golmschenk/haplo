from dataclasses import dataclass

from torch.distributed import Backend


@dataclass
class TrainSystemConfiguration:
    """
    Configuration settings for the system of a train session.

    :ivar preprocessing_processes_per_train_process: The number of processes that are started to preprocess the data
        per train process. The train session will create this many processes for each the train data and the validation
        data.
    """
    preprocessing_processes_per_train_process: int
    distributed_back_end: Backend

    @classmethod
    def new(cls,
            preprocessing_processes_per_train_process: int = 10,
            distributed_back_end: Backend = Backend.GLOO):
        return cls(preprocessing_processes_per_train_process=preprocessing_processes_per_train_process,
                   distributed_back_end=distributed_back_end)
