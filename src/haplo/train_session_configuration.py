from dataclasses import dataclass


@dataclass
class TrainSessionConfiguration:
    """
    Configuration settings for a train session.

    :ivar batch_size: The size of the batch for each train process. Each training step will use a number of examples
        equal to this value multiplied by the number of train processes.
    :ivar cycles: The number of train cycles to run.
    """
    batch_size: int = 100
    cycles: int = 5000
