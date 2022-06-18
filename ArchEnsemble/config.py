from typing import NamedTuple


class Config(NamedTuple):
    seed: int = 42
    num_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1.0
    gamma: float = 0.7
    num_workers: int = 10
    train_log_interval: int = 100
    ensemble_num: int = 5
