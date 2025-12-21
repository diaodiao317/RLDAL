from dataclasses import dataclass, asdict
from typing import Tuple


@dataclass
class RLDALConfig:
    """Configuration for running RLDAL (defaults mirror `utils/parser.py`)."""

    # Paths
    data_path: str = "./scratch/"
    ckpt_path: str = "./scratch/ckpt_seg"
    exp_name: str = "rldal_acdc"
    exp_name_toload: str = ""
    exp_name_toload_rl: str = ""

    # General
    seed: int = 26
    dataset: str = "ACDC"
    al_algorithm: str = "ralis"
    region_size: Tuple[int, int] = (64, 64)
    input_size: Tuple[int, int] = (256, 256)
    scale_size: int = 0
    full_res: bool = False
    only_last_labeled: bool = True

    # Active learning
    num_each_iter: int = 1
    rl_pool: int = 50
    budget_labels: int = 100
    rl_episodes: int = 50
    rl_buffer: int = 3200
    dqn_bs: int = 16
    dqn_gamma: float = 0.9
    dqn_epochs: int = 1
    dqn_action_select: str = "softmax"  # "epsilon" or "softmax"
    dqn_temp: float = 0.7
    bald_iter: int = 20

    # Optimization
    optimizer: str = "SGD"
    lr: float = 0.001
    lr_dqn: float = 0.0001
    weight_decay: float = 1e-4
    momentum: float = 0.95
    gamma: float = 0.998
    gamma_scheduler_dqn: float = 0.99
    epoch_num: int = 5
    patience: int = 20
    train_batch_size: int = 8
    val_batch_size: int = 4
    n_workers: int = 4

    # Checkpointing/test
    snapshot: str = "last_jaccard_val.pth"
    load_weights: bool = False
    load_opt: bool = False
    checkpointer: bool = False
    test: bool = False
    final_test: bool = False
    final_sup_only: bool = False

    # Semi-supervised/EMA knobs (kept for parity, unused in the slim trainer)
    labeled_fraction: float = 0.05
    consistency_weight: float = 1.0
    confidence_threshold: float = 0.6

    def to_data_kwargs(self) -> dict:
        """Map config to kwargs expected by `data.data_utils.get_data`."""
        return {
            "data_path": self.data_path,
            "tr_bs": self.train_batch_size,
            "vl_bs": self.val_batch_size,
            "n_workers": self.n_workers,
            "scale_size": self.scale_size,
            "input_size": self.input_size,
            "num_each_iter": self.num_each_iter,
            "only_last_labeled": self.only_last_labeled,
            "dataset": self.dataset,
            "test": self.test,
            "al_algorithm": self.al_algorithm,
            "full_res": self.full_res,
            "region_size": self.region_size,
        }

    def asdict(self) -> dict:
        return asdict(self)
