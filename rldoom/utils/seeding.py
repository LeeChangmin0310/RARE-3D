# rldoom/utils/seeding.py
import os
import random
import numpy as np
import torch


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
