# rldoom/utils/schedule.py
import numpy as np


def linear_schedule(start: float, end: float, t: int, t_max: int) -> float:
    """Linear interpolation from start to end over t_max steps."""
    t = min(t, t_max)
    return float(start + (end - start) * (t / t_max))


def exponential_schedule(start: float, end: float, t: int, decay: float) -> float:
    """Exponential decay schedule."""
    return float(end + (start - end) * np.exp(-t / decay))
