"""Experimental schedules only: no DDM equations, inference, or file writes."""

import numpy as np

from .science import TTA_LEVELS


def positive_integer(value, name: str) -> int:
    """Reject booleans as counts as well as nonpositive/noninteger values."""
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def validate_levels(values) -> np.ndarray:
    """This example evaluates the nominal conditions supported by training."""
    levels = np.asarray(values, dtype=float)
    if (
        levels.ndim != 1
        or levels.size == 0
        or not np.all(np.isfinite(levels))
        or not np.all(np.isin(levels, TTA_LEVELS))
        or len(np.unique(levels)) != levels.size
    ):
        raise ValueError(f"tta_levels must be distinct values from {TTA_LEVELS}.")
    return levels


def homogeneous_schedule(n_trials: int, tta: float) -> np.ndarray:
    """Repeat one nominal condition; the scientific kernel later adds jitter."""
    positive_integer(n_trials, "n_trials")
    validate_levels([tta])
    return np.full(n_trials, tta, dtype=float)


def mixed_schedule(
    levels: np.ndarray,
    trials_per_tta: int,
    rng: np.random.Generator,
    *,
    shuffle: bool,
) -> np.ndarray:
    """Give each nominal condition an equal count, optionally shuffling trials."""
    levels = validate_levels(levels)
    positive_integer(trials_per_tta, "trials_per_tta")
    if type(shuffle) is not bool:
        raise TypeError("shuffle must be a boolean.")
    schedule = np.repeat(levels, trials_per_tta)
    if shuffle:
        rng.shuffle(schedule)
    return schedule
