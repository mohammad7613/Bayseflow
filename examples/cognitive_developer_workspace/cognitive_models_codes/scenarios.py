"""Researcher-owned recovery experiments for the package's general engine.

These functions generate EvaluationBatch objects. Checkpoint loading, posterior
sampling, normalization, metrics, and NPZ/JSON persistence remain in the package.
"""

import keras
import numpy as np

from bayesflow_models.evaluation import EvaluationBatch

from .designs import (
    homogeneous_schedule,
    mixed_schedule,
    positive_integer,
    validate_levels,
)
from .science import (
    DEFAULT_DT,
    PARAMETER_NAMES,
    TRAINING_TRIALS,
    TTA_LEVELS,
    sample_prior,
    simulate_dataset,
)


def _generate_batch(model_spec, options, *, mixed: bool) -> EvaluationBatch:
    # if model_spec.name != "model_DC" or list(model_spec.par_names) != PARAMETER_NAMES:
    #     raise ValueError(
    #         "This scenario requires the seven-parameter workspace model_DC."
    #     )

    common = {
        "n_datasets",
        "tta_levels",
        "seed",
        "posterior_seed",
        "dt",
        "n_posterior_samples",
    }
    allowed = common | ({"trials_per_tta", "shuffle"} if mixed else {"n_trials"})
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unknown scenario options: {sorted(unknown)}.")

    n_datasets = positive_integer(options.get("n_datasets", 8), "n_datasets")
    levels = validate_levels(options.get("tta_levels", TTA_LEVELS))
    seed = options.get("seed", 2026)
    posterior_seed = options.get("posterior_seed", 2027)
    for name, value in (("seed", seed), ("posterior_seed", posterior_seed)):
        if type(value) is not int or value < 0 or value >= 2**32:
            raise ValueError(f"{name} must be an integer in [0, 2**32).")
    dt = options.get("dt", DEFAULT_DT)
    if type(dt) not in (int, float) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite.")

    resolved = {
        "n_datasets": n_datasets,
        "tta_levels": levels.tolist(),
        "seed": seed,
        "posterior_seed": posterior_seed,
        "dt": float(dt),
    }
    if mixed:
        trials_per_tta = positive_integer(
            options.get("trials_per_tta", 15), "trials_per_tta"
        )
        shuffle = options.get("shuffle", True)
        if type(shuffle) is not bool:
            raise TypeError("shuffle must be a boolean.")
        n_trials = int(len(levels) * trials_per_tta)
        resolved.update(trials_per_tta=trials_per_tta, shuffle=shuffle)
    else:
        n_trials = positive_integer(
            options.get("n_trials", TRAINING_TRIALS), "n_trials"
        )
        resolved["n_trials"] = n_trials

    rng = np.random.default_rng(seed)
    observations, covariates = [], []
    truths = {name: [] for name in PARAMETER_NAMES}
    for _ in range(n_datasets):
        # ONE participant vector is reused across every condition in the schedule.
        participant = sample_prior(rng)
        schedule = (
            mixed_schedule(levels, trials_per_tta, rng, shuffle=shuffle)
            if mixed
            else homogeneous_schedule(n_trials, float(rng.choice(levels)))
        )
        simulated = simulate_dataset(participant, schedule, rng, dt=float(dt))
        observations.append(simulated["x"])
        covariates.append(simulated["tta_per_trial"])
        for name in PARAMETER_NAMES:
            truths[name].append(participant[name])

    # The current engine has no separate posterior-seed hook. Set it explicitly
    # here, before checkpoint loading/sampling, rather than confusing it with
    # the independent local RNG used for simulation above.
    keras.utils.set_random_seed(posterior_seed)
    interpretation = (
        "mixed_design_generalization_stress_test"
        if mixed
        else "training_design_baseline"
        if n_trials == TRAINING_TRIALS and dt == DEFAULT_DT
        else "changed_design_stress_test"
    )
    return EvaluationBatch(
        posterior_conditions={
            "x": np.asarray(observations, dtype=np.float32),
            "tta_per_trial": np.asarray(covariates, dtype=np.float32),
            "number_of_trials": np.full((n_datasets, 1), n_trials, dtype=np.int32),
        },
        true_parameters={name: np.asarray(values) for name, values in truths.items()},
        metadata={
            "design": "mixed_tta" if mixed else "homogeneous_tta",
            "interpretation": interpretation,
            "resolved_options": resolved,
            "n_trials": n_trials,
            "training_trials": TRAINING_TRIALS,
            "jitter": {"distribution": "uniform", "lower": 0.0, "upper": 0.1},
            "observation": "signed_response_rt_without_cpp",
            "scientific_version": model_spec.version,
        },
    )


def generate_homogeneous_tta_batch(model_spec, options):
    """One nominal condition per participant, with source trial-level jitter."""
    return _generate_batch(model_spec, options, mixed=False)


def generate_mixed_tta_batch(model_spec, options):
    """Balanced mixed conditions using the same scientific model and parameters."""
    return _generate_batch(model_spec, options, mixed=True)
