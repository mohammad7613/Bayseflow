"""Researcher-owned recovery experiments for the package's general engine.

These functions generate EvaluationBatch objects. Checkpoint loading, posterior
sampling, normalization, metrics, and NPZ/JSON persistence remain in the package.
"""

import keras
import numpy as np

from bayesflow_models.evaluation import EvaluationBatch

from .mysimulator import make_simulator

from .designs import (
    homogeneous_schedule,
    mixed_schedule,
    positive_integer,
    validate_levels,
)
from .myscience import (
    PAR_NAMES,
    TRAINING_TRIALS,
    TTA_LEVELS,
    prior_DC_simplest_model,
    ddm_DC_TwoBoundary_TrialWise_simplest,
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
    # dt = options.get("dt", DEFAULT_DT)
    # if type(dt) not in (int, float) or not np.isfinite(dt) or dt <= 0:
    #     raise ValueError("dt must be positive and finite.")

    # resolved = {
    #     "n_datasets": n_datasets,
    #     "tta_levels": levels.tolist(),
    #     "seed": seed,
    #     "posterior_seed": posterior_seed,
    #     "dt": float(dt),
    # }
    if mixed:
        trials_per_tta = positive_integer(
            options.get("trials_per_tta", 15), "trials_per_tta"
        )
        shuffle = options.get("shuffle", True)
        if type(shuffle) is not bool:
            raise TypeError("shuffle must be a boolean.")
        n_trials = int(len(levels) * trials_per_tta)
        # resolved.update(trials_per_tta=trials_per_tta, shuffle=shuffle)
    else:
        n_trials = positive_integer(
            options.get("n_trials", TRAINING_TRIALS), "n_trials"
        )
        # resolved["n_trials"] = n_trials

    rng = np.random.default_rng(seed)
    observations, covariates = [], []
    truths = {name: [] for name in PAR_NAMES}
    for _ in range(n_datasets):
        # ONE participant vector is reused across every condition in the schedule.
        participant = prior_DC_simplest_model()
        schedule = (
            mixed_schedule(levels, trials_per_tta, rng, shuffle=shuffle)
            if mixed
            else homogeneous_schedule(n_trials, float(rng.choice(levels)))
        )
        simulated = ddm_DC_TwoBoundary_TrialWise_simplest(participant, schedule, rng)
        observations.append(simulated["x"])
        covariates.append(simulated["tta_per_trial"])
        for name in PAR_NAMES:
            truths[name].append(participant[name])

    # The current engine has no separate posterior-seed hook. Set it explicitly
    # here, before checkpoint loading/sampling, rather than confusing it with
    # the independent local RNG used for simulation above.
    keras.utils.set_random_seed(posterior_seed)
    # interpretation = (
    #     "mixed_design_generalization_stress_test"
    #     if mixed
    #     else "training_design_baseline"
    #     if n_trials == TRAINING_TRIALS and dt == DEFAULT_DT
    #     else "changed_design_stress_test"
    # )
    return EvaluationBatch(
        posterior_conditions={
            "x": np.asarray(observations, dtype=np.float32),
            "tta_per_trial": np.asarray(covariates, dtype=np.float32),
            "number_of_trials": np.full((n_datasets, 1, 1), n_trials, dtype=np.int32),
        },
        true_parameters={name: np.asarray(values) for name, values in truths.items()},
    )


def _simulate_mixed_tta_dataset(
    params: dict,
    tta_values: np.ndarray,
    rng: np.random.Generator,
    dt: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one participant/session with different TTA values per trial."""
    theta = params["theta"]
    b0 = params["b0"]
    mu_ndt = params["mu_ndt"]
    mu_alpah = params["mu_alpha"]

    # Match the current simplest trialwise simulator constants.
    sigma_ndt = 1
    sigma_alpha = 1
    k = 1

    x_all = []
    tta_all = []

    for tta in tta_values:
        jitter = rng.uniform(0.0, 0.1)
        tta0 = float(tta + jitter)
        tta_all.append([tta0])

        evidence = 0.0
        t = 0.0
        alpha_trial = mu_alpah + sigma_alpha * rng.normal()

        boundary = b0 * (1 / (1 + np.exp(-k * (tta0 - t - 0.5 * b0))))
        while np.abs(evidence) < boundary:
            evidence += alpha_trial * (tta0 - t - theta) * dt + np.sqrt(dt) * rng.normal()
            t += dt
            boundary = b0 * (1 / (1 + np.exp(-k * (tta0 - t - 0.5 * b0))))

        nt = rng.normal(mu_ndt, sigma_ndt)
        while nt < 0:
            nt = rng.normal(mu_ndt, sigma_ndt)

        if evidence < 0:
            choicert = -1.0
        else:
            choicert = t + nt

        x_all.append([choicert])

    return np.asarray(x_all, dtype=np.float32), np.asarray(tta_all, dtype=np.float32)



def generate_homogenous_tta_validation_data(n_test_sims: int,
    trials_per_tta: int,
    seed: int = 2026,):

    """Generate batched mixed-TTA validation data and true parameters."""
    rng = np.random.default_rng(seed)
    base_ttas = np.repeat(np.array(TTA_LEVELS).astype(np.float32), trials_per_tta)
    n_trials = int(base_ttas.size)

    x_batch = []
    tta_batch = []
    params_batch = {name: [] for name in PAR_NAMES}

    for _ in range(n_test_sims):
        params = prior_DC_simplest_model()




def generate_homogeneous_tta_validation_data(options) -> EvaluationBatch:
    n_datasets = positive_integer(
        options.get("n_datasets", 8), "n_datasets"
    )
    n_trials = positive_integer(
        options.get("n_trials", TRAINING_TRIALS), "n_trials"
    )

    kwargs = {"number_of_trials": n_trials}
    if "tta_condition" in options:
        kwargs["tta_condition"] = options["tta_condition"]

    data = make_simulator().sample(n_datasets, **kwargs)

    return EvaluationBatch(
        posterior_conditions={
            "x": np.asarray(data["x"], dtype=np.float32),
            "tta_per_trial": np.asarray(
                data["tta_per_trial"], dtype=np.float32
            ),
            # The saved adapter broadcasts to (datasets, trials, features).
            # Keep the dataset axis and supply singleton trial/feature axes.
            "number_of_trials": np.full(
                (n_datasets,1,1), n_trials, dtype=np.float32
            ),
        },
        true_parameters={
            name: np.asarray(data[name])
            for name in PAR_NAMES
        },
    )

    




def generate_mixed_tta_validation_data(
    n_test_sims: int,
    trials_per_tta: int,
    shuffle_tta: bool = True,
    seed: int = 2026,
) -> tuple[dict, dict]:
    """Generate batched mixed-TTA validation data and true parameters."""
    rng = np.random.default_rng(seed)
    base_ttas = np.repeat(np.array(TTA_LEVELS).astype(np.float32), trials_per_tta)
    n_trials = int(base_ttas.size)

    x_batch = []
    tta_batch = []
    params_batch = {name: [] for name in PAR_NAMES}

    for _ in range(n_test_sims):
        tta_values = base_ttas.copy()
        if shuffle_tta:
            rng.shuffle(tta_values)

        params = prior_DC_simplest_model()
        x, tta_per_trial = _simulate_mixed_tta_dataset(params, tta_values, rng)

        x_batch.append(x)
        tta_batch.append(tta_per_trial)
        for name in PAR_NAMES:
            params_batch[name].append(params[name])

    conditions = {
        "x": np.asarray(x_batch, dtype=np.float32),
        "tta_per_trial": np.asarray(tta_batch, dtype=np.float32),
        "number_of_trials": np.full((n_test_sims, 1, 1), n_trials, dtype=np.float32),
    }

    true_params = {
        name: np.expand_dims(np.asarray(values, dtype=np.float32),axis=-1)
        for name, values in params_batch.items()
    }

    return EvaluationBatch(posterior_conditions=conditions, true_parameters=true_params)



def generate_homogeneous_tta_batch(options):
    """One nominal condition per participant, with source trial-level jitter."""
    return generate_homogeneous_tta_validation_data(options=options)


def generate_mixed_tta_batch(options):
    """Balanced mixed conditions using the same scientific model and parameters."""
    return generate_mixed_tta_validation_data(
        n_test_sims=options.get("n_datasets", 8),
        trials_per_tta=options.get("trials_per_tta", 15),
        shuffle_tta=options.get("shuffle", True),
        seed=options.get("seed", 2026),
    )
