"""The seven-parameter model_DC copied from the existing article implementation.

Source: ../article_workspace/plugins/reproduce_models.py, prior_DC/simulator_DC.
Only structure, input validation, and explicit RNG ownership have changed.
In particular, the source calculates the boundary ONCE before accumulation;
we deliberately preserve that behavior, despite its 'collapsing boundary' name.
Changing the equations should create a new scientific model version.
"""

from collections.abc import Mapping

import numpy as np

PARAMETER_NAMES = ["tau", "b0", "k", "mu_ndt", "sigma_ndt", "alpha", "theta"]
TTA_LEVELS = (2.5, 3.0, 3.5, 4.0)
TRAINING_TRIALS = 60
DEFAULT_DT = 0.005


def sample_prior(rng: np.random.Generator) -> dict[str, float]:
    """Sample one participant's parameters, preserving the source prior/order."""
    return {
        "tau": rng.uniform(0.0, 1.0),
        "b0": rng.uniform(0.1, 2.0),
        "k": rng.uniform(1.0, 5.0),
        "mu_ndt": rng.uniform(0.2, 0.6),
        "sigma_ndt": rng.uniform(0.06, 0.1),
        "alpha": rng.uniform(0.0, 1.0),
        "theta": rng.uniform(0.0, 1.0),
    }


def simulate_dataset(
    parameters: Mapping[str, float],
    nominal_ttas: np.ndarray,
    rng: np.random.Generator,
    *,
    dt: float = DEFAULT_DT,
) -> dict[str, np.ndarray]:
    """Simulate one participant with ONE parameter vector across all trials.

    nominal_ttas has shape (T,). Each trial adds Uniform(0, 0.1) jitter.
    Returns x and tta_per_trial, both (T, 1), in float64 like the source.
    x is reaction time plus positive nondecision time, or -1 for the source's
    negative-evidence/timeout response. No CPP feature is generated.

    A homogeneous schedule repeats one nominal TTA; a mixed schedule contains
    several. Both use exactly the same trial equations below.
    """
    if set(parameters) != set(PARAMETER_NAMES):
        raise ValueError(f"Parameters must be exactly {PARAMETER_NAMES}.")
    if not all(np.isfinite(value) for value in parameters.values()):
        raise ValueError("Parameters must be finite.")
    if parameters["b0"] <= 0 or parameters["sigma_ndt"] <= 0:
        raise ValueError("b0 and sigma_ndt must be positive.")
    nominal_ttas = np.asarray(nominal_ttas, dtype=float)
    if (
        nominal_ttas.ndim != 1
        or nominal_ttas.size == 0
        or not np.all(np.isfinite(nominal_ttas))
        or np.any(nominal_ttas <= 0)
    ):
        raise ValueError("nominal_ttas must be a non-empty positive finite vector.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite.")

    b0, tau, k = (parameters[name] for name in ("b0", "tau", "k"))
    mu_ndt, sigma_ndt = (parameters[name] for name in ("mu_ndt", "sigma_ndt"))
    alpha, theta = (parameters[name] for name in ("alpha", "theta"))
    observations, actual_ttas = [], []

    for nominal_tta in nominal_ttas:
        tta0 = float(nominal_tta + rng.uniform(0.0, 0.1))
        actual_ttas.append([tta0])
        evidence, t = 0.0, 0.0

        # Faithful reproduction: do not move this expression into the loop.
        boundary = b0 * (1 / (1 + np.exp(-k * (tta0 - t - tau))))
        while np.abs(evidence) < boundary and t < tta0:
            evidence += alpha * (tta0 - t - theta) * dt + np.sqrt(dt) * rng.normal()
            t += dt

        nt = rng.normal(mu_ndt, sigma_ndt)
        while nt < 0:
            nt = rng.normal(mu_ndt, sigma_ndt)

        choicert = -1.0 if evidence < 0 or t > tta0 else t + nt
        observations.append([choicert])

    return {"x": np.asarray(observations), "tta_per_trial": np.asarray(actual_ttas)}
