"""Pure numerical operations for parameter recovery, independent of model type.

Pipeline:
    normalize_posterior_draws -> compute_recovery_metrics
        (which calls summarize_posterior_draws) -> build_recovery_result_payload

N means datasets (e.g. simulated participants); S means posterior draws.
Each scalar parameter has draws shaped (N, S), truths shaped (N,), and one
summary value per dataset. No function here loads checkpoints or writes files.
"""

from typing import Any

import numpy as np

RecoveryMetrics = dict[str, dict[str, float | None]]
PosteriorSummaries = dict[str, dict[str, np.ndarray]]


def _finite_real_array(values: Any, *, label: str) -> np.ndarray:
    """Reject invalid numbers without coercing strings or complex data."""
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{label} must contain real numbers; received {array.dtype}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values.")
    return array


def _validate_parameter_keys(
    values: dict, parameter_names: list[str], *, label: str
) -> None:
    """Check identities before comparing arrays; extras can hide a model mismatch."""
    if not isinstance(values, dict):
        raise TypeError(f"{label} must be a dictionary keyed by parameter name.")
    expected = set(parameter_names)
    missing = expected - set(values)
    if missing:
        raise KeyError(f"{label} is missing parameters: {sorted(missing)}.")
    extra = set(values) - expected
    if extra:
        raise ValueError(
            f"{label} has unexpected parameters: {sorted(extra, key=str)}."
        )


def _validated_normalized_draws(
    normalized_draws: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Check the canonical contract at public summary/metric entry points."""
    if not isinstance(normalized_draws, dict) or not normalized_draws:
        raise ValueError("normalized_draws must be a non-empty parameter dictionary.")

    validated = {}
    expected_shape = None
    for name, values in normalized_draws.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Parameter names must be non-empty strings.")
        array = _finite_real_array(values, label=f"Draws for {name!r}")
        if array.ndim != 2 or 0 in array.shape:
            raise ValueError(
                f"Normalized draws for {name!r} must have non-empty shape (N, S); "
                f"received {array.shape}."
            )
        if expected_shape is not None and array.shape != expected_shape:
            raise ValueError(
                "All parameters must share the same dataset and draw counts."
            )
        expected_shape = array.shape
        validated[name] = array
    return validated


def normalize_posterior_draws(
    *,
    posterior_samples: dict[str, Any],
    parameter_names: list[str],
    n_datasets: int,
    n_posterior_samples: int,
) -> dict[str, np.ndarray]:
    """Validate sampler output and return one (N, S) array per scalar parameter.

    Accept (N, S) or (N, S, 1), where the final singleton is a scalar feature
    axis. Dataset and draw axes are never inferred, transposed, or squeezed.
    Thus (1, S), (N, 1), (1, 1), and square (N, N) arrays retain their meaning.

    Counts must come from the evaluation batch and requested sampling budget,
    rather than being inferred from the output that we are trying to validate.
    Missing parameter keys raise KeyError; unsupported shapes/counts/non-finite
    values raise ValueError; non-real data raise TypeError.

    The returned dictionary is new. Its arrays may share storage with the
    inputs, but this function and downstream recovery functions do not modify
    that storage. Combined arrays such as (N, S, P) are not accepted; split them
    into explicitly named scalar parameters at the sampler boundary.
    """
    for label, count in (
        ("n_datasets", n_datasets),
        ("n_posterior_samples", n_posterior_samples),
    ):
        # bool is an int subclass, so an exact check avoids accepting True as 1.
        if type(count) is not int:
            raise TypeError(f"{label} must be an integer.")
        if count < 1:
            raise ValueError(f"{label} must be at least 1.")

    if not parameter_names or any(
        not isinstance(name, str) or not name for name in parameter_names
    ):
        raise ValueError("parameter_names must contain non-empty strings.")
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError("parameter_names must not contain duplicates.")
    _validate_parameter_keys(
        posterior_samples, parameter_names, label="Posterior samples"
    )

    normalized = {}
    expected_shape = (n_datasets, n_posterior_samples)
    for name in parameter_names:
        values = _finite_real_array(
            posterior_samples[name], label=f"Draws for {name!r}"
        )
        # Remove only the scalar feature axis. np.squeeze() would also delete
        # a dataset/draw axis when there is one participant or one draw.
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[..., 0]
        if values.shape != expected_shape:
            raise ValueError(
                f"Posterior draws for {name!r} must have shape {expected_shape} "
                f"or {expected_shape + (1,)}; received {values.shape}."
            )
        normalized[name] = values
    return normalized


def summarize_posterior_draws(
    *,
    normalized_draws: dict[str, np.ndarray],
    lower: float = 0.025,
    upper: float = 0.975,
) -> PosteriorSummaries:
    """Compute per-dataset means, standard deviations, and equal-tailed bounds.

    All reductions use axis 1 (posterior draws), leaving arrays of shape (N,).
    Bounds are empirical quantiles with NumPy's linear interpolation; the
    defaults give a central 95% credible interval, not a highest-density one.
    Standard deviation uses ddof=0 to describe the supplied empirical draws.
    With S=1 it is zero and both bounds equal that draw; this does not establish
    that the underlying posterior has zero uncertainty.
    """
    if not 0 <= lower < upper <= 1:
        raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1.")
    draws = _validated_normalized_draws(normalized_draws)
    summaries = {}
    for name, values in draws.items():
        # float64 avoids integer arithmetic overflow and accumulates ordinary
        # float32 sampler output with greater precision.
        try:
            # Finite inputs can still overflow during arithmetic. Fail here
            # rather than allowing an invalid summary to enter a report.
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                values = values.astype(np.float64, copy=False)
                summaries[name] = {
                    "mean": values.mean(axis=1),
                    "std": values.std(axis=1, ddof=0),
                    "lower": np.quantile(values, lower, axis=1, method="linear"),
                    "upper": np.quantile(values, upper, axis=1, method="linear"),
                }
        except FloatingPointError as exc:
            raise ValueError(
                f"Posterior summaries for {name!r} exceeded floating-point limits."
            ) from exc
    return summaries


def compute_recovery_metrics(
    *,
    normalized_draws: dict[str, np.ndarray],
    true_parameters: dict[str, Any],
) -> tuple[RecoveryMetrics, PosteriorSummaries]:
    """Compare posterior summaries with one known truth per dataset/parameter.

    Call normalize_posterior_draws first to check the sampler's expected counts.
    Truths may have shape (N,) or (N, 1); reject other shapes before subtraction
    to prevent accidental broadcasting into an (N, N) error array.

    For errors e_i = posterior_mean_i - truth_i, bias is mean(e_i) and RMSE is
    sqrt(mean(e_i**2)). Coverage is the fraction of truths inside their own
    central 95% intervals, including endpoints. Pearson correlation compares
    truths with posterior means across datasets. It is None for one dataset or
    constant truths/means because correlation is undefined in those cases.

    The first return value contains Python floats/None suitable for JSON. The
    second contains per-dataset NumPy arrays for later NPZ persistence/plotting.
    No input array is modified; there are no model-specific assumptions.
    """
    draws = _validated_normalized_draws(normalized_draws)
    _validate_parameter_keys(true_parameters, list(draws), label="True parameters")
    summaries = summarize_posterior_draws(normalized_draws=draws)
    metrics = {}

    for name, values in draws.items():
        n_datasets = values.shape[0]
        target = _finite_real_array(true_parameters[name], label=f"Truths for {name!r}")
        if target.shape == (n_datasets, 1):
            target = target[:, 0]
        if target.shape != (n_datasets,):
            raise ValueError(
                f"Truths for {name!r} must have shape ({n_datasets},) or "
                f"({n_datasets}, 1); received {target.shape}."
            )
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                target = target.astype(np.float64, copy=False)
                summary = summaries[name]
                mean = summary["mean"]
                error = mean - target

                correlation = None
                if n_datasets > 1 and np.std(mean) > 0 and np.std(target) > 0:
                    correlation = float(np.corrcoef(target, mean)[0, 1])

                metrics[name] = {
                    "bias": float(np.mean(error)),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "pearson": correlation,
                    "posterior_std_mean": float(np.mean(summary["std"])),
                    "coverage_95": float(
                        np.mean(
                            (summary["lower"] <= target) & (target <= summary["upper"])
                        )
                    ),
                }
        except FloatingPointError as exc:
            raise ValueError(
                f"Recovery metrics for {name!r} exceeded floating-point limits."
            ) from exc
    return metrics, summaries


def build_recovery_result_payload(*, metrics: RecoveryMetrics) -> dict:
    """Build the numerical results section of the engine's JSON report.

    Keep this payload small: per-dataset summaries, draws, and truths belong in
    array files, whose paths the engine records in its report envelope. Convert
    real NumPy scalars to Python floats and preserve None as JSON null. Reject
    NaN/infinity explicitly: Python's default JSON encoder allows them although
    they are not valid JSON numbers.

    Returns a fresh nested dictionary without file writes. The engine will
    later call this function after compute_recovery_metrics().
    """
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("metrics must be a non-empty parameter dictionary.")
    payload = {}
    for name, parameter_metrics in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Metric parameter names must be non-empty strings.")
        if not isinstance(parameter_metrics, dict) or not parameter_metrics:
            raise ValueError(f"Metrics for {name!r} must be a non-empty dictionary.")
        payload[name] = {}
        for metric_name, value in parameter_metrics.items():
            if not isinstance(metric_name, str) or not metric_name:
                raise ValueError("Metric names must be non-empty strings.")
            if value is None:
                payload[name][metric_name] = None
                continue
            array = _finite_real_array(value, label=f"Metric {name!r}/{metric_name!r}")
            if array.ndim != 0:
                raise TypeError("Aggregate metrics must be scalars, not arrays.")
            scalar = float(array)
            if not np.isfinite(scalar):
                raise ValueError("Aggregate metrics must fit in a finite Python float.")
            payload[name][metric_name] = scalar
    return {"metrics": payload}
