"""Deterministic recovery examples; no training or checkpoint is needed."""

import json

import numpy as np
import pytest

from bayesflow_models.evaluation.metrics import (
    build_recovery_result_payload,
    compute_recovery_metrics,
    normalize_posterior_draws,
    summarize_posterior_draws,
)


@pytest.mark.parametrize("shape", [(2, 3), (1, 3), (3, 1), (1, 1), (2, 2)])
@pytest.mark.parametrize("feature_axis", [False, True])
def test_normalization_preserves_dataset_and_draw_axes(shape, feature_axis):
    # Unique values reveal a transpose even when the dataset/draw counts match.
    expected = np.arange(np.prod(shape), dtype=float).reshape(shape)
    original = expected[..., None] if feature_axis else expected
    before = original.copy()
    result = normalize_posterior_draws(
        posterior_samples={"a": original},
        parameter_names=["a"],
        n_datasets=shape[0],
        n_posterior_samples=shape[1],
    )
    np.testing.assert_array_equal(result["a"], expected)
    assert result["a"].shape == shape
    np.testing.assert_array_equal(original, before)


@pytest.mark.parametrize("field", ["n_datasets", "n_posterior_samples"])
@pytest.mark.parametrize(
    "count, error",
    [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_normalization_rejects_invalid_requested_counts(field, count, error):
    counts = {"n_datasets": 2, "n_posterior_samples": 3}
    counts[field] = count
    with pytest.raises(error, match=field):
        normalize_posterior_draws(
            posterior_samples={"a": np.zeros((2, 3))},
            parameter_names=["a"],
            **counts,
        )


@pytest.mark.parametrize(
    "shape", [(), (6,), (3, 2), (1, 3), (2, 0), (2, 3, 2), (2, 3, 1, 1)]
)
def test_normalization_rejects_unsupported_or_wrong_shapes(shape):
    with pytest.raises(ValueError, match="must have shape"):
        normalize_posterior_draws(
            posterior_samples={"a": np.zeros(shape)},
            parameter_names=["a"],
            n_datasets=2,
            n_posterior_samples=3,
        )


@pytest.mark.parametrize(
    "samples, error", [({}, KeyError), ({"a": [[1]], "extra": [[2]]}, ValueError)]
)
def test_normalization_checks_parameter_identity(samples, error):
    with pytest.raises(error):
        normalize_posterior_draws(
            posterior_samples=samples,
            parameter_names=["a"],
            n_datasets=1,
            n_posterior_samples=1,
        )


@pytest.mark.parametrize("names", [[], ["a", "a"], [""], [1]])
def test_normalization_rejects_invalid_parameter_declarations(names):
    with pytest.raises(ValueError, match="parameter_names"):
        normalize_posterior_draws(
            posterior_samples={"a": [[1]]},
            parameter_names=names,
            n_datasets=1,
            n_posterior_samples=1,
        )


@pytest.mark.parametrize(
    "value, error",
    [
        (np.nan, ValueError),
        (np.inf, ValueError),
        (-np.inf, ValueError),
        ("1", TypeError),
        (1j, TypeError),
        (True, TypeError),
        (None, TypeError),
    ],
)
def test_normalization_rejects_invalid_values(value, error):
    with pytest.raises(error):
        normalize_posterior_draws(
            posterior_samples={"a": [[value]]},
            parameter_names=["a"],
            n_datasets=1,
            n_posterior_samples=1,
        )


def test_summaries_match_hand_calculated_values():
    summary = summarize_posterior_draws(
        normalized_draws={"a": np.array([[0, 2, 4], [2, 4, 6]])}
    )["a"]
    np.testing.assert_allclose(summary["mean"], [2, 4])
    np.testing.assert_allclose(summary["std"], [np.sqrt(8 / 3)] * 2)
    np.testing.assert_allclose(summary["lower"], [0.1, 2.1])
    np.testing.assert_allclose(summary["upper"], [3.9, 5.9])


def test_custom_summary_interval():
    summary = summarize_posterior_draws(
        normalized_draws={"a": np.array([[0, 2, 4]])}, lower=0.25, upper=0.75
    )["a"]
    np.testing.assert_allclose(summary["lower"], [1])
    np.testing.assert_allclose(summary["upper"], [3])


@pytest.mark.parametrize(
    "bounds", [(-0.1, 0.9), (0.1, 1.1), (0.9, 0.1), (0.5, 0.5), (np.nan, 1)]
)
def test_summaries_reject_invalid_quantiles(bounds):
    with pytest.raises(ValueError, match="Quantiles"):
        summarize_posterior_draws(
            normalized_draws={"a": np.zeros((2, 3))}, lower=bounds[0], upper=bounds[1]
        )


@pytest.mark.parametrize(
    "draws",
    [
        {},
        {"a": np.zeros(3)},
        {"a": np.zeros((0, 3))},
        {"a": np.zeros((2, 0))},
        {"a": np.zeros((2, 3, 1))},
        {"a": np.zeros((2, 3)), "b": np.zeros((2, 4))},
        {"a": np.zeros((2, 3)), "b": np.zeros((1, 3))},
        {"a": np.array([[np.nan]])},
    ],
)
def test_summary_and_metric_entry_points_reject_noncanonical_draws(draws):
    with pytest.raises(ValueError):
        summarize_posterior_draws(normalized_draws=draws)
    with pytest.raises(ValueError):
        compute_recovery_metrics(normalized_draws=draws, true_parameters={})


def test_full_recovery_pipeline_matches_known_errors_and_does_not_mutate_inputs():
    # Means [2, 4] minus truths [1, 6] give errors [1, -2]. Only the first
    # truth is inside its interval: coverage is 1/2 and RMSE is sqrt(2.5).
    samples = {"a": np.array([[[0], [2], [4]], [[2], [4], [6]]])}
    truths = {"a": np.array([[1], [6]])}
    samples_before = samples["a"].copy()
    truths_before = truths["a"].copy()
    normalized = normalize_posterior_draws(
        posterior_samples=samples,
        parameter_names=["a"],
        n_datasets=2,
        n_posterior_samples=3,
    )
    metrics, summaries = compute_recovery_metrics(
        normalized_draws=normalized, true_parameters=truths
    )
    assert metrics["a"] == pytest.approx(
        {
            "bias": -0.5,
            "rmse": np.sqrt(2.5),
            "pearson": 1.0,
            "posterior_std_mean": np.sqrt(8 / 3),
            "coverage_95": 0.5,
        }
    )
    assert all(values.shape == (2,) for values in summaries["a"].values())
    assert all(type(value) is float for value in metrics["a"].values())
    payload = build_recovery_result_payload(metrics=metrics)
    assert json.loads(json.dumps(payload, allow_nan=False)) == {"metrics": metrics}
    np.testing.assert_array_equal(samples["a"], samples_before)
    np.testing.assert_array_equal(truths["a"], truths_before)


def test_perfect_recovery_and_parameter_specific_results():
    metrics, _ = compute_recovery_metrics(
        normalized_draws={
            "a": np.array([[1, 1], [2, 2]]),
            "b": np.array([[6, 6], [8, 8]]),
        },
        true_parameters={"a": np.array([1, 2]), "b": np.array([3, 5])},
    )
    assert metrics["a"]["bias"] == 0
    assert metrics["a"]["rmse"] == 0
    assert metrics["a"]["coverage_95"] == 1
    assert metrics["b"]["bias"] == 3
    assert metrics["b"]["rmse"] == 3
    assert metrics["b"]["coverage_95"] == 0


@pytest.mark.parametrize(
    "draws, truths",
    [
        (np.array([[2]]), np.array([2])),
        (np.array([[1], [2]]), np.array([1, 1])),
        (np.array([[2, 2], [2, 2]]), np.array([1, 2])),
    ],
)
def test_undefined_correlation_is_json_null(draws, truths):
    metrics, summaries = compute_recovery_metrics(
        normalized_draws={"a": draws}, true_parameters={"a": truths}
    )
    assert metrics["a"]["pearson"] is None
    np.testing.assert_array_equal(summaries["a"]["std"], np.zeros(draws.shape[0]))
    serialized = json.dumps(
        build_recovery_result_payload(metrics=metrics), allow_nan=False
    )
    assert json.loads(serialized)["metrics"]["a"]["pearson"] is None


@pytest.mark.parametrize(
    "truths, error",
    [
        ({}, KeyError),
        ({"a": [1, 2], "extra": [1, 2]}, ValueError),
        ({"a": 1}, ValueError),
        ({"a": [1]}, ValueError),
        ({"a": [[1, 2]]}, ValueError),
        ({"a": [[1, 2], [3, 4]]}, ValueError),
        ({"a": [np.nan, 2]}, ValueError),
        ({"a": [np.inf, 2]}, ValueError),
        ({"a": ["1", "2"]}, TypeError),
    ],
)
def test_metrics_reject_truths_that_could_broadcast_or_misidentify_parameters(
    truths, error
):
    with pytest.raises(error):
        compute_recovery_metrics(
            normalized_draws={"a": np.zeros((2, 3))}, true_parameters=truths
        )


def test_payload_converts_numpy_scalars_and_is_independent_of_metrics():
    metrics = {"a": {"bias": np.float32(0.5), "pearson": None}}
    payload = build_recovery_result_payload(metrics=metrics)
    assert type(payload["metrics"]["a"]["bias"]) is float
    payload["metrics"]["a"]["bias"] = 99
    assert metrics["a"]["bias"] == 0.5


@pytest.mark.parametrize(
    "value, error",
    [
        (np.nan, ValueError),
        (np.inf, ValueError),
        (np.array([1]), TypeError),
        ("1", TypeError),
        (True, TypeError),
    ],
)
def test_payload_rejects_nonfinite_or_nonscalar_metrics(value, error):
    with pytest.raises(error):
        build_recovery_result_payload(metrics={"a": {"bias": value}})


def test_finite_inputs_that_overflow_fail_before_creating_invalid_results():
    with pytest.raises(ValueError, match="floating-point limits"):
        summarize_posterior_draws(normalized_draws={"a": np.array([[1e308, 1e308]])})
    with pytest.raises(ValueError, match="floating-point limits"):
        compute_recovery_metrics(
            normalized_draws={"a": np.array([[0.0]])},
            true_parameters={"a": np.array([1e308])},
        )
