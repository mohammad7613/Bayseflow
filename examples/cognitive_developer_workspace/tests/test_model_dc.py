"""Small scientific/encoding tests. The real CLI integration is smoke_test.py."""

import ast
from pathlib import Path

import numpy as np
import pytest
from cognitive_models_codes.adapters import make_adapter
from cognitive_models_codes.designs import homogeneous_schedule, mixed_schedule
from cognitive_models_codes.science import (
    PARAMETER_NAMES,
    TTA_LEVELS,
    sample_prior,
    simulate_dataset,
)
from cognitive_models_codes.simulators import make_training_simulator


@pytest.mark.parametrize("seed, tta", [(7, 2.5), (2023, 4.0)])
def test_matches_previous_prior_and_trial_equations(seed, tta):
    # Read the original function definitions as an independent reference.
    # Do not import its module: that would construct all its other models too.
    source_path = (
        Path(__file__).resolve().parents[2]
        / "article_workspace/plugins/reproduce_models.py"
    )
    if not source_path.exists():
        pytest.skip("Original article source is unavailable in a standalone copy.")
    tree = ast.parse(source_path.read_text())
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"prior_DC", "simulator_DC"}
    ]
    reference = {"np": np, "RNG": np.random.default_rng(seed)}
    exec(  # noqa: S102 - only the trusted local reference function definitions
        compile(
            ast.Module(body=definitions, type_ignores=[]), str(source_path), "exec"
        ),
        reference,
    )
    rng = np.random.default_rng(seed)
    parameters = sample_prior(rng)
    assert parameters == reference["prior_DC"]()
    original = reference["simulator_DC"](
        tta_condition=tta, number_of_trials=12, **parameters
    )
    current = simulate_dataset(parameters, homogeneous_schedule(12, tta), rng)
    for key in ("x", "tta_per_trial"):
        np.testing.assert_array_equal(current[key], original[key])


def test_mixed_schedule_preserves_counts_and_seed():
    levels = np.asarray(TTA_LEVELS)
    first = mixed_schedule(levels, 15, np.random.default_rng(7), shuffle=True)
    second = mixed_schedule(levels, 15, np.random.default_rng(7), shuffle=True)
    np.testing.assert_array_equal(first, second)
    values, counts = np.unique(first, return_counts=True)
    np.testing.assert_array_equal(values, levels)
    np.testing.assert_array_equal(counts, [15, 15, 15, 15])


def test_training_simulator_shape_and_fresh_rng(monkeypatch):
    monkeypatch.setenv("MODEL_DC_SIMULATION_SEED", "2023")
    first = make_training_simulator().sample(2)
    second = make_training_simulator().sample(2)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert first["x"].shape == (2, 60, 1)
    assert first["tta_per_trial"].shape == (2, 60, 1)
    for name in PARAMETER_NAMES:
        assert first[name].shape == (2, 1)
    assert np.all(first["tta_condition"] == first["tta_condition"][0])


def test_adapter_handles_inference_without_truths():
    batch = make_training_simulator().sample(2)
    adapter = make_adapter()
    adapted = adapter(batch)  # Training initializes parameter split indices.
    assert adapted["inference_variables"].shape == (2, 7)
    assert adapted["summary_variables"].shape == (2, 60, 2)
    assert adapted["summary_variables"].dtype == np.float32
    np.testing.assert_array_equal(
        adapted["summary_variables"][..., 0], batch["x"][..., 0].astype("float32")
    )
    np.testing.assert_array_equal(
        adapted["summary_variables"][..., 1],
        batch["tta_per_trial"][..., 0].astype("float32"),
    )

    conditions = {key: batch[key] for key in ("x", "tta_per_trial", "number_of_trials")}
    inferred = adapter(conditions, strict=False)
    assert "inference_variables" not in inferred
    np.testing.assert_array_equal(
        inferred["summary_variables"], adapted["summary_variables"]
    )
