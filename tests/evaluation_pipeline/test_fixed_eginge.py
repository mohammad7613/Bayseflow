from bayesflow_models.evaluation import fixed_engine as engine

import json

# Python standard library
import json
from pathlib import Path

# Installed testing/numerical packages
import numpy as np
import pytest

# The module whose functions we are testing
from bayesflow_models.evaluation import fixed_engine as engine

# Data structures used to construct test inputs
from bayesflow_models.evaluation.interfaces import (
    EvaluationBatch,
    EvaluationSpec,
)

from bayesflow_models.interfaces import ModelSpec


import numpy as np

from bayesflow_models.interfaces import ModelSpec
from bayesflow_models.evaluation.interfaces import (
    EvaluationBatch,
    EvaluationSpec,
)


def make_model_spec(name="test_model"):
    """Describe a test model with two scalar parameters: a and b."""
    return ModelSpec(
        name=name,
        workflow="builtin",
        # This test replaces checkpoint loading, so these factories
        # do not need to construct real simulators or adapters.
        simulator_factory=lambda: object(),
        adapter_factory=lambda: object(),
        family="test",
        par_names=["a", "b"],
        version="1.0",
    )


def make_batch():
    """Create two datasets with known parameter truths."""
    return EvaluationBatch(
        posterior_conditions={
            # Two datasets, three trials each, one observation feature.
            "x": np.zeros((2, 3, 1)),
        },
        true_parameters={
            "a": np.array([1.0, 2.0]),
            "b": np.array([3.0, 4.0]),
        },
        metadata={"seed": 7},
    )


def make_evaluation_spec(
    *,
    model_name="test_model",
    generate_batch=None,
):
    """Create a scenario using either the default or supplied generator."""
    if generate_batch is None:
        generate_batch = lambda model_spec, options: make_batch()

    return EvaluationSpec(
        name="test-scenario",
        model_name=model_name,
        generate_batch=generate_batch,
        description="Evaluation scenario used by tests.",
    )


def test_run_evaluation_saves_results(tmp_path, monkeypatch):
    # ARRANGE: create an isolated workspace.
    # tmp_path is supplied by pytest; nothing touches real checkpoints/results.
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    checkpoint_path = checkpoint_dir / "artifact-001.keras"

    # This is only a placeholder. We replace model loading below,
    # so the file will never be deserialized as a real Keras checkpoint.
    checkpoint_path.touch()

    metadata = {
        "artifact_id": "artifact-001",
        "alias": "run-001",
        "spec_name": "test_model",
        "status": "trained",
        "created_at": "2026-10-29T00:00:00+00:00",
    }
    metadata_path = checkpoint_dir / "artifact-001.meta.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    config = {
        "device": "cpu",
        "paths": {
            "checkpoints": str(checkpoint_dir),
            "results": str(tmp_path / "results"),
            "logs": str(tmp_path / "logs"),
        },
    }

    model_spec = make_model_spec()
    batch = make_batch()
    options = {"n_posterior_samples": 3, "seed": 7}

    def generate_batch(selected_model, selected_options):
        # Check that the developer's generator receives the right inputs.
        assert selected_model is model_spec
        assert selected_options == options
        return batch

    evaluation_spec = make_evaluation_spec(
        generate_batch=generate_batch,
    )

    class FakeApproximator:
        def sample(self, *, conditions, num_samples):
            # Truths must not be supplied to posterior sampling.
            assert conditions is batch.posterior_conditions
            assert "a" not in conditions
            assert "b" not in conditions
            assert num_samples == 3

            # Truths from make_batch():
            # a = [1, 2]; b = [3, 4].
            # a is recovered perfectly; b is consistently overestimated by 1.
            return {
                "a": np.array([[1., 1., 1.], [2., 2., 2.]])[..., None],
                "b": np.array([[4., 4., 4.], [5., 5., 5.]])[..., None],
            }

    def fake_load(checkpoint_path):
        assert checkpoint_path == checkpoint_dir / "artifact-001.keras"
        return FakeApproximator()

    # Patch the function where run_evaluation looks it up.
    # Everything else—resolution, validation, metrics, and saving—runs normally.
    monkeypatch.setattr(engine,"load_artifact_approximator",fake_load,)

    # ACT: execute the actual runner.
    report = engine.run_evaluation(
        artifact_ref="run-001",
        model_spec=model_spec,
        evaluation_spec=evaluation_spec,
        config=config,
        options=options,
    )

    # ASSERT: verify numerical results and recorded configuration.
    metrics = report["results"]["metrics"]

    assert metrics["a"]["rmse"] == pytest.approx(0.0)
    assert metrics["b"]["bias"] == pytest.approx(1.0)
    assert metrics["b"]["rmse"] == pytest.approx(1.0)
    assert report["options"] == options
    assert report["n_datasets"] == 2
    assert report["n_posterior_samples"] == 3

    # Read the actual saved report; returning a dictionary alone is insufficient.
    report_path = Path(report["outputs"]["report"])
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report == report

    # Check saved numerical content, not only file existence.
    with np.load(report["outputs"]["arrays"], allow_pickle=False) as saved:
        assert saved["posterior_a"].shape == (2, 3)
        np.testing.assert_array_equal(saved["true_b"], [3., 4.])
        np.testing.assert_array_equal(saved["posterior_mean_b"], [4., 5.])

    with np.load(report["outputs"]["conditions"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(
            saved["x"],
            batch.posterior_conditions["x"],
        )

    # A second invocation must preserve the first invocation's results.
    second_report = engine.run_evaluation(
        artifact_ref="run-001",
        model_spec=model_spec,
        evaluation_spec=evaluation_spec,
        config=config,
        options=options,
    )

    assert second_report["evaluation_id"] != report["evaluation_id"]
    assert second_report["outputs"]["report"] != str(report_path)
    assert report_path.is_file()