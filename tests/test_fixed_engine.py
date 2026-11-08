import json
from pathlib import Path

import numpy as np
import pytest


from pathlib import Path

from bayesflow_models.evaluation.fixed_engine import (build_evaluation_report_envelope, 
                                                      create_evaluation_output_dir, 
                                                      load_artifact_approximator, 
                                                      resolve_artifact_ref,
                                                      resolve_context_from_artifact_id,
                                                      sample_posterior,
                                                      validate_evaluation_batch,
                                                      write_json_report,)

from bayesflow_models.evaluation.interfaces import EvaluationBatch, EvaluationSpec

from bayesflow_models.interfaces import ModelSpec, Workflow 



## Test-data factories 
def make_model_spec(name="test_model"):
    return ModelSpec(
        name=name,
        workflow="builtin",
        simulator_factory=lambda: object(),
        adapter_factory=lambda: object(),
        family="test",
        par_names=["a", "b"],
        version="1.0",
    )

def make_batch():
    return EvaluationBatch(
        posterior_conditions={
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
    if generate_batch is None:
        generate_batch = (
            lambda model_spec, options: make_batch()
        )

    return EvaluationSpec(
        name="test-scenario",
        model_name=model_name,
        generate_batch=generate_batch,
        description="Test evaluation",
    )


## Test validation of EvaluationBatch
def test_validate_evaluation_batch_valid():
    count = validate_evaluation_batch(batch=make_batch(), model_spec=make_model_spec()
    )

    assert count == 2


def test_validate_evaluation_batch_invalid():
    batch = make_batch()
    batch.posterior_conditions["x"] = np.zeros((1, 3, 1))  # Invalid shape

    with pytest.raises(ValueError):
        validate_evaluation_batch(batch=batch, model_spec=make_model_spec())



def test_create_evaluation_output_dir(tmp_path):
    artifact_results = tmp_path / "artifact-001"

    result = create_evaluation_output_dir(
        results_dir=artifact_results,
        evaluation_name="mixed-tta",
    )

    assert result == (
        artifact_results
        / "evaluations"
        / "mixed-tta"
    )
    assert result.is_dir()



def test_build_report_envelope_is_json_serializable(tmp_path):
    model_spec = make_model_spec()
    evaluation_spec = make_evaluation_spec()

    report = build_evaluation_report_envelope(
        artifact_metadata={
            "artifact_id": "artifact-001",
            "alias": "run-001",
            "spec_name": model_spec.name,
        },
        model_spec=model_spec,
        evaluation_spec=evaluation_spec,
        checkpoint_path=tmp_path / "artifact.keras",
        batch=make_batch(),
        n_datasets=2,
        n_posterior_samples=5,
        result_payload={
            "metrics": {
                "a": {"rmse": 0.1},
            },
        },
        output_paths={
            "arrays": tmp_path / "evaluation.npz",
        },
    )

    json.dumps(report)

    assert report["artifact_id"] == "artifact-001"
    assert report["scenario_metadata"] == {"seed": 7}
    assert report["results"]["metrics"]["a"]["rmse"] == 0.1
    assert isinstance(report["outputs"]["arrays"], str)



def test_write_json_report(tmp_path):
    output = tmp_path / "nested" / "report.json"
    report = {"artifact_id": "artifact-001"}

    write_json_report(
        report=report,
        output_path=output,
    )

    assert json.loads(
        output.read_text(encoding="utf-8")
    ) == report