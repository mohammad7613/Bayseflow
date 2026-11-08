
    

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bayesflow_models.evaluation.interfaces import EvaluationBatch, EvaluationSpec
from bayesflow_models.evaluation.metrics import compute_recovery_metrics, normalize_posterior_draws, build_recovery_result_payload
from bayesflow_models.evaluation.reporting import save_evaluation_arrays, save_recovery_plot
from bayesflow_models.interfaces import ModelSpec, WorkflowContext
from bayesflow_models.workflow import resolve_context_from_artifact_id, resolve_artifact_ref
from bayesflow_models.evaluation.evaluator import RecoveryEvaluator

from uuid import uuid4
import keras

def resolve_evaluation_artifact(
    *,
    artifact_ref: str,
    model_spec: ModelSpec,
    evaluation_spec: EvaluationSpec,
    config: dict,
) -> tuple[dict, WorkflowContext]:

      metadata = resolve_artifact_ref(artifact_ref, config)

      if metadata["spec_name"] != model_spec.name:
          raise ValueError(
              f"Artifact {metadata['artifact_id']!r} belongs to "
              f"model {metadata['spec_name']!r}, not "
              f"{model_spec.name!r}."
          )

      if evaluation_spec.model_name != model_spec.name:
          raise ValueError(
              f"Evaluation {evaluation_spec.name!r} belongs to "
              f"model {evaluation_spec.model_name!r}, not "
              f"{model_spec.name!r}."
          )

      artifact_id = metadata["artifact_id"]
      context = resolve_context_from_artifact_id(
          artifact_id,
          config,
      )

      if context.artifact_id != artifact_id:
          raise ValueError(
              "Resolved artifact context has an inconsistent ID."
          )

      if not context.checkpoint_path.is_file():
          raise FileNotFoundError(
              f"Checkpoint not found: {context.checkpoint_path}"
          )

      return metadata, context




def validate_evaluation_batch(
    *,
    batch: EvaluationBatch,
    model_spec: ModelSpec,
) -> int:
    """Validate the batch and return its dataset count."""

    if not isinstance(batch, EvaluationBatch):
        raise TypeError(
            "Evaluation generator must return an EvaluationBatch instance."
        )

    if not isinstance(batch.posterior_conditions, dict):
        raise TypeError(
            "EvaluationBatch.posterior_conditions must be a dictionary."
        )

    if not batch.posterior_conditions:
        raise ValueError("Posterior conditions cannot be empty.")

    if not isinstance(batch.true_parameters, dict):
        raise TypeError(
            "EvaluationBatch.true_parameters must be a dictionary."
        )

    expected = set(model_spec.par_names)
    actual = set(batch.true_parameters)

    if actual != expected:
        raise ValueError(
            f"Evaluation targets {sorted(actual)} do not match "
            f"model parameters {sorted(expected)}."
        )

    target_lengths: dict[str, int] = {}

    for name in model_spec.par_names:
        target = np.asarray(batch.true_parameters[name])

        if target.ndim == 0:
            raise ValueError(
                f"Evaluation target {name!r} must contain one value "
                "per dataset, not a scalar."
            )

        n_datasets = target.shape[0]

        if target.size != n_datasets:
            raise ValueError(
                f"Evaluation target {name!r} must contain exactly one "
                f"value per dataset; received shape {target.shape}."
            )

        target_lengths[name] = n_datasets

    unique_lengths = set(target_lengths.values())

    if len(unique_lengths) != 1:
        raise ValueError(
            "Evaluation targets must share one dataset dimension; "
            f"received {target_lengths}."
        )

    for item in batch.posterior_conditions.values():
        if not isinstance(item, np.ndarray):
            raise TypeError(
                "All posterior conditions must be NumPy arrays."
            )

        if item.shape[0] != n_datasets:
            raise ValueError(
                "All posterior conditions must contain exactly one "
                f"value per dataset; received shape {item.shape}."
            )

    n_datasets = next(iter(unique_lengths))

    if n_datasets < 1:
        raise ValueError(
            "An evaluation batch must contain at least one dataset."
        )

    if not isinstance(batch.metadata, dict):
        raise TypeError(
            "EvaluationBatch.metadata must be a dictionary."
        )

    try:
        json.dumps(batch.metadata)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "EvaluationBatch.metadata must be JSON-serializable."
        ) from exc

    return n_datasets



def load_artifact_approximator(
    checkpoint_path: Path,
)-> keras.Model:
    """Load the approximator from a checkpoint."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    return keras.saving.load_model(checkpoint_path)



def sample_posterior(
    *,
    approximator,
    batch: EvaluationBatch,
    n_posterior_samples: int,
):
    """Sample from the posterior using the approximator."""

    if not isinstance(n_posterior_samples, int):
        raise TypeError(
            "n_posterior_samples must be an integer."
        )

    if n_posterior_samples < 1:
        raise ValueError(
            "n_posterior_samples must be at least 1."
        )

    posterior_conditions = batch.posterior_conditions

    # Sample from the approximator
    posterior_samples =approximator.sample(
        conditions=posterior_conditions,
        num_samples=n_posterior_samples,
    )

    return posterior_samples


def create_evaluation_output_dir(
    *,
    results_dir: Path,
    evaluation_name: str,
) -> Path:
    """Create a directory for evaluation outputs."""

    evaluation_dir = (
        results_dir
        / "evaluations"
        / evaluation_name
    )

    evaluation_dir.mkdir(parents=True, exist_ok=True)

    return evaluation_dir


def build_evaluation_report_envelope(
    *,
    artifact_metadata: dict,
    model_spec: ModelSpec,
    evaluation_spec: EvaluationSpec,
    checkpoint_path: Path,
    batch: EvaluationBatch,
    n_datasets: int,
    n_posterior_samples: int,
    result_payload: dict,
    output_paths: dict[str, Path],
) -> dict:
    """Build and validate the JSON-safe envelope for an evaluation report."""

    # Preconditions: callers must provide the engine's established contract
    # types. Checking them here prevents unclear failures while assembling the
    # report and keeps this function free of filesystem side effects.
    if not isinstance(artifact_metadata, dict):
        raise TypeError("artifact_metadata must be a dictionary.")
    if not isinstance(model_spec, ModelSpec):
        raise TypeError("model_spec must be a ModelSpec instance.")
    if not isinstance(evaluation_spec, EvaluationSpec):
        raise TypeError(
            "evaluation_spec must be an EvaluationSpec instance."
        )
    if not isinstance(checkpoint_path, Path):
        raise TypeError("checkpoint_path must be a pathlib.Path instance.")
    if not isinstance(batch, EvaluationBatch):
        raise TypeError("batch must be an EvaluationBatch instance.")
    if not isinstance(result_payload, dict):
        raise TypeError("result_payload must be a dictionary.")
    if not isinstance(output_paths, dict):
        raise TypeError("output_paths must be a dictionary.")

    # Preconditions: these metadata fields form the minimum stable identity
    # needed by every evaluation report.
    required_metadata = {"artifact_id", "spec_name"}
    missing_metadata = required_metadata - set(artifact_metadata)
    if missing_metadata:
        raise ValueError(
            "Artifact metadata is missing required fields: "
            f"{sorted(missing_metadata)}."
        )

    # Preconditions: identity relationships should already have been checked
    # by artifact resolution, but repeating them at the report boundary stops
    # an internally inconsistent report from becoming durable output.
    if artifact_metadata["spec_name"] != model_spec.name:
        raise ValueError(
            f"Artifact belongs to model "
            f"{artifact_metadata['spec_name']!r}, not {model_spec.name!r}."
        )
    if evaluation_spec.model_name != model_spec.name:
        raise ValueError(
            f"Evaluation {evaluation_spec.name!r} belongs to model "
            f"{evaluation_spec.model_name!r}, not {model_spec.name!r}."
        )

    # Preconditions: execution counts are positive integers. An exact type
    # check rejects booleans, which Python otherwise treats as integers.
    if type(n_datasets) is not int:
        raise TypeError("n_datasets must be an integer.")
    if n_datasets < 1:
        raise ValueError("n_datasets must be at least 1.")
    if type(n_posterior_samples) is not int:
        raise TypeError("n_posterior_samples must be an integer.")
    if n_posterior_samples < 1:
        raise ValueError("n_posterior_samples must be at least 1.")

    # Preconditions: every output entry must have a stable string name and a
    # Path value. Paths are converted to strings for the JSON representation;
    # this pure function neither creates nor checks the corresponding files.
    serialized_output_paths: dict[str, str] = {}
    for name, path in output_paths.items():
        if not isinstance(name, str) or not name:
            raise TypeError(
                "Every output path name must be a non-empty string."
            )
        if not isinstance(path, Path):
            raise TypeError(
                f"Output path {name!r} must be a pathlib.Path instance."
            )
        serialized_output_paths[name] = str(path)

    # Preconditions: scenario metadata and result payload cross the JSON
    # persistence boundary. Raw NumPy arrays should be stored in NPZ output,
    # not silently converted inside the report envelope.
    try:
        json.dumps(batch.metadata)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "EvaluationBatch.metadata must be JSON-serializable."
        ) from exc
    try:
        json.dumps(result_payload)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "result_payload must be JSON-serializable."
        ) from exc

    report = {
        "artifact_id": artifact_metadata["artifact_id"],
        "alias": artifact_metadata.get("alias"),
        "spec_name": model_spec.name,
        "evaluation_name": evaluation_spec.name,
        "evaluation_description": evaluation_spec.description,
        "checkpoint_path": str(checkpoint_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_datasets": n_datasets,
        "n_posterior_samples": n_posterior_samples,
        "scenario_metadata": batch.metadata,
        "results": result_payload,
        "outputs": serialized_output_paths,
    }

    # Postcondition: the complete returned value—not only its individual
    # payloads—is JSON-serializable and therefore safe for write_json_report().
    # This check also catches non-serializable values in artifact identity or
    # EvaluationSpec fields.
    try:
        json.dumps(report)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "The completed evaluation report must be JSON-serializable."
        ) from exc

    # Postconditions on return:
    # - engine identity and execution fields occupy stable top-level keys;
    # - scenario-owned facts remain under scenario_metadata;
    # - evaluation-type results remain under results;
    # - every Path is represented as a string;
    # - no file or directory has been created by this function.
    return report


def write_json_report(
    *,
    report: dict,
    output_path: Path,
) -> None:

    """Write the evaluation report to a JSON file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f)



def run_evaluation(
    *,
    artifact_ref: str,
    model_spec: ModelSpec,
    evaluation_spec: EvaluationSpec,
    config: dict,
    options: dict,
    visualize: bool = False,
) -> dict:
    """Run parameter recovery, record its data, and save a recovery PDF."""

    # 1. Resolve the artifact and check model/scenario identity.
    artifact_metadata, context = resolve_evaluation_artifact(
        artifact_ref=artifact_ref,
        model_spec=model_spec,
        evaluation_spec=evaluation_spec,
        config=config,
    )

    # 2. Generate the experiment.
    # Homogeneous versus mixed TTA is decided by this function.
    batch = evaluation_spec.generate_batch(options)

    n_datasets = validate_evaluation_batch(
        batch=batch,
        model_spec=model_spec,
    )

    parameter_names = list(model_spec.par_names)

    n_posterior_samples = options.get("n_posterior_samples", 1000)
    if type(n_posterior_samples) is not int:
        raise TypeError("n_posterior_samples must be an integer.")
    if n_posterior_samples < 1:
        raise ValueError("n_posterior_samples must be at least 1.")

    # 3. Load the trained network and sample the posterior ONCE.
    # Truths stay separate from the posterior's input conditions.
    with keras.device(str(config.get("device", "cpu"))):
        approximator = load_artifact_approximator(
            checkpoint_path=context.checkpoint_path,
        )

        posterior_samples = sample_posterior(
            approximator=approximator,
            batch=batch,
            n_posterior_samples=n_posterior_samples,
        )

    # 4. Establish one common representation: (datasets, draws) per parameter.
    normalized_draws = normalize_posterior_draws(
        posterior_samples=posterior_samples,
        parameter_names=parameter_names,
        n_datasets=n_datasets,
        n_posterior_samples=n_posterior_samples,
    )

    # 5. Calculate numerical recovery results.
    # Use the EvaluationSpec INSTANCE, not the EvaluationSpec class.
    evaluator = evaluation_spec.evaluator
    if evaluator is None:
        evaluator = RecoveryEvaluator()

    result = evaluator.compute(
        posterior_samples=normalized_draws,
        batch=batch,
        parameter_names=parameter_names,
    )

    # 6. Create a unique output directory for this evaluation execution.
    evaluation_dir = create_evaluation_output_dir(
        results_dir=context.results_dir,
        evaluation_name=evaluation_spec.name,
    )

    evaluation_id = uuid4().hex
    output_dir = evaluation_dir / evaluation_id
    output_dir.mkdir(exist_ok=False)

    output_paths = {
        "arrays": output_dir / "evaluation_arrays.npz",
        "conditions": output_dir / "conditions.npz",
        "recovery_plot": output_dir / "recovery.pdf",
        "report": output_dir / "report.json",
    }

    # 7. Assemble the report and validate its JSON contents.
    report = build_evaluation_report_envelope(
        artifact_metadata=artifact_metadata,
        model_spec=model_spec,
        evaluation_spec=evaluation_spec,
        checkpoint_path=context.checkpoint_path,
        batch=batch,
        n_datasets=n_datasets,
        n_posterior_samples=n_posterior_samples,
        result_payload=result.result_payload,
        output_paths=output_paths,
    )

    report["evaluation_id"] = evaluation_id
    report["evaluator"] = evaluator.name
    report["parameter_names"] = parameter_names
    report["options"] = dict(options)

    json.dumps(report, allow_nan=False)

    # 8. Record numerical data before attempting visualization.
    save_evaluation_arrays(
        output_path=output_paths["arrays"],
        posterior_samples=normalized_draws,
        true_parameters=batch.true_parameters,
        posterior_summary=result.posterior_summary,
    )

    np.savez(
        output_paths["conditions"],
        **batch.posterior_conditions,
    )

    # 9. VISUALIZATION:
    # Convert to BayesFlow's dictionary representation.
    # Estimates: (datasets, draws, 1) per parameter.
    # Truths:    (datasets, 1) per parameter.
    plot_samples = {
        name: normalized_draws[name][..., np.newaxis]
        for name in parameter_names
    }

    plot_truths = {
        name: np.asarray(batch.true_parameters[name]).reshape(
            n_datasets, 1
        )
        for name in parameter_names
    }

    try:
        save_recovery_plot(
            posterior_samples=plot_samples,
            true_parameters=plot_truths,
            parameter_names=parameter_names,
            output_path=output_paths["recovery_plot"],
            visualize=visualize,
        )
    except Exception as exc:
        # Preserve the numerical results and explain the plotting failure.
        report["plotting"] = {
            "status": "failed",
            "error": str(exc),
        }
        write_json_report(
            report=report,
            output_path=output_paths["report"],
        )
        raise

    # 10. Finalize the report after successful plot saving/display.
    report["plotting"] = {"status": "completed"}

    write_json_report(
        report=report,
        output_path=output_paths["report"],
    )

    return report