from bayesflow_models.evaluation.interfaces import EvaluationSpec
from bayesflow_models.interfaces import ModelSpec
from bayesflow_models.workflow import resolve_artifact_ref,resolve_context_from_artifact_id,ensure_context_directories,build_model_from_spec
import keras

from metrics import compute_recovery_metrics
from reporting import save_recovery_plot, save_evaluation_arrays, create_evaluation_output_dir, build_evaluation_report

def run_evaluation(
    *,
    artifact_ref: str,
    spec: ModelSpec,
    evaluation_spec: EvaluationSpec,
    config: dict,
    options: dict,
) -> dict:


    ## Valiation 
    metadata = resolve_artifact_ref(artifact_ref=artifact_ref, config=config)

    ## Check that the artifact belongs to spec.name.
    if metadata["model_name"] != spec.name:
        raise ValueError(
            f"Artifact {artifact_ref} belongs to model {metadata['model_name']}, "
            f"but evaluation is for model {spec.name}"
        )



    ## Resolve Checkpoint Path and load Model 
    spec_name=spec.name
    artifact_id = metadata["artifact_id"]

    ctx = resolve_context_from_artifact_id(artifact_id, config)

    if not ctx.checkpoint_path.exists():
      raise FileNotFoundError(
          f"Checkpoint not found: {ctx.checkpoint_path}"
      )

    assert ctx.artifact_id == artifact_id

    if metadata["spec_name"] != spec_name:
        raise ValueError(
            f"Artifact {artifact_id} belongs to spec {metadata['spec_name']}, not {spec.name}"
        )

    if evaluation_spec.model_name != spec.name:
      raise ValueError(
          f"Evaluation {evaluation_spec.name!r} belongs to "
          f"{evaluation_spec.model_name!r}, not {spec.name!r}"
      )
    approximator = keras.saving.load_model(ctx.checkpoint_path)



    ## Generate Evaluation Batch

    batch = evaluation_spec.generate_batch(spec, options)

    expected = set(spec.par_names)
    actual = set(batch.true_parameters)

    if actual != expected:
      raise ValueError(
          f"Evaluation targets {sorted(actual)} do not match "
          f"model parameters {sorted(expected)}"
      )

    if not batch.posterior_conditions:
        raise ValueError("Posterior conditions cannot be empty.")




    ## Generate posterior samples using the approximator and the posterior conditions from the batch
    posterior_samples = approximator.sample(
        batch.posterior_conditions,
        num_samples=config.get("num_posterior_samples", 1000),
    )


    ## Compute Recovery Metrics
    metrics, posterior_summary = compute_recovery_metrics(
      posterior_samples=posterior_samples,
      true_parameters=batch.true_parameters,
      parameter_names=spec.par_names,
    )

    ### Reporting 
    output_dir = create_evaluation_output_dir(
        ctx.results_dir,
        evaluation_spec.name,
    )

    output_paths = {
        "recovery_plot": output_dir / "recovery.pdf",
        "arrays": output_dir / "evaluation.npz",
        "report": output_dir / "report.json",
    }

    save_recovery_plot(
        posterior_samples=posterior_samples,
        true_parameters=batch.true_parameters,
        parameter_names=spec.par_names,
        output_path=output_paths["recovery_plot"],
        visualize=options.get("visualize", False),
    )

    save_evaluation_arrays(
        output_path=output_paths["arrays"],
        posterior_samples=posterior_samples,
        true_parameters=batch.true_parameters,
        posterior_summary=posterior_summary,
    )

    report = build_evaluation_report(
        artifact_metadata=metadata,
        model_spec=spec,
        evaluation_spec=evaluation_spec,
        checkpoint_path=ctx.checkpoint_path,
        batch=batch,
        n_posterior_samples=n_posterior_samples,
        metrics=metrics,
        output_paths=output_paths,
    )





