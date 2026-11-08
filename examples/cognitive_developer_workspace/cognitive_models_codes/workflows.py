"""A small, explicit workspace workflow for usable checkpoint continuation.

Training delegates to the existing package. General evaluation also stays in
the package. Only resume differs: the current built-in implementation restores
an exhausted cosine schedule (learning rate zero), which prevents weight updates.
This workspace plugin preserves the optimizer object/state and gives it a new,
constant learning-rate schedule. No package function is patched or modified.
"""

import os
from datetime import UTC, datetime

import bayesflow as bf
import keras
import numpy as np

from bayesflow_models.interfaces import WorkflowResult
from bayesflow_models.workflow import (
    build_metadata_payload,
    build_workflow_context,
    ensure_context_directories,
    generate_next_alias,
    load_metadata,
    resolve_context_from_artifact_id,
    train_from_spec,
    write_metadata,
)

from .designs import positive_integer


def _seed_metadata() -> dict:
    return {
        "simulation_seed": int(os.environ.get("MODEL_DC_SIMULATION_SEED", "2023")),
        "network_seed": int(os.environ.get("MODEL_DC_NETWORK_SEED", "2023")),
        "simulation_stream_policy": "fresh_stream_for_each_cli_invocation",
    }


def train_model(spec, config):
    """Use package training unchanged, then annotate its metadata with seeds."""
    result = train_from_spec(spec, config)
    metadata = load_metadata(result.artifact_id, config)
    metadata["reproducibility"] = _seed_metadata()
    metadata["training"]["batches_per_epoch"] = config["training"]["n_sim"]
    write_metadata(
        metadata, build_workflow_context(spec, config, result.artifact_id).metadata_path
    )
    return result


def restart_learning_rate(optimizer, learning_rate: float) -> None:
    """Preserve Adam moments/iterations while replacing its finished schedule.

    Keras disallows assigning a float over an existing schedule. A serializable
    ExponentialDecay with decay_rate=1 is constant at every iteration, including
    the restored iteration count. No optimizer variables are reset or replaced.
    """
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Resume learning rate must be positive and finite.")
    optimizer.learning_rate = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=float(learning_rate),
        decay_steps=1,
        decay_rate=1.0,
    )


def resume_model(spec, config):
    """Continue the actual saved model; --mode new preserves the source artifact."""
    metadata = config["metadata"]
    if metadata["spec_name"] != spec.name or metadata.get("version") != spec.version:
        raise ValueError(
            "Checkpoint model name/version does not match this workspace model."
        )
    previous = resolve_context_from_artifact_id(metadata["artifact_id"], config)
    if not previous.checkpoint_path.is_file():
        raise FileNotFoundError(previous.checkpoint_path)
    mode = config.get("mode", "new")
    if mode not in {"new", "old"}:
        raise ValueError("Resume mode must be new or old.")
    epochs = positive_integer(config["training"]["resume_epochs"], "resume_epochs")
    batches = positive_integer(config["training"]["n_sim"], "batches_per_epoch")
    batch_size = positive_integer(config["training"]["batch_size"], "batch_size")
    learning_rate = float(os.environ.get("MODEL_DC_RESUME_LEARNING_RATE", "0.0005"))
    context = build_workflow_context(spec, config) if mode == "new" else previous
    ensure_context_directories(context)

    with keras.device(str(context.device)):
        approximator = keras.saving.load_model(previous.checkpoint_path)
        optimizer = approximator.optimizer
        if optimizer is None:
            raise RuntimeError(
                "Cannot preserve optimizer state: the checkpoint has no optimizer."
            )
        iterations_before = int(keras.ops.convert_to_numpy(optimizer.iterations))
        restart_learning_rate(optimizer, learning_rate)

        # Use the saved adapter and networks, not freshly initialized ones.
        workflow = bf.BasicWorkflow(
            simulator=spec.simulator_factory(),
            adapter=approximator.adapter,
            inference_network=approximator.inference_network,
            summary_network=approximator.summary_network,
            optimizer=optimizer,
            checkpoint_filepath=None,
        )
        workflow.approximator = approximator
        approximator.to(str(context.device))
        history = workflow.fit_online(
            epochs=epochs,
            batch_size=batch_size,
            num_batches_per_epoch=batches,
            keep_optimizer=True,
            max_queue_size=100,
        )
        approximator.save(context.checkpoint_path)
        iterations_after = int(keras.ops.convert_to_numpy(optimizer.iterations))

    alias = generate_next_alias(config) if mode == "new" else metadata["alias"]
    updated = build_metadata_payload(
        spec,
        context,
        config,
        status="resumed",
        alias=alias,
        created_at=None if mode == "new" else metadata["created_at"],
    )
    updated["training"].update(epochs=epochs, batches_per_epoch=batches)
    updated["reproducibility"] = _seed_metadata()
    updated["last_resumed_at"] = datetime.now(UTC).isoformat()
    updated["resume_policy"] = {
        "optimizer": "preserve_moments_and_iterations",
        "learning_rate": learning_rate,
        "learning_rate_policy": "restart_constant_schedule",
        "optimizer_iterations_before": iterations_before,
        "optimizer_iterations_after": iterations_after,
        "additional_epochs": epochs,
    }
    if mode == "new":
        updated["parent_artifact_id"] = previous.artifact_id
    elif "parent_artifact_id" in metadata:
        updated["parent_artifact_id"] = metadata["parent_artifact_id"]
    write_metadata(updated, context.metadata_path)
    print(
        f"Resumed {previous.artifact_id} -> {context.artifact_id}; learning rate {learning_rate}"
    )
    return WorkflowResult(
        artifact_id=context.artifact_id,
        spec_name=spec.name,
        workflow=spec.workflow,
        checkpoint_path=str(context.checkpoint_path),
        metadata_path=str(context.metadata_path),
        results_dir=str(context.results_dir),
        logs_dir=str(context.logs_dir),
        status="resumed",
        history=history,
    )
