import os
os.environ.setdefault("KERAS_BACKEND", "torch")

from bayesflow_models.interfaces import ModelSpec
from bayesflow_models.train import train_amortizer_resume
import secrets
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from bayesflow_models.interfaces import ModelSpec, WorkflowContext, WorkflowResult
import json
import keras

from bayesflow.diagnostics.plots import recovery
import matplotlib.pyplot as plt


def validate_spec(spec: ModelSpec) -> None:
    if not isinstance(spec, ModelSpec):
        raise TypeError(f"Expected ModelSpec, got {type(spec)!r}")

    if not spec.name:
        raise ValueError("ModelSpec.name must not be empty")

    if not spec.workflow:
        raise ValueError(f"ModelSpec {spec.name!r} must define workflow")

    if not callable(spec.simulator_factory):
        raise ValueError(f"ModelSpec {spec.name!r} has non-callable simulator_factory")

    if not callable(spec.adapter_factory):
        raise ValueError(f"ModelSpec {spec.name!r} has non-callable adapter_factory")

def build_model_from_spec(spec: ModelSpec)-> tuple:
    validate_spec(spec)
    simulator = spec.simulator_factory()
    adapter = spec.adapter_factory()
    return simulator, adapter


def generate_artifact_id(spec_name: str, now: datetime | None = None) -> str:
    now = now or datetime.now()
    timestamp = now.strftime("%Y%m%dT%H%M%S")
    suffix = secrets.token_hex(4)
    return f"{spec_name}__{timestamp}__{suffix}"


def iter_metadata_files(config: dict) -> list[Path]:
    checkpoint_dir = Path(config["paths"]["checkpoints"])

    if not checkpoint_dir.exists():
        return []

    return sorted(checkpoint_dir.glob("*.meta.json"))


def read_metadata_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Metadata file must contain a JSON object: {path}")
    
    
    required_keys = {"artifact_id", "spec_name", "status", "created_at"}
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(f"Metadata file {path} is missing keys: {sorted(missing)}")

    return data

def generate_next_alias(config: dict) -> str:
    max_index = 0

    for meta_path in iter_metadata_files(config):
        try:
            metadata = read_metadata_file(meta_path)
        except Exception:
            # Ignore malformed metadata for alias generation
            continue

        alias = metadata.get("alias")
        if not isinstance(alias, str):
            continue

        if not alias.startswith("run-"):
            continue

        number_part = alias[4:]
        if not number_part.isdigit():
            continue

        max_index = max(max_index, int(number_part))

    next_index = max_index + 1
    return f"run-{next_index:03d}"



def build_workflow_context(
    spec: ModelSpec,
    config: dict,
    artifact_id: str | None = None,
) -> WorkflowContext:
    artifact_id = artifact_id or generate_artifact_id(spec.name)

    checkpoint_dir = Path(config["paths"]["checkpoints"])
    results_root = Path(config["paths"]["results"])
    logs_root = Path(config["paths"]["logs"])

    checkpoint_path = checkpoint_dir / f"{artifact_id}.keras"
    metadata_path = checkpoint_dir / f"{artifact_id}.meta.json"
    results_dir = results_root / artifact_id
    logs_dir = logs_root / artifact_id

    return WorkflowContext(
        artifact_id=artifact_id,
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        results_dir=results_dir,
        logs_dir=logs_dir,
        device=config["device"],
    )



def ensure_context_directories(ctx: WorkflowContext) -> None:
    ctx.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.results_dir.mkdir(parents=True, exist_ok=True)
    ctx.logs_dir.mkdir(parents=True, exist_ok=True)

def build_metadata_payload(
    spec: ModelSpec,
    ctx: WorkflowContext,
    config: dict,
    status: str,
    alias: str,
    created_at: str | None = None,
    stage: str| None = None
) -> dict:
    now = datetime.now().isoformat()
    
    return {
        "artifact_id": ctx.artifact_id,
        "alias": alias,
        "spec_name": spec.name,
        "workflow": spec.workflow,
        "family": spec.family,
        "version": spec.version,
        "description": spec.description,
        "checkpoint_path": str(ctx.checkpoint_path),
        "metadata_path": str(ctx.metadata_path),
        "results_dir": str(ctx.results_dir),
        "logs_dir": str(ctx.logs_dir),
        "status": status,
        "created_at": created_at or now,
        "updated_at": now,
        "training": {
            "n_sim": config["training"]["n_sim"],
            "epochs": config["training"]["epochs"],
            "batch_size": config["training"]["batch_size"],
        },
    }

def write_metadata(payload: dict, metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def load_metadata(artifact_id: str, config: dict) -> dict:
    metadata_path = Path(config["paths"]["checkpoints"]) / f"{artifact_id}.meta.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found for artifact_id={artifact_id}: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_metadata_alias(metadata: dict, metadata_path: Path, config: dict) -> dict:
      alias = metadata.get("alias")

      if isinstance(alias, str) and alias.startswith("run-"):
          return metadata

      alias = generate_next_alias(config)
      metadata["alias"] = alias
      write_metadata(metadata, metadata_path)
      return metadata

def resolve_context_from_artifact_id(artifact_id: str, config: dict) -> WorkflowContext:
    metadata = load_metadata(artifact_id, config)

    checkpoint_root = Path(config["paths"]["checkpoints"])
    results_root = Path(config["paths"]["results"])
    logs_root = Path(config["paths"]["logs"])

    return WorkflowContext(
        artifact_id=artifact_id,
        checkpoint_path=checkpoint_root / f"{artifact_id}.keras",
        metadata_path=checkpoint_root / f"{artifact_id}.meta.json",
        results_dir=results_root / artifact_id,
        logs_dir=logs_root / artifact_id,
        device=config["device"],
    )


## Visualization Helper functions 
def format_created_timestamp(value: str) -> str:
      if value in ["",'-']:
          return value
      dt = datetime.fromisoformat(value)
      return dt.strftime("%Y-%m-%d %H:%M")


def list_checkpoint_records(config: dict) -> list[dict]:
      records = []

      for metadata_path in iter_metadata_files(config):
          try:
              metadata = read_metadata_file(metadata_path)
              metadata = ensure_metadata_alias(metadata, metadata_path, config)
          except Exception:
              continue

          records.append({
              "alias": metadata["alias"],
              "artifact_id": metadata["artifact_id"],
              "spec_name": metadata["spec_name"],
              "status": metadata.get("status", "unknown"),
              "resumed_display": format_created_timestamp(metadata.get("last_resumed_at","")),
              "created_display": format_created_timestamp(metadata.get("created_at", "")),
              "metadata_path": str(metadata_path),
          })

      def alias_key(row):
          alias = row["alias"]
          if alias.startswith("run-") and alias[4:].isdigit():
              return int(alias[4:])
          return 10**9

      records.sort(key=alias_key)
      return records


def resolve_artifact_ref(ref: str, config: dict) -> dict:
      # First try exact artifact id
      metadata_path = Path(config["paths"]["checkpoints"]) / f"{ref}.meta.json"
      if metadata_path.exists():
          return read_metadata_file(metadata_path)

      # Then try alias lookup
      for meta_path in iter_metadata_files(config):
          try:
              metadata = read_metadata_file(meta_path)
              metadata = ensure_metadata_alias(metadata, meta_path, config)
          except Exception:
              continue

          if metadata.get("alias") == ref:
              return metadata

      raise FileNotFoundError(f"No artifact found for reference: {ref}")






def train_from_spec(spec: ModelSpec, config: dict) -> WorkflowResult:
    validate_spec(spec)
    model = build_model_from_spec(spec)
    ctx = build_workflow_context(spec, config)
    ensure_context_directories(ctx)

    history = train_amortizer_resume(
        model=model,
        model_name=ctx.artifact_id,
        n_sim=config["training"]["n_sim"],
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        initial_lr=5e-4,
        checkpoint_dir=config["paths"]["checkpoints"],
        device=ctx.device
    )

    result = WorkflowResult(
        artifact_id=ctx.artifact_id,
        spec_name=spec.name,
        workflow=spec.workflow,
        checkpoint_path=str(ctx.checkpoint_path),
        metadata_path=str(ctx.metadata_path),
        results_dir=str(ctx.results_dir),
        logs_dir=str(ctx.logs_dir),
        status="trained",
        history=history,
    )
    alias = generate_next_alias(config)
    metadata = build_metadata_payload(spec, ctx, config, status="trained", alias=alias)
    write_metadata(metadata, ctx.metadata_path)

    return result



def resume_from_artifact(spec: ModelSpec, config: dict) -> WorkflowResult:
    metadata = config["metadata"]
    spec_name=spec.name
    artifact_id = metadata["artifact_id"]
    oldctx = resolve_context_from_artifact_id(artifact_id, config)
    

    assert oldctx.artifact_id == artifact_id
    if not oldctx.checkpoint_path.exists():
      raise FileNotFoundError(f"Checkpoint not found for artifact {artifact_id}: {oldctx.checkpoint_path}")

    if metadata["spec_name"] != spec_name:
        raise ValueError(
            f"Artifact {artifact_id} belongs to spec {metadata['spec_name']}, not {spec.name}"
        )

    model = build_model_from_spec(spec)
    
    
    if config["mode"] == "new":
        ctx = build_workflow_context(spec, config)
        ensure_context_directories(ctx)
        
        history = train_amortizer_resume(
            model=model,
            model_name=metadata["artifact_id"],
            n_sim=config["training"]["n_sim"],
            epochs=config["training"]["resume_epochs"],
            batch_size=config["training"]["batch_size"],
            initial_lr=5e-4,
            checkpoint_dir=config["paths"]["checkpoints"],
            checkpoint_save=str(ctx.checkpoint_path),
            device=ctx.device
        )
        result = WorkflowResult(
        artifact_id=ctx.artifact_id,
        spec_name=spec.name,
        workflow=spec.workflow,
        checkpoint_path=str(ctx.checkpoint_path),
        metadata_path=str(ctx.metadata_path),
        results_dir=str(ctx.results_dir),
        logs_dir=str(ctx.logs_dir),
        status="resumed",
        history=history,
        )
        alias = generate_next_alias(config)
        metadata = build_metadata_payload(spec, ctx, config, status="resumed", alias=alias)
        metadata["training"]["epochs"] = config["training"]["resume_epochs"]
        metadata["parent_artifact_id"] = oldctx.artifact_id
        metadata["last_resumed_at"] = datetime.now().isoformat()
        write_metadata(metadata, ctx.metadata_path)
    else:
        ctx = oldctx
        history = train_amortizer_resume(
            model=model,
            model_name=metadata["artifact_id"],
            n_sim=config["training"]["n_sim"],
            epochs=config["training"]["resume_epochs"],
            batch_size=config["training"]["batch_size"],
            initial_lr=5e-4,
            checkpoint_dir=config["paths"]["checkpoints"],
            device=ctx.device
        )

        result = WorkflowResult(
            artifact_id=ctx.artifact_id,
            spec_name=spec.name,
            workflow=spec.workflow,
            checkpoint_path=str(ctx.checkpoint_path),
            metadata_path=str(ctx.metadata_path),
            results_dir=str(ctx.results_dir),
            logs_dir=str(ctx.logs_dir),
            status="resumed",
            history=history,
        )


        updated_metadata = build_metadata_payload(
            spec=spec,
            ctx=ctx,
            config=config,
            status="resumed",
            alias=metadata.get("alias"),
            created_at=metadata["created_at"],
        )
        updated_metadata["training"]["epochs"] = config["training"]["resume_epochs"]
        updated_metadata["last_resumed_at"] = datetime.now().isoformat()

        write_metadata(updated_metadata, ctx.metadata_path)
    return result


def recovery_from_artifact(spec: ModelSpec, config: dict) -> WorkflowResult:
    metadata = config["metadata"]
    spec_name=spec.name
    artifact_id = metadata["artifact_id"]

    ctx = resolve_context_from_artifact_id(artifact_id, config)

    assert ctx.artifact_id == artifact_id

    if metadata["spec_name"] != spec_name:
        raise ValueError(
            f"Artifact {artifact_id} belongs to spec {metadata['spec_name']}, not {spec.name}"
        )

    simulator,adaptor = build_model_from_spec(spec)
    ensure_context_directories(ctx)
    approximator = keras.saving.load_model(ctx.checkpoint_path)



    # Simulate validation data (unseen during training)
    val_sims = simulator.sample(config["recovery"]["n_test_sims"])

    post_draws1 = approximator.sample(conditions=val_sims, num_samples=config["recovery"]["n_posterior_samples"])
    par_names = spec.par_names

    f = recovery(
        estimates=post_draws1,
        targets=val_sims,
        variable_names=par_names
    )
    
    pdf_path = ctx.results_dir / "recovery.pdf"

    # BayesFlow recovery(...) returns a matplotlib figure-like object
    f.savefig(pdf_path, bbox_inches="tight")
    if config["mode"] == "visualize":
       plt.show()
    else:   
       plt.close(f)
    

    # save image in result directory



    # history = train_amortizer_resume(
    #     model=model,
    #     model_name=metadata["artifact_id"],
    #     n_sim=config["training"]["n_sim"],
    #     epochs=config["training"]["resume_epochs"],
    #     batch_size=config["training"]["batch_size"],
    #     initial_lr=5e-4,
    #     checkpoint_dir=config["paths"]["checkpoints"],
    # )

    result = WorkflowResult(
        artifact_id=ctx.artifact_id,
        spec_name=spec.name,
        workflow=spec.workflow,
        checkpoint_path=str(ctx.checkpoint_path),
        metadata_path=str(ctx.metadata_path),
        results_dir=str(ctx.results_dir),
        logs_dir=str(ctx.logs_dir),
        status=metadata['status']
    )


    # updated_metadata = build_metadata_payload(
    #     spec=spec,
    #     ctx=ctx,
    #     config=config,
    #     status=metadata["status"],
    #     alias=metadata.get("alias"),
    #     created_at=metadata["created_at"],
    # )
    # updated_metadata["recovery"]["n_test_sims"] = config["recovery"]["n_test_sims"]
    # updated_metadata["recovery"]["n_posterior_samples"] = config["recovery"]["n_posterior_samples"]
    # updated_metadata["last_recovery_at"] = datetime.now().isoformat()

    # write_metadata(updated_metadata, ctx.metadata_path)
    return result
