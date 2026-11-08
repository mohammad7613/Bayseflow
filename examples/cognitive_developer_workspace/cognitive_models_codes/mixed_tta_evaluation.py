import json
from datetime import datetime
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np

from bayesflow_models import workflow as wf
# I have to add condition to the models 
from bayesflow_models.DDM_DC_Pedestrain_TrialWise import (
    CONDITIONS,
)
from bayesflow_models.interfaces import ModelSpec
from bayesflow.diagnostics.plots import recovery

PARAMETER_NAMES = ["theta", "b0", "mu_ndt", "mu_alpah"]


def _sample_simplest_params(rng: np.random.Generator) -> dict:
    """Sample the same priors used by prior_DC_simplest_model with a local RNG."""
    return {
        "theta": rng.uniform(0.1, 3.0),
        "b0": rng.uniform(2, 5),
        "mu_ndt": rng.uniform(0.2, 0.6),
        "mu_alpah": rng.uniform(0.1, 1),
    }


def _simulate_mixed_tta_dataset(
    params: dict,
    tta_values: np.ndarray,
    rng: np.random.Generator,
    dt: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one participant/session with different TTA values per trial."""
    theta = params["theta"]
    b0 = params["b0"]
    mu_ndt = params["mu_ndt"]
    mu_alpah = params["mu_alpah"]

    # Match the current simplest trialwise simulator constants.
    sigma_ndt = 1
    sigma_alpha = 1
    k = 1

    x_all = []
    tta_all = []

    for tta in tta_values:
        jitter = rng.uniform(0.0, 0.1)
        tta0 = float(tta + jitter)
        tta_all.append([tta0])

        evidence = 0.0
        t = 0.0
        alpha_trial = mu_alpah + sigma_alpha * rng.normal()

        boundary = b0 * (1 / (1 + np.exp(-k * (tta0 - t - 0.5 * b0))))
        while np.abs(evidence) < boundary:
            evidence += alpha_trial * (tta0 - t - theta) * dt + np.sqrt(dt) * rng.normal()
            t += dt
            boundary = b0 * (1 / (1 + np.exp(-k * (tta0 - t - 0.5 * b0))))

        nt = rng.normal(mu_ndt, sigma_ndt)
        while nt < 0:
            nt = rng.normal(mu_ndt, sigma_ndt)

        if evidence < 0:
            choicert = -1.0
        else:
            choicert = t + nt

        x_all.append([choicert])

    return np.asarray(x_all, dtype=np.float32), np.asarray(tta_all, dtype=np.float32)


def generate_mixed_tta_validation_data(
    n_test_sims: int,
    trials_per_tta: int,
    shuffle_tta: bool = True,
    seed: int = 2026,
) -> tuple[dict, dict]:
    """Generate batched mixed-TTA validation data and true parameters."""
    rng = np.random.default_rng(seed)
    base_ttas = np.repeat(CONDITIONS.astype(np.float32), trials_per_tta)
    n_trials = int(base_ttas.size)

    x_batch = []
    tta_batch = []
    params_batch = {name: [] for name in PARAMETER_NAMES}

    for _ in range(n_test_sims):
        tta_values = base_ttas.copy()
        if shuffle_tta:
            rng.shuffle(tta_values)

        params = _sample_simplest_params(rng)
        x, tta_per_trial = _simulate_mixed_tta_dataset(params, tta_values, rng)

        x_batch.append(x)
        tta_batch.append(tta_per_trial)
        for name in PARAMETER_NAMES:
            params_batch[name].append(params[name])

    conditions = {
        "x": np.asarray(x_batch, dtype=np.float32),
        "tta_per_trial": np.asarray(tta_batch, dtype=np.float32),
        "number_of_trials": np.full((n_test_sims,), n_trials, dtype=np.float32),
    }

    true_params = {
        name: np.expand_dims(np.asarray(values, dtype=np.float32),axis=-1)
        for name, values in params_batch.items()
    }

    return conditions, true_params


def _posterior_array_for_param(posterior, param_name: str, param_index: int) -> np.ndarray:
    if isinstance(posterior, dict):
        values = np.asarray(posterior[param_name])
    else:
        values = np.asarray(posterior)[..., param_index]

    return np.squeeze(values)


def _posterior_mean_and_std(values: np.ndarray, n_test_sims: int) -> tuple[np.ndarray, np.ndarray]:
    """Infer the posterior sample axis robustly for common BayesFlow outputs."""
    if values.ndim == 1:
        if values.shape[0] != n_test_sims:
            raise ValueError(f"Cannot infer posterior shape for values with shape {values.shape}")
        return values, np.zeros_like(values)

    if values.shape[0] == n_test_sims:
        sample_axis = 1
    elif values.shape[1] == n_test_sims:
        sample_axis = 0
    else:
        sample_axis = 0

    return values.mean(axis=sample_axis), values.std(axis=sample_axis)


def compute_mixed_tta_metrics(posterior, true_params: dict) -> tuple[dict, dict]:
    n_test_sims = len(next(iter(true_params.values())))
    metrics = {}
    posterior_summary = {}

    for index, name in enumerate(PARAMETER_NAMES):
        values = _posterior_array_for_param(posterior, name, index)
        posterior_mean, posterior_std = _posterior_mean_and_std(values, n_test_sims)
        target = true_params[name]
        error = posterior_mean - target

        if n_test_sims > 1 and np.std(posterior_mean) > 0 and np.std(target) > 0:
            pearson = float(np.corrcoef(target, posterior_mean)[0, 1])
        else:
            pearson = float("nan")

        metrics[name] = {
            "bias": float(np.mean(error)),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "pearson": pearson,
            "posterior_std_mean": float(np.mean(posterior_std)),
        }
        # posterior_summary[name] = {
        #     "mean": posterior_mean.astype(np.float32),
        #     "std": posterior_std.astype(np.float32),
        # }

        posterior_summary[name] = _posterior_summary(values, n_test_sims)
        coverage = np.mean((posterior_summary[name]["lower"] <= target) & (target <= posterior_summary[name]["upper"]))
        metrics[name]["coverage_95"] = float(coverage)

    return metrics, posterior_summary


def _posterior_summary(values: np.ndarray, n_test_sims: int) -> dict:
      if values.ndim == 1:
          if values.shape[0] != n_test_sims:
              raise ValueError(f"Cannot infer posterior shape for values with shape {values.shape}")
          return {
              "mean": values,
              "std": np.zeros_like(values),
              "lower": values,
              "upper": values,
          }

      if values.shape[0] == n_test_sims:
          sample_axis = 1
      elif values.shape[1] == n_test_sims:
          sample_axis = 0
      else:
          sample_axis = 0

      return {
          "mean": values.mean(axis=sample_axis),
          "std": values.std(axis=sample_axis),
          "lower": np.percentile(values, 2.5, axis=sample_axis),
          "upper": np.percentile(values, 97.5, axis=sample_axis),
      }


def plot_mixed_tta_recovery(
    true_params: dict,
    posterior_summary: dict,
    metrics: dict,
    save_path: Path,
) -> None:
    n_params = len(PARAMETER_NAMES)
    n_cols = 2
    n_rows = int(np.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows))
    axes = np.asarray(axes).reshape(-1)

    for ax, name in zip(axes, PARAMETER_NAMES):
        target = true_params[name]
        estimate = posterior_summary[name]["mean"]
        # ax.scatter(target, estimate, s=24, alpha=0.7)
        lower = posterior_summary[name]["lower"]
        upper = posterior_summary[name]["upper"]
        yerr = np.vstack([estimate - lower, upper - estimate])
        ax.errorbar(target,estimate,yerr=yerr,fmt="o",markersize=4,alpha=0.7,ecolor="0.55",elinewidth=1,capsize=2,)

        low = float(min(np.min(target), np.min(estimate)))
        high = float(max(np.max(target), np.max(estimate)))
        ax.plot([low, high], [low, high], "r--", linewidth=1.5)
        ax.set_title(f"{name} | r={metrics[name]['pearson']:.2f}, RMSE={metrics[name]['rmse']:.3f}")
        ax.set_title(f"{name} | r={metrics[name]['pearson']:.2f}, RMSE={metrics[name]['rmse']:.3f}, Cov={metrics[name]['coverage_95']:.2f}")
        ax.set_xlabel("true")
        # ax.set_ylabel("posterior mean")
        ax.set_ylabel("posterior mean with 95% CI")

    for ax in axes[n_params:]:
        ax.axis("off")

    fig.suptitle("Mixed-TTA recovery stress test\ntrained on homogeneous-TTA simulations")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _posterior_npz_payload(posterior) -> dict:
    if isinstance(posterior, dict):
        return {f"posterior_{name}": np.asarray(values) for name, values in posterior.items()}
    return {"posterior_samples": np.asarray(posterior)}


def evaluate_mixed_tta_artifact_with_bf_recovery(
    artifact_ref: str,
    spec: ModelSpec,
    config: dict,
    n_test_sims: int = 100,
    n_posterior_samples: int = 1000,
    trials_per_tta: int = 15,
    shuffle_tta: bool = True,
    seed: int = 2026,
) -> dict:
    """Load a trained artifact and test posterior recovery on mixed-TTA simulations."""
    metadata = wf.resolve_artifact_ref(artifact_ref, config)
    par_names = spec.par_names
    if metadata["spec_name"] != spec.name:
        raise ValueError(
            f"Artifact {metadata['artifact_id']} belongs to {metadata['spec_name']}, not {spec.name}"
        )

    ctx = wf.resolve_context_from_artifact_id(metadata["artifact_id"], config)
    if not ctx.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ctx.checkpoint_path}")

    approximator = keras.saving.load_model(ctx.checkpoint_path)
    val_sims, true_params = generate_mixed_tta_validation_data(
        n_test_sims=n_test_sims,
        trials_per_tta=trials_per_tta,
        shuffle_tta=shuffle_tta,
        seed=seed,
    )
    val_sims.update(true_params)
    posterior = approximator.sample(
        conditions=val_sims,
        num_samples=n_posterior_samples,
    )
    

    f = recovery(
        estimates=posterior,
        targets=val_sims,
        variable_names=par_names
    )
    output_dir = ctx.results_dir / "mixed_tta_recovery"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "mixed_tta_recovery.pdf"
    # BayesFlow recovery(...) returns a matplotlib figure-like object
    f.savefig(pdf_path, bbox_inches="tight")
    if config["mode"] == "visualize":
       plt.show()
    else:   
      plt.close(f)

    report = {
        "artifact_id": metadata["artifact_id"],
        "alias": metadata.get("alias"),
        "spec_name": spec.name,
        "checkpoint_path": str(ctx.checkpoint_path),
        "created_at": datetime.now().isoformat(),
        "interpretation": "out_of_distribution_stress_test",
        "n_test_sims": n_test_sims,
        "n_posterior_samples": n_posterior_samples,
        "trials_per_tta": trials_per_tta,
        "n_trials": int(val_sims["x"].shape[1]),
        "shuffle_tta": shuffle_tta,
        "seed": seed,
    }
    json_path = output_dir / "mixed_tta_recovery.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
    
def evaluate_mixed_tta_artifact(
    artifact_ref: str,
    spec: ModelSpec,
    config: dict,
    n_test_sims: int = 100,
    n_posterior_samples: int = 1000,
    trials_per_tta: int = 15,
    shuffle_tta: bool = True,
    seed: int = 2026,
) -> dict:
    """Load a trained artifact and test posterior recovery on mixed-TTA simulations."""
    metadata = wf.resolve_artifact_ref(artifact_ref, config)
    if metadata["spec_name"] != spec.name:
        raise ValueError(
            f"Artifact {metadata['artifact_id']} belongs to {metadata['spec_name']}, not {spec.name}"
        )

    ctx = wf.resolve_context_from_artifact_id(metadata["artifact_id"], config)
    if not ctx.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ctx.checkpoint_path}")

    approximator = keras.saving.load_model(ctx.checkpoint_path)
    conditions, true_params = generate_mixed_tta_validation_data(
        n_test_sims=n_test_sims,
        trials_per_tta=trials_per_tta,
        shuffle_tta=shuffle_tta,
        seed=seed,
    )
    posterior = approximator.sample(
        conditions=conditions,
        num_samples=n_posterior_samples,
    )

    metrics, posterior_summary = compute_mixed_tta_metrics(posterior, true_params)

    output_dir = ctx.results_dir / "mixed_tta_recovery"
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "mixed_tta_recovery.npz"
    json_path = output_dir / "mixed_tta_recovery.json"
    png_path = output_dir / "mixed_tta_recovery.png"

    np.savez(
        npz_path,
        x=conditions["x"],
        tta_per_trial=conditions["tta_per_trial"],
        number_of_trials=conditions["number_of_trials"],
        **{f"true_{name}": values for name, values in true_params.items()},
        **{
            f"posterior_mean_{name}": summary["mean"]
            for name, summary in posterior_summary.items()
        },
        **{
            f"posterior_std_{name}": summary["std"]
            for name, summary in posterior_summary.items()
        },
        **_posterior_npz_payload(posterior),
    )

    plot_mixed_tta_recovery(true_params, posterior_summary, metrics, png_path)

    report = {
        "artifact_id": metadata["artifact_id"],
        "alias": metadata.get("alias"),
        "spec_name": spec.name,
        "checkpoint_path": str(ctx.checkpoint_path),
        "created_at": datetime.now().isoformat(),
        "interpretation": "out_of_distribution_stress_test",
        "n_test_sims": n_test_sims,
        "n_posterior_samples": n_posterior_samples,
        "trials_per_tta": trials_per_tta,
        "n_trials": int(conditions["x"].shape[1]),
        "shuffle_tta": shuffle_tta,
        "seed": seed,
        "metrics": metrics,
        "outputs": {
            "npz": str(npz_path),
            "json": str(json_path),
            "plot": str(png_path),
        },
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
