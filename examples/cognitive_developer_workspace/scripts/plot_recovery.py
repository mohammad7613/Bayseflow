"""Plot the generic engine's saved recovery arrays without resampling.

Example: fish scripts/homogeneous_recovery.fish run-001 (calls this afterward),
or: uv run --no-sync python scripts/plot_recovery.py --report PATH_TO_REPORT
The plot is a workspace convenience; the numerical report remains engine-owned.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--report", type=Path)
    source.add_argument("--artifact", help="Workspace artifact ID or alias.")
    parser.add_argument("--scenario", choices=["homogeneous-tta", "mixed-tta"])
    args = parser.parse_args()
    report_path = args.report
    if report_path is None:
        if args.scenario is None:
            parser.error("--scenario is required with --artifact.")
        matches = []
        for path in (WORKSPACE / "checkpoints").glob("*.meta.json"):
            metadata = json.loads(path.read_text())
            if args.artifact in (metadata["artifact_id"], metadata.get("alias")):
                matches.append(metadata["artifact_id"])
        if len(matches) != 1:
            parser.error("Artifact reference must identify exactly one workspace run.")
        candidates = list(
            (WORKSPACE / "results" / matches[0] / "evaluations" / args.scenario).glob(
                "*/report.json"
            )
        )
        if not candidates:
            parser.error("No saved report found. Run evaluate first.")
        report_path = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    report_path = report_path.resolve()
    if WORKSPACE not in report_path.parents:
        parser.error("Plots must be written inside this workspace.")
    report = json.loads(report_path.read_text())
    names = report["parameter_names"]
    fig, axes = plt.subplots(
        int(np.ceil(len(names) / 3)), 3, figsize=(12, 9), squeeze=False
    )
    with np.load(report["outputs"]["arrays"], allow_pickle=False) as saved:
        for ax, name in zip(axes.flat, names):
            truth = saved[f"true_{name}"].reshape(-1)
            mean = saved[f"posterior_mean_{name}"]
            lower, upper = (
                saved[f"posterior_lower_{name}"],
                saved[f"posterior_upper_{name}"],
            )
            # Draw intervals directly: a posterior mean need not lie within an
            # equal-tailed interval, so asymmetric errorbar lengths can be negative.
            ax.vlines(truth, lower, upper, color="0.55", alpha=0.7)
            ax.scatter(truth, mean, s=18)
            low = float(min(truth.min(), mean.min(), lower.min()))
            high = float(max(truth.max(), mean.max(), upper.max()))
            if low == high:
                low, high = low - 0.1, high + 0.1
            ax.plot([low, high], [low, high], "--", color="tab:red", linewidth=1)
            metrics = report["results"]["metrics"][name]
            ax.set_title(
                f"{name}: RMSE={metrics['rmse']:.3g}, coverage={metrics['coverage_95']:.2f}"
            )
            ax.set_xlabel("True parameter")
            ax.set_ylabel("Posterior mean and 95% interval")
        for ax in list(axes.flat)[len(names) :]:
            ax.set_visible(False)
    interpretation = report["scenario_metadata"]["interpretation"]
    fig.suptitle(
        f"{report['evaluation_name']} | {interpretation}\n{report['artifact_id']}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output = report_path.parent / "recovery.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
