"""Exercise real CLI training, fresh-process resume, and both recovery scenarios.

Run with: uv run --no-sync python examples/cognitive_developer_workspace/scripts/smoke_test.py
This is a software execution check; tiny budgets do not establish scientific
recovery accuracy. No package function is patched or replaced in this check.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from uuid import uuid4

WORKSPACE = Path(__file__).resolve().parents[1]


def main():
    log_dir = WORKSPACE / "logs" / f"smoke-{uuid4().hex[:12]}"
    log_dir.mkdir(parents=True)
    env = os.environ.copy()
    env.update(
        {
            "KERAS_BACKEND": "torch",
            "MPLBACKEND": "Agg",
            "CUDA_VISIBLE_DEVICES": "-1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "2023",
            "MPLCONFIGDIR": str(WORKSPACE / ".cache/matplotlib"),
            "KERAS_HOME": str(WORKSPACE / ".cache/keras"),
            "MODEL_DC_SIMULATION_SEED": "2023",
            "MODEL_DC_NETWORK_SEED": "2023",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        }
    )
    uv = [
        "uv",
        "--directory",
        str(WORKSPACE),
        "--offline",
        "--no-cache",
        "run",
        "--no-sync",
    ]
    cli = uv + [
        "bayesflow-cli",
        "--plugin-path",
        str(WORKSPACE),
        "--device",
        "cpu",
        "--checkpoint-dir",
        str(WORKSPACE / "checkpoints"),
        "--results-dir",
        str(WORKSPACE / "results"),
        "--logs-dir",
        str(WORKSPACE / "logs"),
    ]
    verification = {"status": "running", "steps": [], "workspace": str(WORKSPACE)}

    def run(name, command):
        logfile = log_dir / f"{name}.log"
        print(f"Running {name}; log: {logfile}", flush=True)
        start = time.monotonic()
        with logfile.open("w") as stream:
            result = subprocess.run(
                command,
                cwd=WORKSPACE,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=300,
                check=False,
            )
        verification["steps"].append(
            {
                "name": name,
                "command": command,
                "exit_code": result.returncode,
                "seconds": round(time.monotonic() - start, 2),
                "log": str(logfile),
            }
        )
        if result.returncode:
            raise RuntimeError(
                f"{name} failed. Read {logfile}.\n{logfile.read_text()[-5000:]}"
            )

    def artifact_ids():
        return {
            p.stem.removesuffix(".meta")
            for p in (WORKSPACE / "checkpoints").glob("*.meta.json")
        }

    def new_artifact(before):
        created = artifact_ids() - before
        if len(created) != 1:
            raise AssertionError(f"Expected one new artifact; found {created}.")
        artifact_id = created.pop()
        metadata = json.loads(
            (WORKSPACE / "checkpoints" / f"{artifact_id}.meta.json").read_text()
        )
        return artifact_id, metadata

    def inspect(name, artifact_id):
        output = log_dir / f"{name}.json"
        run(
            name,
            uv
            + [
                "python",
                str(WORKSPACE / "scripts/inspect_checkpoint.py"),
                str(WORKSPACE / "checkpoints" / f"{artifact_id}.keras"),
                str(output),
            ],
        )
        return json.loads(output.read_text())

    try:
        run("models", cli + ["models", "list"])
        run("scenarios", cli + ["evaluations", "list"])
        before = artifact_ids()
        run(
            "train",
            cli
            + [
                "train",
                "--models",
                "model_DC",
                "--batches-per-epoch",
                "2",
                "--batch-size",
                "4",
                "--epochs",
                "1",
            ],
        )
        original_id, original_metadata = new_artifact(before)
        original_state = inspect("original_state", original_id)

        before = artifact_ids()
        env.update(MODEL_DC_SIMULATION_SEED="2024", MODEL_DC_NETWORK_SEED="2024")
        run(
            "resume",
            cli
            + [
                "resume",
                "--artifacts",
                original_id,
                "--mode",
                "new",
                "--batches-per-epoch",
                "2",
                "--batch-size",
                "4",
                "--epochs",
                "1",
            ],
        )
        resumed_id, resumed_metadata = new_artifact(before)
        resumed_state = inspect("resumed_state", resumed_id)
        assert resumed_metadata["parent_artifact_id"] == original_id
        assert (
            resumed_state["optimizer_iterations"]
            > original_state["optimizer_iterations"]
        )
        verification["original_artifact"] = original_metadata
        verification["resumed_artifact"] = resumed_metadata
        verification["original_state"] = original_state
        verification["resumed_state"] = resumed_state

        # Crucial: successful resume exit + a new file do not prove learning.
        if (
            resumed_state["trainable_weights_sha256"]
            == original_state["trainable_weights_sha256"]
        ):
            raise AssertionError(
                "Resume performed no trainable-weight updates. Inspect the saved optimizer learning rate."
            )

        # Evaluate the resumed checkpoint in separate processes, without truths
        # in sampler inputs. Both operations call the same package engine.
        reports = {}
        for scenario, filename in (
            ("homogeneous-tta", "homogeneous_recovery.yaml"),
            ("mixed-tta", "mixed_recovery.yaml"),
        ):
            run(
                scenario,
                cli
                + [
                    "evaluate",
                    "--artifacts",
                    resumed_id,
                    "--scenario",
                    scenario,
                    "--options-file",
                    str(WORKSPACE / "configs" / filename),
                    "--n-posterior-samples",
                    "32",
                ],
            )
            scenario_dir = WORKSPACE / "results" / resumed_id / "evaluations" / scenario
            candidates = list(scenario_dir.glob("*/report.json"))
            assert len(candidates) == 1
            report = json.loads(candidates[0].read_text())
            assert report["n_datasets"] == 8 and report["n_posterior_samples"] == 32
            assert set(report["results"]["metrics"]) == {
                "tau",
                "b0",
                "k",
                "mu_ndt",
                "sigma_ndt",
                "alpha",
                "theta",
            }
            for path in report["outputs"].values():
                assert Path(path).is_file(), path
            assert report["scenario_metadata"]["design"] == (
                "mixed_tta" if scenario == "mixed-tta" else "homogeneous_tta"
            )
            reports[scenario] = str(candidates[0])
        verification["evaluation_reports"] = reports
        verification["status"] = "passed"
    except Exception as exc:
        verification["status"] = "failed"
        verification["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        target = log_dir / "verification.json"
        target.write_text(json.dumps(verification, indent=2, allow_nan=False) + "\n")
        print(f"Verification: {target}", flush=True)
    print(
        "PASS: train, resume with weight updates, homogeneous recovery, and mixed recovery."
    )


if __name__ == "__main__":
    main()
