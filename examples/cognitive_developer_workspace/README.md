# model_DC: a cognitive developer's workspace

This example supplies a scientific model and two recovery scenarios to the
existing BayesFlow CLI. All source, configurations, checkpoints, results, and
logs for the example live in this directory. The original article code and the
package are not modified. Reuse the parent project's existing uv environment.

The completed checks and actual artifact aliases are recorded in
[reference/verification.md](reference/verification.md).

## Read the code in this order

```text
cognitive_developer_workspace/
├── models_registeration.py           # MODEL_SPECS: model name and factories
├── evaluations_registerations.py     # EVALUATIONS: homogeneous/mixed recovery
├── workflow_registrations.py         # WORKFLOWS: documented resume policy
├── cognitive_models_codes/
│   ├── science.py                   # Seven priors and one shared trial kernel
│   ├── designs.py                   # Nominal homogeneous/mixed TTA schedules
│   ├── simulators.py                # BayesFlow online-training data wrapper
│   ├── adapters.py                  # Raw inputs -> parameters and trial features
│   ├── scenarios.py                 # EvaluationBatch generators, not evaluators
│   └── workflows.py                 # Standard training + explicit optimizer resume
├── configs/
│   ├── homogeneous_recovery.yaml
│   └── mixed_recovery.yaml
├── scripts/                         # Fish commands and real CLI verification
├── tests/                           # Scientific parity, shapes, and contracts
├── reference/model_contract.md       # Equations, coding, limits, source mapping
├── checkpoints/                     # Generated .keras and .meta.json pairs
├── results/                         # Generated numerical recovery and figures
└── logs/                            # Generated smoke logs and verification.json
```

The two registration filenames supplied with the empty workspace are retained,
including their spelling. Filenames are arbitrary: discovery recognizes the
`MODEL_SPECS`, `EVALUATIONS`, and `WORKFLOWS` declarations. Pass this directory
itself to `--plugin-path`; nested scientific modules are ordinary Python imports.

## Run from fish

From the repository root:

```fish
cd examples/cognitive_developer_workspace
fish scripts/list.fish
fish scripts/train.fish
```

Training prints a checkpoint table. Use the actual alias from that table:

```fish
# Example aliases only: replace them if you already have workspace runs.
fish scripts/resume.fish run-001
fish scripts/homogeneous_recovery.fish run-002
fish scripts/mixed_recovery.fish run-002
```

Resume defaults to `--mode new`, preserving the source artifact. Its new metadata
records the parent artifact. To update an artifact deliberately, append
`--mode old`. Each evaluation creates a unique invocation directory.

The scripts use `uv run --no-sync`, the existing parent environment, explicit
workspace paths, Torch, and CPU. They do not install dependencies. They work from
other working directories too when invoked by their full paths. CUDA is hidden
for these CPU examples; GPU execution has not been validated here.

Default training is deliberately tiny: one epoch, two batches, four datasets per
batch. This verifies execution and does not establish useful posterior accuracy.
For a larger experiment, override the arguments and record the chosen budget:

```fish
fish scripts/train.fish --epochs 20 --batches-per-epoch 100 --batch-size 32
fish scripts/homogeneous_recovery.fish run-003 --n-posterior-samples 1000
fish scripts/mixed_recovery.fish run-003 --n-posterior-samples 1000
```

Those larger numbers are an example, not a scientifically validated budget.
Edit the YAML files to change scenario dataset counts or TTA composition. The
CLI draw-count flag overrides any identically named YAML entry.

## The equivalent generic CLI

These are real commands exposed by the current package, not TTA-specific CLI
extensions. Run them from this workspace:

```fish
set -gx KERAS_BACKEND torch
set -gx MPLBACKEND Agg
set -gx CUDA_VISIBLE_DEVICES -1

uv run --no-sync bayesflow-cli --plugin-path . models list
uv run --no-sync bayesflow-cli --plugin-path . evaluations list
uv run --no-sync bayesflow-cli --plugin-path . --device cpu \
    --checkpoint-dir checkpoints --results-dir results \
    evaluate --artifacts run-002 --scenario homogeneous-tta \
    --options-file configs/homogeneous_recovery.yaml --n-posterior-samples 32

uv run --no-sync bayesflow-cli --plugin-path . --device cpu \
    --checkpoint-dir checkpoints --results-dir results \
    evaluate --artifacts run-002 --scenario mixed-tta \
    --options-file configs/mixed_recovery.yaml --n-posterior-samples 32
```

Prefer the supplied scripts: they also configure local caches, seeds, paths, and
recovery plots. Avoid the old package `mixed-recovery` command; both workspace
scenarios use `evaluate` and the same package engine.

## Why this example registers a workflow

The current built-in trainer saves a cosine learning-rate schedule that reaches
zero at the end of training. An actual fresh-process check found that built-in
resume advanced optimizer iterations but changed no trainable weights.

The `dc_example` workflow is an explicit, discoverable workspace extension.
Its training callback delegates to package training. Its resume callback loads
the saved adapter, networks, optimizer moments, and iteration counter, then
replaces the finished schedule with a constant learning rate (default 0.0005).
The policy and seeds are recorded in artifact metadata. This is continuation
with a restarted schedule, not bit-for-bit continuation of an interrupted
original schedule. Set `MODEL_DC_RESUME_LEARNING_RATE` to choose another rate.

General evaluation does not call this workflow; it continues to use the
unchanged package runner. No package monkeypatches or edits are involved.

## Verify the example

Fast tests, with generated test files kept inside the workspace:

```fish
source scripts/common.fish
model_dc_python -m pytest tests -q -p no:cacheprovider --basetemp=.cache/pytest
```

Full real CLI check (creates new artifacts and reports):

```fish
uv run --no-sync python scripts/smoke_test.py
```

The smoke check runs training, checkpoint inspection, resume into a new artifact,
and both evaluations in separate processes. It verifies that optimizer steps
increase AND trainable weights actually change. Read its printed
`logs/smoke-.../verification.json` and per-step logs for exact commands/results.
The original package-resume failure is retained as diagnostic evidence in its
own failed smoke log; subsequent runs use the explicit workspace workflow.

## Read the results

```text
results/<artifact_id>/evaluations/<scenario>/<evaluation_id>/
├── report.json                 # Engine metrics, options, identity, output paths
├── evaluation_arrays.npz        # Draws, truths, and per-dataset summaries
├── conditions.npz               # Actual raw inputs used for inference
└── recovery.pdf                 # Added by the workspace plotting script
```

The recovery scripts produce `recovery.pdf` from the saved arrays without
resampling. The figure is a workspace output; the engine report is left intact.
For a specific invocation, run `model_dc_python scripts/plot_recovery.py --report
PATH_TO_REPORT`. The figure uses the same posterior means and central 95%
intervals as the engine metrics.

Simulation seeds use local NumPy generators. `posterior_seed` explicitly seeds
Keras before checkpoint loading/sampling because the current engine has no
separate posterior-seed hook. Resolved scenario defaults appear in
`report.json -> scenario_metadata -> resolved_options`; top-level `options`
records values supplied to the engine.

Homogeneous recovery is the training-design baseline. Mixed recovery shares
parameters across all conditions but changes the within-participant design;
label it a generalization stress test. Passing the smoke test demonstrates
software execution, not scientific calibration or adequate training.
