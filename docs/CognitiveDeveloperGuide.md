# Cognitive developer guide

This guide describes the current `train`, `resume`, and `evaluate` commands and
the extension contracts in this checkout. Start with the commands below; the
older example README and Fish scripts describe a different model registration.
See [cleanup before pushing](CleaningBeforeGitPush.md) for repository preparation
and [future development](CognitiveDeveloperRoadmap.md) for planned improvements.

## 1. What the package does, and what you control

The package discovers your model, dispatches training/resume, resolves saved
artifacts, and runs the evaluation engine. Your workspace defines the science.
You normally add or change workspace files without editing package internals.

| Area | Package constraint | Your freedom |
| --- | --- | --- |
| Scientific model | The simulator must supply data the chosen adapter/workflow accepts. | Priors, equations, parameterization, observation coding, covariates, trial design. |
| Model registration | Supply a `ModelSpec` with callable, zero-argument factories and a registered workflow name. Model names must be unique across loaded plugins. | Registration filenames, model names, family/version labels, internal module layout. |
| Adapter | Must produce the inputs required by the chosen BayesFlow approximator and support inference without true parameters. | Transformations, parameter constraints, feature engineering, summary inputs. |
| Built-in training | Uses `SetTransformer(summary_dim=10)` and `CouplingFlow`; the CLI does not expose network/optimizer configuration. | Training budget, batch size, device, simulator, adapter. Register a custom workflow for different networks or training policy. |
| Evaluation scenario | Register an `EvaluationSpec`; `generate_batch(options)` returns `EvaluationBatch`. | Dataset generation, experimental conditions, stress tests, scenario options and metadata. |
| Evaluation result | The current engine requires named scalar parameters, truths, and posterior draws with explicit dataset/draw dimensions. | A custom evaluator can calculate different metrics from those normalized draws. |
| Persistence | The built-in lifecycle uses `.keras` checkpoints plus `.meta.json`, artifact IDs, and standard output paths. | Output root directories; a custom workflow may implement its own training policy while preserving usable artifacts. |

Changing a prior or observation encoding can change the inference problem even
if array shapes remain equal. Give materially different scientific models new
versions or names and train appropriate artifacts. Current built-in evaluation
checks model names, but does not fully verify scientific version/schema compatibility.

## 2. Prepare the environment and workspace

Use Python 3.12 as the current verification baseline. From the repository root,
install the locked CPU environment and development tools when setting up a new
checkout:

```bash
uv sync --locked --extra cpu --extra dev
export KERAS_BACKEND=torch
```

The project also declares `cu118` and `cu128` extras. Select one accelerator
extra; use an environment compatible with your hardware. The examples below
use CPU. A device option does not install an accelerator runtime.

`uv run --no-sync` uses the environment already installed; it does not install
or update dependencies. Use it after environment setup. Reuse the repository
environment when running the bundled workspace.

The relevant layout is:

```text
cli.py                                   CLI entry point
bayesflow_models/                        Package execution and contracts
examples/article_workspace/plugins/      Article source/reference implementations
examples/cognitive_developer_workspace/
  models_registeration.py                MODEL_SPECS
  evaluations_registerations.py          EVALUATIONS
  workflow_registrations.py              Optional WORKFLOWS
  cognitive_models_codes/                Your model, adapter, simulator, scenarios
  configs/                               Scenario settings
  tests/                                 Scientific and interface checks
  scripts/                               Example helpers (see caveat below)
```

Registration filenames are arbitrary despite the existing spelling. Discovery
scans only the top-level `*.py` files of each supplied plugin directory. It
recognizes `MODEL_SPECS`, `WORKFLOWS`, `EVALUATIONS`, or their corresponding
`register_models()`, `register_workflows()`, `register_evaluations()` functions.
Keep scientific implementation modules in an importable subpackage. Plugins
are executable Python code and their dependencies must already be installed.

### Current example wiring

| Model | Parameters | Workflow | Registered evaluations |
| --- | --- | --- | --- |
| `model_DC1` | `theta`, `b0`, `mu_ndt`, `mu_alpha` | `builtin` | `homogeneous-tta`, `mixed-tta` |
| `model_DC` | Seven parameters defined in `science.py` | `dc_example` | None currently registered for this name |

`model_DC1` uses `myscience.py`, `mysimulator.py`, `myadapter.py`, and the public
generators in `myscenarios.py`. The other model uses `science.py`, `simulators.py`,
and `adapters.py`. They are separate scientific definitions.

The current Fish training script selects `model_DC`, while the evaluation
registrations select `model_DC1`. Therefore chaining those scripts does not
currently provide a matching train/evaluate sequence. Use the commands below
for `model_DC1`. To evaluate `model_DC`, register compatible scenarios for that
model, including its seven true parameters; changing only the registration name
is insufficient.

## 3. Set explicit paths and inspect registrations

The following Bash function keeps commands readable. Run this block from the
repository root; the function remains defined in the current shell.

```bash
repo_root="$PWD"
workspace_root="$repo_root/examples/cognitive_developer_workspace"

bfcli() {
  uv run --no-sync bayesflow-cli \
    --plugin-path "$workspace_root" \
    --device cpu \
    --checkpoint-dir "$workspace_root/checkpoints" \
    --results-dir "$workspace_root/results" \
    --logs-dir "$workspace_root/logs" \
    "$@"
}

bfcli models list
bfcli evaluations list
bfcli checkpoints list
```

Global arguments such as `--plugin-path`, `--device`, and output roots go
**before** the subcommand. Command arguments go after it. Multiple plugin roots
can be supplied by repeating `--plugin-path`; duplicate registrations are rejected.
Without explicit roots, discovery and output directories are based on the current
working directory. `--plugin-path` alone does not relocate outputs.

An alias such as `run-002` is local to a checkpoint directory. The root repository
and example workspace can each have a different `run-002`. Use one checkpoint
root consistently, and record the immutable artifact ID in experiment records.

## 4. Train a new artifact

```bash
bfcli train --models model_DC1 \
  --epochs 1 --batches-per-epoch 2 --batch-size 4
bfcli checkpoints list
```

These small settings check execution, not posterior accuracy. Replace them with
a justified training budget for your experiment.

| Argument | Meaning | Current default |
| --- | --- | --- |
| `--models` | One or more registered model names; required. | None |
| `--epochs` | Number of training epochs. | `10` |
| `--batches-per-epoch` | Online simulation/training batches per epoch. | `100` |
| `--n-sim` | Alias for `--batches-per-epoch`, despite its historical name. | Same setting |
| `--batch-size` | Simulated datasets in each training batch. | `32` |

The nominal dataset budget is `epochs × batches_per_epoch × batch_size`; this
counts datasets, not trials within each dataset, and excludes any initialization
or validation simulations. Trial counts and schedules come from your simulator.

Training calls `workflow.train_fn(spec, config)`. The built-in workflow calls
both factories, trains online, saves an artifact, and writes metadata. A new
artifact gets a generated ID and the next available display alias. Its files are:

```text
checkpoints/<artifact_id>.keras
checkpoints/<artifact_id>.meta.json
results/<artifact_id>/
logs/<artifact_id>/
```

Directories are created, but a log directory is not a promise of captured console
logs. Capture stdout/stderr explicitly when you need a transcript. Inspect the
checkpoint table after training rather than assuming the next alias.

## 5. Resume an artifact

Select an actual artifact from `bfcli checkpoints list`. Replace the example
alias in this command:

```bash
bfcli resume --artifacts run-001 --mode new \
  --epochs 1 --batches-per-epoch 2 --batch-size 4
bfcli checkpoints list
```

| Argument | Meaning | Current default |
| --- | --- | --- |
| `--artifacts` | One or more artifact IDs or aliases; required. | None |
| `--epochs` | Additional epochs for this resume invocation. | `3` |
| `--batches-per-epoch` / `--n-sim` | Batches per additional epoch. | `100` |
| `--batch-size` | Datasets per new training batch. | `32` |
| `--mode new` | Save continuation as a new artifact with a parent ID. | Explicitly select this to preserve the source. |
| `--mode old` | Update the selected artifact in place. | **CLI default** |

Resume uses the current registered model and dispatches
`workflow.resume_fn(spec, config)`. Its configuration includes the resolved
artifact metadata and selected mode. Keep the scientific simulator, parameter
order, adapter schema, and checkpoint compatible. Editing an adapter factory does
not rewrite the adapter serialized inside an existing checkpoint.

### Resume success must include learning

The built-in trainer restores the saved optimizer. An exhausted saved
learning-rate schedule can prevent weight updates even when optimizer iterations
increase and the command exits successfully. This remains relevant to
`model_DC1`, which currently uses `builtin`.

The separate `dc_example` workflow used by `model_DC` has an explicit policy:
preserve optimizer moments/iterations and restart a constant learning rate,
default `0.0005`. `MODEL_DC_RESUME_LEARNING_RATE` controls that workspace policy;
it does **not** configure built-in resume for `model_DC1`.

For research runs, compare trainable weights and optimizer iterations before and
after continuation and inspect the effective learning rate. If you need another
resume policy, register a workspace workflow with a distinct name and compatible
callbacks. Record whether you restored, restarted, or reset optimizer state.

## 6. Evaluate homogeneous and mixed designs

Evaluation always selects scenarios by the pair `(model_name, scenario_name)`.
Use an artifact whose metadata names `model_DC1` for these current registrations.
Replace `run-002` with your actual selected artifact:

```bash
bfcli evaluate --artifacts run-002 --scenario homogeneous-tta \
  --n-posterior-samples 16 --mode save_only

bfcli evaluate --artifacts run-002 --scenario mixed-tta \
  --n-posterior-samples 16 --mode save_only
```

| Argument | Meaning | Current default |
| --- | --- | --- |
| `--artifacts` | One or more artifact IDs/aliases; required. | None |
| `--scenario` | Registered scenario name for the artifact's model; required. | None |
| `--options-file` | YAML mapping passed to your generator. | Empty mapping |
| `--n-posterior-samples` | Positive number of draws per dataset. | `1000` |
| `--mode save_only` | Save arrays, report, and recovery PDF. | Default |
| `--mode visualize` | Also show the recovery figure using Matplotlib. | Requires a display-capable backend. |

For a server or headless session, set `MPLBACKEND=Agg` and use `save_only`.
The CLI draw-count flag overrides a same-named YAML value. Sixteen draws are
useful for checking execution; they do not establish reliable interval coverage.

### Options actually implemented by the active generators

Create your own small YAML mapping using the supported entries below and pass
it with `--options-file path/to/options.yaml`.

| Setting | Homogeneous (`myscenarios.py`) | Mixed (`myscenarios.py`) |
| --- | --- | --- |
| `n_datasets` | Supported; default `8`. | Supported; default `8`. |
| `n_trials` | Supported; default `60`. | Not read; trial count is `4 × trials_per_tta`. |
| `tta_condition` | Forwarded to the simulator when supplied. | Not read. |
| `trials_per_tta` | Not read. | Supported; default `15`. |
| `shuffle` | Not read. | Supported; default `true`. |
| `seed` | Not read by the active homogeneous wrapper. | Seeds the local trial simulation/schedule RNG; default `2026`. |
| `tta_levels`, `posterior_seed`, `dt` | Not read. | Not read. |

Mixed generation currently uses the four levels in `TTA_LEVELS`. Its prior uses
the module-global `myscience.RNG`, so `seed` alone does not reset parameter draws.
Homogeneous generation uses both module/global RNG state and its training
simulator; its meta-function chooses one nominal TTA for the sampled batch.
The current options do not provide fully isolated or reproducible random streams.

The bundled recovery YAML files contain extra settings inherited from the other
scenario implementation (`scenarios.py`). The active `myscenarios.py` wrappers
silently ignore some of them. In particular, recording a seed in top-level report
options does not prove it was applied. Add explicit validation/seeding in your
generator when your study requires it.

## 7. Implement a scenario: required data contract

```python
from bayesflow_models.evaluation import EvaluationBatch, EvaluationSpec

def generate_batch(options):
    # Your simulation produces raw_conditions and named_truths.
    return EvaluationBatch(
        posterior_conditions=raw_conditions,
        true_parameters=named_truths,
        metadata={"design": "my_experiment"},
    )

EVALUATIONS = [
    EvaluationSpec(
        name="my-experiment",
        model_name="my_model",
        generate_batch=generate_batch,
        description="State which experiment this evaluates.",
    )
]
```

The current callback receives **only `options`**, not `(model_spec, options)`.
Import your scientific definitions inside the workspace as needed. The engine
calls the generator once, then the loaded approximator samples once.

For the current scalar recovery engine, use the following conventions. `N` is
datasets/participants, `T` is trials, `S` is draws, and `D` is observation features.

| Value | Required/recommended representation |
| --- | --- |
| `posterior_conditions` | Nonempty dictionary of NumPy arrays; every array has leading dimension `N`. Raw keys/shapes must match the saved adapter. |
| `true_parameters` | Exactly the keys in `ModelSpec.par_names`; one scalar per dataset per name. Use `(N,)` or `(N, 1)`. |
| `metadata` | JSON-serializable dictionary of design facts, applied seeds, and resolved settings. |
| Raw trial inputs | Commonly `(N, T, D)`; for `model_DC1`, `x` and `tta_per_trial` are each `(N, T, 1)`. |
| `model_DC1` trial count | `(N, 1, 1)`, which its saved adapter broadcasts to `(N, T, 1)`. |
| Sampler output | Dictionary per scalar parameter, each `(N, S)` or `(N, S, 1)`; normalized to `(N, S)`. |

Keep truths separate from inference conditions. The saved adapter must handle
conditions alone with `strict=False`. Test adaptation after training has
initialized parameter split indices. Do not pre-adapt data and then pass it
through the same adapter again during sampling.

The `(N, 1, 1)` trial count is a requirement of **this example's adapter**, not a
universal shape imposed on every model. The package validates the common batch
axis; adapter-specific axes remain your responsibility. Test batch sizes greater
than one: singleton batches can hide broadcasting errors.

A mixed design should simulate one shared participant parameter vector across
that participant's conditions when that is the scientific assumption. Changing
the within-participant design is a generalization test; matching the training
trial count does not by itself establish that design's calibration.

### Custom metrics and the current limit

`EvaluationSpec.evaluator` may supply an object with `name` and
`compute(*, posterior_samples, batch, parameter_names)`, returning
`EvaluationResult(result_payload=..., posterior_summary=...)`. It receives
normalized draws, not an approximator. The default is `RecoveryEvaluator`.

Custom metrics are supported within this pipeline, but normalization still
expects scalar parameters and the engine still saves a recovery plot. Vector or
hierarchical targets, posterior predictive evaluation, and evaluations without
truths require additional engine work; an evaluator alone does not remove those
constraints.

## 8. Read and preserve outputs

Each evaluation gets a unique directory:

```text
results/<artifact_id>/evaluations/<scenario>/<evaluation_id>/
  report.json
  evaluation_arrays.npz
  conditions.npz
  recovery.pdf
```

The report contains identity, draw/dataset counts, supplied options, scenario
metadata, evaluator results, and output paths. Arrays include `posterior_<name>`,
`true_<name>`, and saved summaries; conditions contain the exact raw inference
inputs. Default metrics include mean-based bias/RMSE, Pearson correlation,
posterior spread, and central 95% interval coverage. Undefined correlation is
JSON `null`. Inspect saved summaries when comparing the recovery plot to metrics;
the numerical errors use posterior means.

Numerical files are saved before plotting. A plotting exception records
`plotting.status = "failed"` and raises an error; inspect that invocation's report
and arrays before repeating sampling.

Preserve a checkpoint and its matching metadata together for later resume or
evaluation. Reports alone cannot reconstruct the trained network. Keep research
artifacts outside Git or in a separate artifact store.

## 9. Extend the package without mixing in article code

For a new model, implement scientific modules, register a `ModelSpec`, check
training and inference shapes, register compatible scenarios, and run a tiny
save/load/evaluate cycle before scaling up. Add a custom `Workflow` only when
training or resume policy needs it. Actual CLI callbacks currently receive
`(spec, config)`; the resume/recovery type aliases in `interfaces.py` still need
alignment with this runtime interface.

Custom workflows have broad control, but generic evaluation still needs the
standard metadata/checkpoint layout and an approximator loadable through
`keras.saving.load_model`. Custom serialized objects must be registered/importable
in a fresh process. Prefer built-in serializable transforms when sufficient.

Use `evaluate` for both current recovery scenarios. The older `mixed-recovery`
route imports an article-specific helper with a removed package import; it is
not the supported path for these examples. `evaluation/fixed_engine.py` is the
active engine; `evaluation/runner.py` is an obsolete alternate implementation.

For verification, run from the repository root:

```bash
KERAS_BACKEND=torch MPLBACKEND=Agg uv run --no-sync python -m pytest \
  tests examples/cognitive_developer_workspace/tests -q
```

The baseline on 2026-11-06 was **121 passed, 2 failed**. The failures concern the
old two-argument generator test and legacy `mixed-recovery` import. They are
tracked in the roadmap. The existing full example smoke script also needs its
model/scenario wiring reconciled before being used as release acceptance.
