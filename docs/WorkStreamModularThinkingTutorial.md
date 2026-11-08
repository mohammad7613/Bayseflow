# Workstream 2 Modular-Thinking Tutorial

## Purpose

This tutorial reviews the reasoning started in
`docs/WorkStream2ModularThinking.md` and turns it into an implementation plan.
The objective is not merely to split a long function into several Python files.
The objective is to learn how to derive modules from responsibilities,
ownership, contracts, variation points, and failure boundaries.

The concrete product goal is a reusable evaluation pipeline for trained
posterior networks. Mixed-TTA recovery must become one researcher-defined
scenario that uses this pipeline, rather than scientific TTA assumptions being
embedded in the CLI engine.

This document does not ask you to create additional modules beyond the
Workstream 2 design. It explains how to think from the abstract system down to
the functions you will implement in the existing planned files.

---

## 1. Review of the Original Modular Reasoning

The draft correctly identified this pipeline:

```text
generate validation data
        ↓
sample a posterior
        ↓
calculate evaluation metrics
        ↓
report the result
```

This is a good first decomposition because each stage transforms a different
kind of information.

The draft also correctly noticed three important kinds of variation:

- evaluation data generation changes with the scientific scenario;
- metric calculation changes with the kind of evaluation;
- reporting may produce plots, arrays, JSON, or interactive output.

However, several design questions were still unspecified.

### 1.1 A stage is not automatically a Python module

A pipeline stage describes runtime behavior. A Python module describes a code
ownership and dependency boundary. They often correspond, but not always.

For example, posterior sampling is only one operation, but it belongs inside a
`runner.py` module because something must coordinate artifact resolution,
validation, checkpoint loading, sampling, metrics, and reporting. The value of
the runner module is orchestration, not algorithmic variation.

### 1.2 The scientific ownership boundary was missing

The most important modular boundary is not metrics versus reporting. It is:

```text
engine-owned reusable mechanisms
              versus
researcher-owned scientific scenario
```

The engine can know what a posterior sample is, how to calculate recovery
metrics, and how to save a report. It must not know what TTA means, which TTA
values are scientifically valid, or how a particular DDM trial is simulated.

### 1.3 Data contracts were not yet explicit

The draft names “evaluation data,” “posterior samples,” and “metric data,” but
does not yet state their required shapes, keys, ownership, or serialization
rules. Without those contracts, modules remain coupled through assumptions.

### 1.4 Validation and failure ownership were missing

Before loading an expensive network, the system must decide:

- whether the scenario belongs to the selected model;
- whether target names match `ModelSpec.par_names`;
- whether all targets describe the same number of datasets;
- whether posterior conditions are non-empty;
- whether scenario metadata is JSON-serializable;
- whether the checkpoint exists.

These checks are part of the modular design because a contract is incomplete
until its failure behavior is defined.

### 1.5 Discovery and command dispatch were missing

A modular scenario is not useful merely because its function exists. The
product needs a way to register, discover, select, and invoke it. Therefore,
plugin discovery and CLI dispatch are part of the complete architecture even
though they are not numerical evaluation stages.

---

## 2. Begin With Questions, Not Files

When designing a modular feature, answer the following questions in order.

### Question 1: What behavior must remain stable?

For Workstream 2:

- resolve a trained artifact by ID or alias;
- verify that it belongs to the requested model;
- load its posterior network;
- generate scientifically valid evaluation datasets;
- obtain posterior samples;
- compare samples with known ground truth;
- produce recovery plots, numeric metrics, arrays, and metadata;
- preserve the scientific meaning of the existing mixed-TTA stress test.

### Question 2: What is expected to vary?

Expected variation includes:

- model family: DC, DF, SF, or a future model;
- scenario: mixed-TTA, ordinary recovery, prior predictive checks, or a future
  non-TTA scenario;
- scenario options: seed, dataset count, condition values, trials per
  condition, shuffle policy;
- number and names of inferred parameters;
- report presentation and optional diagnostics.

### Question 3: Who owns each decision?

| Decision | Owner | Reason |
|---|---|---|
| Valid TTA values | Researcher scenario | Scientific meaning |
| Parameter prior and simulator equations | Researcher model plugin | Model-specific knowledge |
| Construction of posterior conditions | Researcher scenario using the model contract | Adapter compatibility |
| Artifact and checkpoint resolution | Engine | Common runtime mechanism |
| Loading and sampling the approximator | Engine runner | Common execution mechanism |
| Recovery metric formulas | Engine metrics | Model-independent mathematics |
| Output directory convention | Engine reporting | Product-wide consistency |
| Scenario-specific report facts | Researcher scenario | The engine cannot infer their meaning |
| Scenario selection | CLI and plugin registry | Product capability dispatch |

### Question 4: What crosses each boundary?

Only stable data contracts should cross module boundaries:

```text
scenario → EvaluationBatch → runner
runner → posterior samples + targets → metrics
runner → execution facts + results → reporting
reporting → serializable report → CLI/user
```

Passing a simulator implementation, TTA constants, or arbitrary workspace
state into the generic reporting layer would weaken the boundary.

### Question 5: Where should failure be detected?

Detect a failure at the earliest layer that has enough knowledge to explain it:

- scenario rejects invalid scenario options;
- runner validates the `EvaluationBatch` contract;
- metrics validates posterior and target shapes;
- reporting validates serializability and output paths;
- discovery rejects malformed or duplicate registrations;
- CLI rejects unknown model/scenario choices and malformed `KEY=VALUE` input.

This produces errors that name the violated contract rather than failing later
inside Keras, NumPy, or JSON.

---

## 3. The Resulting Architecture

The architecture has two ownership regions.

```text
Researcher workspace
┌──────────────────────────────────────────────────────────────┐
│ model scientific kernels and parameter samplers              │
│ mixed-TTA scenario generator                                 │
│ EVALUATIONS registration                                     │
└───────────────────────────┬──────────────────────────────────┘
                            │ EvaluationBatch
                            ▼
Engine package
┌──────────────────────────────────────────────────────────────┐
│ interfaces.py  stable shared data contracts                  │
│ runner.py      workflow orchestration and validation         │
│ metrics.py     model-neutral recovery calculations           │
│ reporting.py   plots, arrays, JSON, output convention        │
│ discovery.py   capability registration                       │
│ cli.py         user input and capability selection           │
└──────────────────────────────────────────────────────────────┘
```

Dependency direction matters:

```text
workspace scenario ──imports──> public engine interfaces
engine runner ──imports──> interfaces, metrics, reporting
metrics ──imports──> NumPy only
reporting ──imports──> diagnostics, NumPy, plotting, interfaces
```

The engine must never import the article workspace or mixed-TTA scenario.
Discovery inverts that dependency: the workspace registers a capability and
the engine loads it through the public interface.

---

## 4. Contract Module: `interfaces.py`

### 4.1 Responsibility

`interfaces.py` defines the smallest stable vocabulary shared by the engine and
researcher plugins. It performs no artifact loading, simulation, metric
calculation, plotting, or filesystem writes.

### 4.2 `EvaluationBatch`

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationBatch:
    posterior_conditions: dict[str, Any]
    true_parameters: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
```

Field contracts:

#### `posterior_conditions`

- Exact input accepted by `approximator.sample(conditions=...)`.
- First dimension represents independent evaluation datasets.
- May contain trial dimensions internally.
- Keys are model/adapter-specific and therefore are not prescribed by the
  engine.

Example:

```python
{
    "x": np.ndarray(shape=(20, 60, 1)),
    "tta_per_trial": np.ndarray(shape=(20, 60, 1)),
    "number_of_trials": np.ndarray(shape=(20,)),
}
```

#### `true_parameters`

- Keys must exactly equal `ModelSpec.par_names`.
- Every value contains one ground-truth value per dataset.
- Every value must therefore have the same leading dimension.

Example:

```python
{
    "b0": np.ndarray(shape=(20,)),
    "alpha": np.ndarray(shape=(20,)),
}
```

#### `metadata`

- Contains scientific scenario facts, not engine execution facts.
- Must be JSON-serializable.
- Examples include condition values, trial count, seed, shuffle policy, and
  interpretation.
- It must not contain the posterior network or open file handles.

### 4.3 `EvaluationSpec`

```python
from collections.abc import Callable

from bayesflow_models.interfaces import ModelSpec


GenerateEvaluationBatch = Callable[
    [ModelSpec, dict[str, Any]],
    EvaluationBatch,
]


@dataclass(frozen=True)
class EvaluationSpec:
    name: str
    model_name: str
    generate_batch: GenerateEvaluationBatch
    description: str = ""
```

The callable receives:

- `ModelSpec`: the model selected by the artifact;
- `options`: scenario-owned configuration such as `n_test_sims`, seed, and
  mixed-TTA options.

It returns one valid `EvaluationBatch` and performs no posterior sampling or
report writing.

### 4.4 Why there is no metric protocol yet

The first product requirement is one recovery-metric implementation. A metric
protocol would create an abstraction without a second proven implementation.
Introduce it only when a genuinely different evaluation family, such as
simulation-based calibration or posterior predictive checking, must be
selected dynamically.

---

## 5. Scenario Generation: Researcher-Owned Code

### 5.1 Responsibility

The scenario answers:

> Which known parameters and observations constitute a scientifically valid
> evaluation dataset for this trained model?

It does not answer:

- where the checkpoint lives;
- how Keras loads the approximator;
- how RMSE is calculated;
- where the report is written.

### 5.2 Required scientific functions

The model workspace should first expose reusable scientific kernels:

```python
def sample_dc_parameters(
    rng: np.random.Generator,
) -> dict[str, float]:
    ...


def simulate_dc_trial(
    *,
    parameters: dict[str, float],
    tta: float,
    rng: np.random.Generator,
) -> float:
    ...
```

Training and evaluation must call the same functions. Otherwise, evaluation
may test a duplicated approximation of the model rather than the trained
model's actual scientific assumptions.

### 5.3 Mixed-TTA generator signature

```python
def generate_dc_mixed_tta(
    model_spec: ModelSpec,
    options: dict[str, Any],
) -> EvaluationBatch:
    ...
```

Inputs expected in `options`:

```text
n_test_sims: int
trials_per_tta: int
condition_values: list[float]  # optional default owned by scenario
shuffle_tta: bool
seed: int
```

The generator should:

1. validate and normalize its options;
2. create a local `np.random.Generator` from `seed`;
3. allocate conditions and target collections;
4. for each dataset, sample one parameter vector;
5. create its mixed condition sequence;
6. reuse that one parameter vector for every trial in the dataset;
7. simulate each trial through the shared scientific kernel;
8. construct the exact posterior-condition keys used during training;
9. stack true parameters with shape `(n_test_sims,)` per parameter;
10. return an `EvaluationBatch` with JSON-safe scenario metadata.

Example return:

```python
return EvaluationBatch(
    posterior_conditions={
        "x": observations,
        "tta_per_trial": tta_per_trial,
        "number_of_trials": number_of_trials,
    },
    true_parameters=true_parameters,
    metadata={
        "interpretation": "out_of_distribution_stress_test",
        "condition_values": condition_values.tolist(),
        "trials_per_tta": trials_per_tta,
        "n_trials": int(observations.shape[1]),
        "shuffle_tta": shuffle_tta,
        "seed": seed,
    },
)
```

### 5.4 Registration

```python
EVALUATIONS = [
    EvaluationSpec(
        name="mixed-tta",
        model_name="model_DC",
        generate_batch=generate_dc_mixed_tta,
        description=(
            "Mixed-TTA recovery after homogeneous-TTA training"
        ),
    ),
]
```

Registration separates capability definition from command dispatch. The CLI
selects `(model_name, scenario_name)` without importing a named article module.

---

## 6. Runner Module: `runner.py`

### 6.1 Responsibility

The runner owns execution order. It is an application-service module: it
coordinates other components but should contain neither TTA equations nor
metric formulas.

### 6.2 Main function

```python
def run_evaluation(
    *,
    artifact_ref: str,
    model_spec: ModelSpec,
    evaluation_spec: EvaluationSpec,
    config: dict,
    options: dict,
) -> dict:
    ...
```

Inputs:

- `artifact_ref`: ID or alias supplied by the user;
- `model_spec`: discovered model contract;
- `evaluation_spec`: discovered scenario capability;
- `config`: engine runtime paths and device configuration;
- `options`: evaluation/scenario options.

Output:

- a JSON-serializable report dictionary;
- durable files written beneath the configured results root.

### 6.3 Correct execution order

```text
resolve metadata
    ↓
validate artifact/model/scenario identity
    ↓
resolve and verify checkpoint path
    ↓
generate EvaluationBatch
    ↓
validate EvaluationBatch
    ↓
load approximator
    ↓
sample posterior
    ↓
validate posterior shape
    ↓
compute recovery metrics
    ↓
write reports
    ↓
return report
```

Generate and validate the relatively cheap batch before loading or sampling a
large network. Checkpoint existence can be checked earlier because it is a
cheap artifact-contract check.

### 6.4 Batch validator

Do not leave validation as scattered inline checks. Give it one private
function with no I/O:

```python
def _validate_evaluation_batch(
    batch: EvaluationBatch,
    model_spec: ModelSpec,
) -> int:
    """Validate the batch and return its number of datasets."""
```

Required checks:

```python
if not isinstance(batch, EvaluationBatch):
    raise TypeError(...)

if not batch.posterior_conditions:
    raise ValueError(...)

expected = set(model_spec.par_names)
actual = set(batch.true_parameters)
if actual != expected:
    raise ValueError(...)

target_lengths = {
    name: np.asarray(values).reshape(-1).shape[0]
    for name, values in batch.true_parameters.items()
}

if len(set(target_lengths.values())) != 1:
    raise ValueError(...)

n_datasets = next(iter(target_lengths.values()))
if n_datasets < 1:
    raise ValueError(...)

json.dumps(batch.metadata)
return n_datasets
```

Also validate that each relevant posterior-condition array has a compatible
leading dataset dimension. Some scalar or global scenario fields may be valid,
so this rule must follow the adapter contract rather than blindly requiring
every value to be an array of length `n_datasets`.

### 6.5 Sampling contract

For the installed BayesFlow API, use keyword-only arguments:

```python
n_posterior_samples = int(
    options.get("n_posterior_samples", 1000)
)
if n_posterior_samples < 1:
    raise ValueError(...)

posterior_samples = approximator.sample(
    conditions=batch.posterior_conditions,
    num_samples=n_posterior_samples,
)
```

For ordinary global parameter inference, the conceptual combined shape is:

```text
(number of datasets, posterior draws, parameters)
```

BayesFlow may return a dictionary whose parameter values each have shape:

```text
(number of datasets, posterior draws)
```

The trial dimension belongs to the input conditions and normally does not
remain in the global parameter posterior.

### 6.6 Runner pseudo-implementation

```python
def run_evaluation(...):
    metadata = resolve_artifact_ref(artifact_ref, config)

    if metadata["spec_name"] != model_spec.name:
        raise ValueError(...)

    if evaluation_spec.model_name != model_spec.name:
        raise ValueError(...)

    ctx = resolve_context_from_artifact_id(
        metadata["artifact_id"],
        config,
    )
    if not ctx.checkpoint_path.exists():
        raise FileNotFoundError(...)

    batch = evaluation_spec.generate_batch(model_spec, options)
    n_datasets = _validate_evaluation_batch(batch, model_spec)

    approximator = keras.saving.load_model(ctx.checkpoint_path)
    n_posterior_samples = _require_positive_int(
        options,
        "n_posterior_samples",
        default=1000,
    )
    posterior_samples = approximator.sample(
        conditions=batch.posterior_conditions,
        num_samples=n_posterior_samples,
    )

    metrics, summaries = compute_recovery_metrics(
        posterior_samples=posterior_samples,
        true_parameters=batch.true_parameters,
        parameter_names=model_spec.par_names,
        expected_n_datasets=n_datasets,
    )

    return write_evaluation_outputs(
        artifact_metadata=metadata,
        model_spec=model_spec,
        evaluation_spec=evaluation_spec,
        context=ctx,
        batch=batch,
        posterior_samples=posterior_samples,
        posterior_summaries=summaries,
        metrics=metrics,
        n_posterior_samples=n_posterior_samples,
        visualize=bool(options.get("visualize", False)),
    )
```

The runner should not call `build_model_from_spec()`: scientific data
generation belongs to the scenario, and the saved approximator already owns
the adapter required for sampling.

---

## 7. Metrics Module: `metrics.py`

### 7.1 Responsibility

`metrics.py` transforms posterior samples and known targets into numeric,
model-independent summaries. It performs no simulation, artifact lookup,
plotting, JSON writing, or directory creation.

The first implementation is specifically a generic **parameter-recovery**
metric function. It is not yet a universal implementation for every possible
evaluation family.

### 7.2 Public function

```python
def compute_recovery_metrics(
    *,
    posterior_samples: dict[str, np.ndarray] | np.ndarray,
    true_parameters: dict[str, np.ndarray],
    parameter_names: list[str],
    expected_n_datasets: int,
) -> tuple[
    dict[str, dict[str, float | None]],
    dict[str, dict[str, np.ndarray]],
]:
    ...
```

Outputs:

- `metrics`: JSON-safe scalar values for each parameter;
- `summaries`: NumPy arrays used for plots and NPZ output.

Example:

```python
metrics = {
    "b0": {
        "bias": 0.03,
        "rmse": 0.12,
        "pearson": 0.91,
        "posterior_std_mean": 0.08,
        "coverage_95": 0.94,
    }
}

summaries = {
    "b0": {
        "mean": np.ndarray(shape=(N,)),
        "std": np.ndarray(shape=(N,)),
        "lower": np.ndarray(shape=(N,)),
        "upper": np.ndarray(shape=(N,)),
    }
}
```

### 7.3 Normalize one parameter at a time

```python
def _posterior_draws_for_parameter(
    posterior_samples: dict[str, np.ndarray] | np.ndarray,
    parameter_name: str,
    parameter_index: int,
) -> np.ndarray:
    """Return shape (datasets, posterior_draws)."""
```

Rules:

- dictionary form: obtain `posterior_samples[parameter_name]`;
- combined-array form: select `[..., parameter_index]`;
- accept only shape `(N, S)` after extraction;
- do not call unrestricted `np.squeeze()`, because `(1, S)` would become
  `(S,)` and lose the dataset axis;
- if BayesFlow returns a documented singleton parameter axis `(N, S, 1)`,
  remove only that final axis explicitly.

### 7.4 Summarize posterior draws

```python
def _summarize_parameter_draws(
    draws: np.ndarray,
    *,
    lower_quantile: float = 0.025,
    upper_quantile: float = 0.975,
) -> dict[str, np.ndarray]:
    ...
```

Given `draws.shape == (N, S)`, calculate across `axis=1`:

```python
{
    "mean": np.mean(draws, axis=1),
    "std": np.std(draws, axis=1),
    "lower": np.quantile(draws, lower_quantile, axis=1),
    "upper": np.quantile(draws, upper_quantile, axis=1),
}
```

Every returned array has shape `(N,)`.

### 7.5 Calculate scalar recovery metrics

For each parameter:

```python
error = summary["mean"] - target

bias = float(np.mean(error))
rmse = float(np.sqrt(np.mean(error**2)))
coverage = float(np.mean(
    (summary["lower"] <= target)
    & (target <= summary["upper"])
))
```

Pearson correlation is defined only when:

- at least two datasets exist;
- target values have non-zero variance;
- posterior means have non-zero variance.

Use `None` when it is undefined so JSON output remains standards-compliant.

### 7.6 BayesFlow diagnostics versus engine metrics

BayesFlow 2.0.8 already provides diagnostics such as:

- `bayesflow.diagnostics.recovery` for plotting;
- `root_mean_squared_error`;
- `calibration_error`;
- `posterior_z_score`;
- `posterior_contraction`.

Do not assume these are exact replacements for the old metrics. For example,
BayesFlow's RMSE implementation computes errors for posterior draws and then
aggregates, while the current mixed-TTA metric computes RMSE from posterior
means. Both are meaningful, but they are not numerically equivalent.

For the first refactor:

1. preserve the old posterior-mean recovery metrics in `metrics.py`;
2. use BayesFlow's `recovery()` for the generic recovery plot;
3. add further BayesFlow calibration diagnostics later as explicitly named
   report fields and tests.

This preserves behavior before expanding it.

### 7.7 Problems to correct in the current implementation

The current `metrics.py` should be revised as follows:

- remove unrestricted `np.squeeze()`;
- remove duplicate calculation of `std`, `lower`, and `upper`;
- replace the tautological comparison
  `n_datasets != draws.shape[0]` with comparison against the validated target
  length or `expected_n_datasets`;
- stop guessing whether dataset axis is 0 or 1 after the contract has been
  defined; require `(datasets, draws)` and fail clearly otherwise;
- verify `target.shape == (expected_n_datasets,)` before subtraction;
- validate quantile bounds if they become configurable.

---

## 8. Reporting Module: `reporting.py`

### 8.1 Responsibility

Reporting converts evaluation results into durable user-facing artifacts. It
owns filenames and output layout, but not scientific interpretation.

It should support two levels that existed in the old mixed-TTA module:

#### Level A: visual recovery

- BayesFlow recovery figure;
- basic JSON execution report.

This reproduces the behavior of
`evaluate_mixed_tta_artifact_with_bf_recovery()`.

#### Level B: extended recovery

- recovery figure;
- posterior samples and targets in NPZ;
- posterior summaries;
- bias, RMSE, correlation, and coverage in JSON.

This reproduces the additional behavior of `evaluate_mixed_tta_artifact()`.

The generic pipeline can produce Level B by default; Level A is naturally a
subset rather than a separate scientific pipeline.

### 8.2 Small reporting functions

```python
def create_evaluation_output_dir(
    results_dir: Path,
    evaluation_name: str,
) -> Path:
    """Create and return results/<artifact>/evaluations/<name>."""
```

```python
def save_recovery_plot(
    *,
    posterior_samples: dict[str, np.ndarray] | np.ndarray,
    true_parameters: dict[str, np.ndarray],
    parameter_names: list[str],
    output_path: Path,
    visualize: bool = False,
) -> None:
    ...
```

Use BayesFlow:

```python
figure = recovery(
    estimates=posterior_samples,
    targets=true_parameters,
    variable_keys=parameter_names,
    variable_names=parameter_names,
)
```

```python
def save_evaluation_arrays(
    *,
    output_path: Path,
    posterior_samples: dict[str, np.ndarray] | np.ndarray,
    true_parameters: dict[str, np.ndarray],
    posterior_summaries: dict[str, dict[str, np.ndarray]],
) -> None:
    ...
```

Use stable prefixes:

```text
posterior_<parameter>
true_<parameter>
posterior_mean_<parameter>
posterior_std_<parameter>
posterior_lower_<parameter>
posterior_upper_<parameter>
```

```python
def build_evaluation_report(
    *,
    artifact_metadata: dict,
    model_spec: ModelSpec,
    evaluation_spec: EvaluationSpec,
    checkpoint_path: Path,
    batch: EvaluationBatch,
    n_datasets: int,
    n_posterior_samples: int,
    metrics: dict,
    output_paths: dict[str, Path],
) -> dict:
    """Build, but do not write, a JSON-safe report."""
```

```python
def write_json_report(
    report: dict,
    output_path: Path,
) -> None:
    ...
```

### 8.3 Separate report construction from writing

Yes, `build_evaluation_report()` should be separate. It is a pure
transformation that can be unit-tested without touching the filesystem.
`write_json_report()` owns persistence. This separation makes failure and tests
clearer.

The generic report should look like:

```python
{
    "artifact_id": artifact_metadata["artifact_id"],
    "alias": artifact_metadata.get("alias"),
    "spec_name": model_spec.name,
    "evaluation_name": evaluation_spec.name,
    "evaluation_description": evaluation_spec.description,
    "checkpoint_path": str(checkpoint_path),
    "created_at": datetime.now().isoformat(),
    "n_datasets": n_datasets,
    "n_posterior_samples": n_posterior_samples,
    "scenario_metadata": batch.metadata,
    "metrics": metrics,
    "outputs": {
        name: str(path)
        for name, path in output_paths.items()
    },
}
```

Do not hard-code these fields in engine reporting:

```text
trials_per_tta
shuffle_tta
condition_values
interpretation = out_of_distribution_stress_test
```

They belong in `batch.metadata` and appear beneath `scenario_metadata`.

### 8.4 Reporting orchestrator

The runner can call every small reporting function directly. If that begins to
obscure the runner, add one orchestration function inside the same module—not a
new module:

```python
def write_evaluation_outputs(
    *,
    artifact_metadata: dict,
    model_spec: ModelSpec,
    evaluation_spec: EvaluationSpec,
    context: WorkflowContext,
    batch: EvaluationBatch,
    posterior_samples,
    posterior_summaries: dict,
    metrics: dict,
    n_datasets: int,
    n_posterior_samples: int,
    visualize: bool,
) -> dict:
    ...
```

It should:

1. create the generic output directory;
2. define the PDF, NPZ, and JSON paths;
3. save the recovery plot;
4. save arrays;
5. build the report;
6. write JSON;
7. return the report.

### 8.5 Problems to correct in the current implementation

- Import `datetime` correctly. `import datetime` requires
  `datetime.datetime.now()`, while `from datetime import datetime` allows
  `datetime.now()`.
- Remove unused workflow imports from `reporting.py`; reporting should not build
  a model or create checkpoint/log directories.
- Add `write_json_report()`; the current code builds a report but does not write
  it.
- Ensure `output_path.parent` exists or call output-directory creation before
  saving.
- Verify that report values are JSON-safe before writing.

---

## 9. Discovery Module Extension

### 9.1 Responsibility

Discovery turns plugin declarations into a capability registry. It does not run
an evaluation.

### 9.2 Required contract changes

Add markers:

```python
"EVALUATIONS"
"register_evaluations"
```

Extend the registry:

```python
@dataclass(frozen=True)
class PluginRegistry:
    models: dict[str, ModelSpec]
    workflows: dict[str, Workflow]
    evaluations: dict[tuple[str, str], EvaluationSpec]
```

Add:

```python
def _validate_evaluations(
    value,
    source: str,
) -> list[EvaluationSpec]:
    ...
```

```python
def _collect_evaluations_from_module(
    module,
) -> list[EvaluationSpec]:
    ...
```

Registry key:

```python
key = (evaluation.model_name, evaluation.name)
```

Reject duplicates rather than silently overwriting them.

### 9.3 Invariant across registries

Discovery can validate type and duplicate keys immediately. Whether every
evaluation refers to a discovered model may be validated after all modules have
been collected, because plugin scan order should not determine correctness.

---

## 10. CLI Module Extension

### 10.1 Responsibility

The CLI translates user syntax into typed runtime values and selects registered
capabilities. It should not generate TTA sequences, calculate recovery metrics,
or create plots.

### 10.2 Generic command

```text
bayesflow-cli evaluate \
    --artifact run-001 \
    --scenario mixed-tta \
    --n-test-sims 20 \
    --n-posterior-samples 1000 \
    --seed 2026 \
    --scenario-option trials_per_tta=15 \
    --scenario-option shuffle_tta=true
```

### 10.3 Option parser

```python
def parse_scenario_options(
    entries: list[str],
) -> dict[str, str | int | float | bool]:
    ...
```

Requirements:

- require `KEY=VALUE`;
- reject empty keys and duplicate keys;
- parse documented booleans, integers, floats, and strings;
- never use `eval()`;
- scenario performs final semantic validation.

### 10.4 Dispatch helper

```python
def require_evaluation(
    evaluations: dict[tuple[str, str], EvaluationSpec],
    model_name: str,
    scenario_name: str,
) -> EvaluationSpec:
    ...
```

On failure, list scenarios available for that model.

Dispatch order:

1. discover plugins once;
2. resolve artifact metadata;
3. resolve its model spec;
4. resolve `(model_name, scenario_name)`;
5. parse options;
6. call `run_evaluation()`;
7. print the report/output location.

---

## 11. Tests Derived From Contracts

Tests should be derived from module boundaries, not merely from lines of code.

### 11.1 Interface and scenario tests

- fixed seed returns identical batches;
- target keys equal the model parameter names;
- all target arrays have equal dataset counts;
- mixed datasets contain every requested condition;
- one parameter vector is reused across trials of one dataset;
- DC, DF, and SF call their correct shared scientific kernels;
- evaluation conditions are compatible with the training adapter contract.

### 11.2 Runner tests with fakes

Use a fake approximator and monkeypatch checkpoint loading. Test:

- wrong artifact model fails before sampling;
- wrong scenario model fails before sampling;
- missing checkpoint fails clearly;
- empty conditions fail before loading the approximator;
- mismatched target keys fail;
- unequal target lengths fail;
- non-serializable metadata fails;
- `sample()` receives exactly `batch.posterior_conditions`;
- requested posterior sample count is forwarded;
- runner works for a fake non-TTA scenario.

### 11.3 Metric tests with small exact arrays

Use arrays whose expected values can be calculated by hand:

```python
draws = {
    "theta": np.array([
        [0.0, 1.0, 2.0],
        [2.0, 3.0, 4.0],
    ])
}
targets = {"theta": np.array([1.0, 3.0])}
```

Test:

- summary shapes are `(2,)`;
- means are `[1.0, 3.0]`;
- bias and RMSE are zero;
- coverage is one;
- one-dataset shape `(1, S)` remains two-dimensional;
- wrong posterior rank fails;
- target length mismatch fails;
- undefined Pearson correlation becomes `None`.

### 11.4 Reporting tests

- output directory follows the generic convention;
- report construction performs no filesystem write;
- JSON report is serializable;
- scenario metadata is preserved unchanged;
- NPZ contains stable generic keys;
- plotting is called with targets separate from posterior conditions;
- reporting creates no checkpoint or log directory.

### 11.5 Discovery and CLI tests

- `EVALUATIONS` and `register_evaluations()` are discovered;
- invalid return values fail;
- duplicate `(model, scenario)` keys fail;
- unknown scenario errors list valid scenarios;
- scenario options are parsed safely;
- CLI dispatches to the generic runner once.

---

## 12. Implementation Order

Implement in dependency order so each layer can be tested before the next.

### Checkpoint 1: Stabilize interfaces

Files:

```text
bayesflow_models/evaluation/interfaces.py
bayesflow_models/evaluation/__init__.py
```

Actions:

1. finalize `EvaluationBatch` and `EvaluationSpec`;
2. export only public plugin-facing types;
3. write simple construction/type tests.

### Checkpoint 2: Complete batch validation

File:

```text
bayesflow_models/evaluation/runner.py
```

Actions:

1. implement `_validate_evaluation_batch()`;
2. test keys, lengths, empty batches, and metadata serialization;
3. do not load a real checkpoint in these tests.

### Checkpoint 3: Complete recovery metrics

File:

```text
bayesflow_models/evaluation/metrics.py
```

Actions:

1. normalize one parameter's draws to `(N, S)`;
2. calculate summaries across axis 1;
3. calculate scalar metrics;
4. test with hand-computed arrays;
5. preserve the semantics of the old mixed-TTA metrics.

### Checkpoint 4: Complete reporting

File:

```text
bayesflow_models/evaluation/reporting.py
```

Actions:

1. create generic output directory;
2. save BayesFlow recovery PDF;
3. save generic NPZ arrays;
4. build a pure report dictionary;
5. write JSON;
6. test using `tmp_path` and a mocked plotting function.

### Checkpoint 5: Complete the runner

File:

```text
bayesflow_models/evaluation/runner.py
```

Actions:

1. correct artifact and scenario identity checks;
2. use keyword-only `approximator.sample()` arguments;
3. remove unused model construction;
4. connect validation, sampling, metrics, and reporting;
5. return the report;
6. test with a fake approximator.

### Checkpoint 6: Extend discovery

File:

```text
bayesflow_models/discovery.py
```

Actions:

1. add evaluation markers;
2. validate and collect evaluation specs;
3. add the evaluation registry;
4. test malformed and duplicate registrations.

### Checkpoint 7: Build the workspace scenario

Files:

```text
examples/article_workspace/plugins/reproduce_models.py
examples/article_workspace/plugins/mixed_tta_scenarios.py
```

Actions:

1. extract shared seeded parameter samplers;
2. extract shared per-trial scientific kernels;
3. generate model-specific mixed-TTA batches;
4. register the three model/scenario pairs;
5. test scientific invariants and adapter compatibility.

### Checkpoint 8: Add generic CLI dispatch

File:

```text
cli.py
```

Actions:

1. add `evaluate` parser;
2. parse generic scenario options safely;
3. select the model-bound scenario;
4. call the generic runner;
5. test dispatch without loading a real network.

### Checkpoint 9: Prove equivalence, then remove legacy code

Actions:

1. run old and new pipelines with the same small seed/options where possible;
2. compare condition/target shapes and scientific invariants;
3. compare expected output categories, not timestamps or generated filenames;
4. verify the new report includes the old mixed-TTA facts under
   `scenario_metadata`;
5. remove `mixed-recovery` only after the generic command works;
6. delete `bayesflow_models/mixed_tta_evaluation.py` only after equivalence and
   regression tests pass.

---

## 13. Current-Code Review Checklist

Before continuing from the present implementation, correct these concrete
issues.

### `metrics.py`

- [ ] Avoid unrestricted `np.squeeze()`.
- [ ] Require one parameter's draws to have shape `(datasets, draws)`.
- [ ] Remove duplicate summary calculations.
- [ ] Compare posterior dataset count with target length.
- [ ] Remove sample-axis guessing once the shape contract is enforced.
- [ ] Return JSON-safe scalar metrics and array-valued summaries separately.

### `reporting.py`

- [ ] Correct the `datetime` import/use mismatch.
- [ ] Remove unused workflow/model-building imports.
- [ ] Add `write_json_report()`.
- [ ] Preserve TTA fields only through `batch.metadata`.
- [ ] Keep report construction separate from report writing.
- [ ] Optionally add `write_evaluation_outputs()` to simplify the runner.

### `runner.py`

- [ ] Remove the invalid `metadata["model_name"]` check; metadata uses
  `spec_name`.
- [ ] Import metrics/reporting through package-relative imports.
- [ ] Generate and validate the batch before loading the large approximator.
- [ ] Remove `build_model_from_spec()` and broad directory setup.
- [ ] Call `sample(conditions=..., num_samples=...)` with keyword arguments.
- [ ] Read `n_posterior_samples` from evaluation options.
- [ ] Define `n_posterior_samples` before passing it to reporting.
- [ ] Write the JSON report and return it.

### Discovery, scenario, and CLI

- [ ] Extend registration to evaluations.
- [ ] Implement reusable scientific kernels rather than copying the old
  simulator.
- [ ] Register model-specific mixed-TTA scenarios.
- [ ] Add generic `evaluate` dispatch.
- [ ] Retain the old command until equivalence tests pass.

---

## 14. General Lessons for Modular Product Design

### Lesson 1: Modular thinking starts with change and ownership

Do not begin by asking, “How many files should I create?” Begin by asking,
“Which decisions change for different reasons, and who has the knowledge to
make them?”

### Lesson 2: A contract includes shapes and failures

Naming a value `posterior_samples` is insufficient. A useful contract says:

```text
dict keyed by parameter name
each value shaped (datasets, posterior draws)
keys exactly match ModelSpec.par_names
wrong shapes fail before metrics are calculated
```

### Lesson 3: Keep policy close to its owner

The engine owns output conventions. The scenario owns scientific
interpretation. Therefore, the engine decides where `report.json` is written,
while the scenario supplies `"interpretation": "out_of_distribution..."`.

### Lesson 4: Orchestration is a real module responsibility

A runner can be valuable even if there is only one sampling implementation.
Its responsibility is sequencing and enforcing boundaries, not providing
multiple sampling algorithms.

### Lesson 5: Reuse behavior before introducing abstractions

First generalize the proven mixed-TTA recovery behavior. Introduce metric
protocols, reporter classes, or dependency-injection frameworks only after a
second real implementation demonstrates what must vary.

### Lesson 6: Pure functions make boundaries testable

Functions such as `_validate_evaluation_batch()`,
`compute_recovery_metrics()`, and `build_evaluation_report()` should avoid I/O.
They become easy to test with small values, while persistence remains isolated
in reporting functions.

### Lesson 7: Extensibility is constrained replaceability

A system is extensible when a new scenario can be added by implementing a
small, validated contract and registering it—without editing artifact loading,
metrics, reporting, or CLI scientific logic.

That is the final test of Workstream 2: a fake non-TTA scenario must pass
through the same runner without the engine learning any new scientific terms.

