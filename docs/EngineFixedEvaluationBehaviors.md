# Engine-Fixed Evaluation Behaviors

## Purpose

This document practices modular thinking for only the **engine-fixed** part of
Workstream 2. It deliberately excludes:

- mixed-TTA generation;
- model-specific simulation and priors;
- recovery metric formulas;
- recovery plots;
- scenario-specific report fields;
- future evaluation-specific algorithms.

The question answered here is:

> Which mechanisms must the application perform in the same way for every
> registered evaluation scenario?

The functions below are design candidates. The purpose of the templates is to
decide boundaries before writing code. Not every candidate must become a new
public function. Some should remain private helpers, and existing artifact
functions should be reused rather than duplicated.

---

## 1. Criteria for an Engine-Fixed Behavior

A behavior belongs to the engine-fixed category when:

1. it is required regardless of the scientific scenario;
2. its rules do not mention TTA, DDM equations, or a particular model family;
3. the engine has enough information to perform it correctly;
4. researchers should not have to reimplement it in every plugin;
5. centralizing it improves consistency, validation, portability, or security.

Examples include artifact resolution, contract validation, checkpoint loading,
posterior sampling coordination, output layout, and orchestration.

An important distinction is:

```text
fixed mechanism != fixed scientific algorithm
```

For example, “invoke the selected metric implementation” could eventually be
an engine-fixed mechanism, while the selected metric formula may vary. In the
first Workstream 2 version, recovery metrics are one reusable
evaluation-type-specific implementation and are therefore outside this
document.

---

## 2. Function Inventory

The engine-fixed evaluation path can be expressed as these candidate
functions:

```text
resolve_evaluation_artifact()
        ↓
validate_evaluation_batch()
        ↓
load_artifact_approximator()
        ↓
sample_posterior()
        ↓
create_evaluation_output_dir()
        ↓
build_evaluation_report_envelope()
        ↓
write_json_report()
```

One orchestration function connects them:

```text
run_evaluation()
```

At the application boundary, two additional fixed helpers support generic CLI
dispatch:

```text
parse_scenario_options()
require_evaluation()
```

The following sections apply the complete design template to each function.

---

## 3. Resolve and Validate an Evaluation Artifact

### Function question

How does every evaluation turn an artifact ID or alias into a verified,
portable runtime context?

### Proposed function

```python
def resolve_evaluation_artifact(
    *,
    artifact_ref: str,
    model_spec: ModelSpec,
    evaluation_spec: EvaluationSpec,
    config: dict,
) -> tuple[dict, WorkflowContext]:
    ...
```

### Responsibility

- Resolve an artifact reference through the existing artifact API.
- Verify that the artifact belongs to `model_spec`.
- Verify that the evaluation scenario belongs to `model_spec`.
- Reconstruct current portable paths through the runtime configuration.
- Verify that the checkpoint file exists.
- Return artifact metadata and its runtime context.

It must not load the Keras model, generate evaluation data, or create output
directories.

### Owner

Engine, most naturally `evaluation/runner.py` as a private helper.

### Input

- `artifact_ref: str`: artifact ID or alias such as `run-001`;
- `model_spec: ModelSpec`: discovered model selected for the artifact;
- `evaluation_spec: EvaluationSpec`: discovered scenario;
- `config: dict`: current checkpoint, results, logs, and device configuration.

### Output

```python
tuple[dict, WorkflowContext]
```

- resolved metadata;
- runtime context whose paths come from current configuration.

### Preconditions

- `config["paths"]["checkpoints"]` exists as a configured value;
- `model_spec` and `evaluation_spec` were obtained from trusted discovery;
- the artifact metadata file is present in the configured checkpoint root.

### Postconditions

- `metadata["spec_name"] == model_spec.name`;
- `evaluation_spec.model_name == model_spec.name`;
- `context.artifact_id == metadata["artifact_id"]`;
- `context.checkpoint_path` is reconstructed from current configuration;
- `context.checkpoint_path.exists()` is true.

### Side effects

Normally read-only. The existing alias-resolution implementation may add a
missing alias to legacy metadata. If that behavior remains, document it as an
artifact-layer side effect rather than hiding it.

### Errors

- `FileNotFoundError`: metadata or checkpoint is absent;
- `ValueError`: artifact model and requested model differ;
- `ValueError`: evaluation scenario and model differ;
- metadata parsing errors if the file is malformed.

### Dependencies

- `resolve_artifact_ref()`;
- `resolve_context_from_artifact_id()`;
- `ModelSpec`, `EvaluationSpec`, and `WorkflowContext`.

### Variation

Artifact storage may later change from a local filesystem to another backend.
That variation should remain behind the existing artifact API. Scientific
scenarios must not affect this function.

### Tests

- exact artifact ID resolves;
- alias resolves;
- wrong model fails;
- scenario registered for another model fails;
- missing checkpoint fails;
- moved workspace resolves paths from the current config, not stored absolute
  metadata paths.

---

## 4. Validate the Evaluation Batch Contract

### Function question

How does the engine reject invalid scenario output before loading or sampling a
large posterior network?

### Proposed function

```python
def validate_evaluation_batch(
    *,
    batch: EvaluationBatch,
    model_spec: ModelSpec,
) -> int:
    """Validate the batch and return its dataset count."""
```

### Responsibility

- Verify the returned object is an `EvaluationBatch`.
- Verify posterior conditions are present.
- Verify true-parameter keys exactly match `model_spec.par_names`.
- Verify every target has the same non-zero dataset dimension.
- Verify scenario metadata is JSON-serializable.
- Return the validated number of datasets for later shape checks and reports.

It must not validate TTA values or other scientific rules. Those belong to the
scenario.

### Owner

Engine, `evaluation/runner.py`. It can begin as a private helper named
`_validate_evaluation_batch`.

### Input

- `batch: EvaluationBatch`;
- `model_spec: ModelSpec`.

### Output

```python
int
```

The number of independent evaluation datasets.

### Preconditions

- a scenario generator has returned a value;
- `model_spec.par_names` defines the trained model's target ordering.

### Postconditions

- target names exactly match model parameter names;
- every target can be treated as a one-dimensional array of length
  `n_datasets`;
- `n_datasets >= 1`;
- `batch.metadata` can be included in JSON reporting.

### Side effects

None. This should be a pure function.

### Errors

- `TypeError`: scenario returned the wrong object type;
- `ValueError`: conditions are empty;
- `ValueError`: target names differ;
- `ValueError`: target dataset dimensions differ or are empty;
- `TypeError` or a wrapped `ValueError`: metadata is not JSON-serializable.

### Dependencies

- `EvaluationBatch`;
- `ModelSpec`;
- NumPy for shape normalization;
- `json.dumps()` for serialization validation.

### Variation

The exact keys and inner shapes of posterior conditions vary by model adapter.
The common validator should only enforce rules that are universally known. A
scenario/model compatibility test should enforce detailed condition shapes.

### Tests

- valid batch returns dataset count;
- wrong return type fails;
- empty conditions fail;
- missing, additional, or misspelled targets fail;
- unequal target lengths fail;
- zero datasets fail;
- non-serializable metadata fails;
- arbitrary model-specific condition keys are accepted when otherwise valid.

---

## 5. Load the Saved Posterior Approximator

### Function question

How does the engine load a verified checkpoint without mixing model loading
with artifact resolution or scientific simulation?

### Proposed function

```python
def load_artifact_approximator(
    checkpoint_path: Path,
):
    ...
```

### Responsibility

Load and return the saved posterior approximator from a checkpoint path that
has already been resolved and verified.

### Owner

Engine, `evaluation/runner.py` as a private I/O helper, unless an existing
shared checkpoint loader is introduced later.

### Input

```python
checkpoint_path: Path
```

### Output

The loaded Keras/BayesFlow approximator. Its narrow static type may initially
be `Any` because the stable behavior required by the runner is structural:

```python
approximator.sample(
    conditions=...,
    num_samples=...,
)
```

### Preconditions

- the path belongs to the resolved artifact context;
- the checkpoint exists;
- the image contains compatible Keras/BayesFlow dependencies.

### Postconditions

- the returned object exposes a callable `sample()` method;
- no report directory has been created.

### Side effects

- reads a potentially large file;
- allocates CPU/GPU memory;
- may initialize framework/backend state.

### Errors

- `FileNotFoundError`: preferably caught before calling this helper;
- Keras deserialization errors for incompatible or corrupted artifacts;
- resource errors such as insufficient memory.

Do not replace these with a misleading “evaluation failed” message. Add
artifact context while preserving the original exception as the cause if an
application-specific error is introduced.

### Dependencies

- `keras.saving.load_model()`;
- `pathlib.Path`.

### Variation

The serialization backend may change later. Scientific scenarios do not vary
this behavior. If multiple checkpoint formats appear, this helper becomes the
single dispatch point.

### Tests

- monkeypatch `keras.saving.load_model` and verify the exact path;
- verify the returned fake approximator is unchanged;
- verify loading happens only after batch validation in a runner test;
- integration-test one real compatible checkpoint separately from unit tests.

---

## 6. Sample the Posterior

### Function question

How does the engine invoke the approximator consistently for every scenario?

### Proposed function

```python
def sample_posterior(
    *,
    approximator,
    batch: EvaluationBatch,
    n_posterior_samples: int,
):
    ...
```

### Responsibility

- Validate that the requested sample count is positive.
- Call the approximator with the scenario-provided posterior conditions.
- Return posterior samples without calculating metrics or saving output.

### Owner

Engine, `evaluation/runner.py`. It may remain inline in `run_evaluation()` if
it is only a few clear lines. The template is still useful even when no helper
is created.

### Input

- loaded approximator;
- validated `EvaluationBatch`;
- positive posterior sample count.

### Output

BayesFlow posterior samples, normally:

```python
dict[str, np.ndarray]
```

with each global parameter typically shaped:

```text
(datasets, posterior draws)
```

or a combined representation shaped:

```text
(datasets, posterior draws, parameters)
```

### Preconditions

- batch validation has succeeded;
- approximator loading has succeeded;
- `n_posterior_samples >= 1`;
- conditions match the adapter embedded in the saved approximator.

### Postconditions

- one posterior distribution has been sampled for every evaluation dataset;
- no metric or report has been produced.

### Side effects

- computationally expensive CPU/GPU execution;
- memory allocation;
- progress output may be emitted by BayesFlow.

### Errors

- `ValueError`: sample count is not positive;
- adapter errors if condition keys/shapes are incompatible;
- backend/resource errors.

### Dependencies

- the approximator's public `sample()` API;
- `EvaluationBatch`.

### Variation

Batch size or sampling backend could later become generic execution options.
Scientific scenarios vary the conditions but do not replace the engine's
invocation contract in the first version.

### Tests

- fake approximator receives exactly `batch.posterior_conditions`;
- fake approximator receives the requested sample count;
- keyword-only call uses `conditions=` and `num_samples=`;
- zero or negative counts fail before calling the approximator;
- fake return value is passed through unchanged.

---

## 7. Create the Generic Evaluation Output Directory

### Function question

How does every evaluation receive a predictable output location independent of
its scientific meaning?

### Proposed function

```python
def create_evaluation_output_dir(
    *,
    results_dir: Path,
    evaluation_name: str,
) -> Path:
    ...
```

### Responsibility

Create and return the standard directory:

```text
results/<artifact-id>/evaluations/<evaluation-name>/
```

The incoming `results_dir` is already the artifact-specific result directory
from `WorkflowContext`.

### Owner

Engine, `evaluation/reporting.py`.

### Input

- artifact-specific `results_dir`;
- registered evaluation name.

### Output

The created `Path`.

### Preconditions

- `results_dir` came from the portable runtime context;
- evaluation name is a validated registry name, not an arbitrary path.

### Postconditions

- returned directory exists;
- no checkpoint or log directory was created.

### Side effects

Creates directories beneath the configured results root.

### Errors

- permission and filesystem errors;
- invalid evaluation names if path-safety validation is added.

### Dependencies

- `pathlib.Path`.

### Variation

The global output convention may change in one engine release. Individual
scenarios should not select arbitrary output roots.

### Tests

- use `tmp_path` and verify the exact directory;
- repeated calls are idempotent;
- checkpoint/log paths remain absent;
- unsafe names such as path traversal are rejected if names can originate from
  untrusted input.

---

## 8. Build the Generic Report Envelope

### Function question

How does the engine describe every evaluation consistently while allowing
scenario- and evaluation-specific result payloads?

### Proposed function

```python
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
    ...
```

### Responsibility

Build a JSON-safe report with a stable engine-owned envelope. Accept
evaluation-type-specific results as an opaque `result_payload` rather than
hard-coding recovery metrics into the engine-wide contract.

### Owner

Engine, `evaluation/reporting.py`.

### Input

- artifact/model/evaluation identity;
- portable checkpoint path;
- validated batch metadata;
- execution counts;
- result payload produced by the selected evaluation mechanism;
- paths to durable outputs.

### Output

A dictionary such as:

```python
{
    "artifact_id": "...",
    "alias": "run-001",
    "spec_name": "model_DC",
    "evaluation_name": "mixed-tta",
    "evaluation_description": "...",
    "checkpoint_path": "...",
    "created_at": "...",
    "n_datasets": 20,
    "n_posterior_samples": 1000,
    "scenario_metadata": {...},
    "results": result_payload,
    "outputs": {...},
}
```

### Preconditions

- batch and counts were validated;
- scenario metadata is JSON-safe;
- result payload contains no raw NumPy arrays unless converted;
- output paths are finalized.

### Postconditions

- the returned report is JSON-serializable;
- scenario facts remain under `scenario_metadata`;
- evaluation results remain under `results`;
- no file has been written.

### Side effects

None. This should be a pure function except for obtaining the current
timestamp. For fully deterministic tests, the timestamp could be passed in or
checked structurally rather than exactly.

### Errors

- missing required metadata keys;
- `TypeError` if the final report is not JSON-serializable.

### Dependencies

- public interfaces;
- `datetime`;
- `json.dumps()` for final validation.

### Variation

The result payload varies by evaluation type. The stable envelope should
change slowly and can later receive an explicit schema version.

### Tests

- report contains all stable identity fields;
- scenario metadata is preserved unchanged;
- arbitrary JSON-safe result payload is preserved;
- paths are converted to strings;
- NumPy arrays in the JSON payload fail or are explicitly converted;
- building a report performs no filesystem write.

---

## 9. Write the JSON Report

### Function question

How does the engine persist a previously validated report consistently?

### Proposed function

```python
def write_json_report(
    *,
    report: dict,
    output_path: Path,
) -> None:
    ...
```

### Responsibility

Serialize the report as human-readable UTF-8 JSON at the selected path.

### Owner

Engine, `evaluation/reporting.py`.

### Input

- JSON-safe report dictionary;
- final report path.

### Output

`None`.

### Preconditions

- parent evaluation output directory exists;
- report construction/validation succeeded.

### Postconditions

- a readable JSON file exists at `output_path`;
- its decoded value equals the input report.

### Side effects

Writes a durable file.

### Errors

- serialization errors;
- permission, disk-space, and other filesystem errors.

### Dependencies

- Python `json` module;
- `pathlib.Path`.

### Variation

Atomic writes or schema versioning may be added later. Scenario code should not
implement its own JSON formatting.

### Tests

- write under `tmp_path` and read it back;
- Unicode metadata is preserved;
- non-serializable values fail clearly;
- no unrelated files or directories are created.

---

## 10. Orchestrate the Evaluation

### Function question

Which fixed component determines the execution order while delegating
scientific generation and evaluation-type calculations?

### Proposed function

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

### Responsibility

Coordinate the complete evaluation transaction:

1. resolve and validate the artifact;
2. invoke the selected scenario generator;
3. validate its batch;
4. load the approximator;
5. sample the posterior;
6. invoke the configured evaluation-type calculation/reporting behavior;
7. return the final report.

The runner owns sequence, not scientific formulas.

### Owner

Engine, `evaluation/runner.py`. This is the main public engine service used by
CLI dispatch.

### Input

- artifact reference;
- discovered model and evaluation specifications;
- engine runtime configuration;
- generic and scenario evaluation options.

### Output

A JSON-serializable report dictionary whose output files already exist.

### Preconditions

- discovery succeeded;
- CLI options were syntactically parsed;
- runtime configuration contains current portable roots.

### Postconditions

- either the evaluation completes and returns a report, or an exception clearly
  identifies the failed boundary;
- engine code has not interpreted TTA-specific values;
- durable outputs live under the configured results root.

### Side effects

- reads metadata and checkpoint files;
- executes researcher-provided trusted scenario code;
- performs potentially expensive posterior sampling;
- writes evaluation outputs.

### Errors

Errors from every delegated boundary remain possible. The runner should add
context only when doing so improves diagnosis and should preserve exception
causes.

### Dependencies

- artifact API;
- `EvaluationSpec.generate_batch`;
- batch validator;
- Keras/BayesFlow loader and sampler;
- evaluation-type metric/reporting behavior;
- reporting persistence.

### Variation

The sequence is fixed. Scenario generation varies through `EvaluationSpec`.
Recovery calculations are initially called directly; if a second evaluation
family later requires a different calculator, that proven variation can
motivate a new contract.

### Tests

- use fakes for scenario, approximator, metrics, and reporting;
- verify exact call order;
- invalid batch prevents model loading;
- model loading failure prevents sampling/reporting;
- sampling failure prevents report writing;
- successful result is returned unchanged from reporting;
- fake non-TTA scenario passes through without engine changes.

---

## 11. Parse Generic Scenario Options

### Function question

How does the engine safely translate repeated CLI `KEY=VALUE` syntax into
basic Python values without knowing scenario semantics?

### Proposed function

```python
def parse_scenario_options(
    entries: list[str],
) -> dict[str, str | int | float | bool]:
    ...
```

### Responsibility

- parse generic scalar syntax;
- reject malformed or duplicate entries;
- never execute the input;
- leave semantic validation to the selected scenario.

### Owner

Engine CLI support, initially `cli.py`.

### Input

Examples:

```python
[
    "trials_per_tta=15",
    "shuffle_tta=true",
]
```

### Output

```python
{
    "trials_per_tta": 15,
    "shuffle_tta": True,
}
```

### Preconditions

- argparse has collected raw strings.

### Postconditions

- every key is non-empty and unique;
- every value belongs to the supported scalar types;
- no user string has been executed as Python.

### Side effects

None.

### Errors

- `ValueError`: missing `=`;
- `ValueError`: empty key;
- `ValueError`: duplicate key;
- `ValueError`: unsupported value syntax if parsing is deliberately strict.

### Dependencies

Only standard-library parsing helpers. Never use `eval()`.

### Variation

Each scenario decides which keys are allowed and their valid ranges. The
generic parser only handles syntax and simple scalar conversion.

### Tests

- strings, integers, floats, and documented booleans parse;
- malformed and duplicate entries fail;
- dangerous Python-looking strings are treated as strings or rejected, never
  executed.

---

## 12. Select a Model-Bound Evaluation

### Function question

How does generic CLI dispatch select a scenario and produce a useful error when
it is unavailable?

### Proposed function

```python
def require_evaluation(
    *,
    evaluations: dict[tuple[str, str], EvaluationSpec],
    model_name: str,
    evaluation_name: str,
) -> EvaluationSpec:
    ...
```

### Responsibility

Look up `(model_name, evaluation_name)` and, on failure, list scenarios
available for that model.

### Owner

Engine CLI support, initially `cli.py` alongside `require_model()` and
`require_workflow()`.

### Input

- evaluation registry;
- artifact model name;
- user-requested scenario name.

### Output

The selected `EvaluationSpec`.

### Preconditions

- discovery has completed;
- registry duplicate validation has already succeeded.

### Postconditions

- returned scenario is registered for the requested model.

### Side effects

None.

### Errors

`ValueError` containing:

- requested scenario;
- model name;
- available scenarios for that model, or `none`.

### Dependencies

- `EvaluationSpec`;
- plugin registry structure.

### Variation

Available capabilities vary through researcher plugins. Lookup policy remains
fixed.

### Tests

- exact pair resolves;
- same scenario name for another model does not resolve;
- error lists only scenarios belonging to the requested model;
- empty registry reports `none`.

---

## 13. Functions Deliberately Excluded From This Category

The following functions are important, but they are not engine-wide fixed
behaviors.

### Recovery-specific reusable behavior

```python
compute_recovery_metrics(...)
save_recovery_plot(...)
save_recovery_arrays(...)
```

They apply to parameter recovery but not necessarily to posterior predictive
checks, classification evaluations, or every future evaluation family.

### Scenario-specific scientific behavior

```python
sample_dc_parameters(...)
simulate_dc_trial(...)
generate_dc_mixed_tta(...)
```

They belong to the researcher workspace because they contain scientific model
knowledge.

### Discovery implementation details

```python
_validate_evaluations(...)
_collect_evaluations_from_module(...)
```

These are engine-fixed plugin mechanisms, but they belong to the capability
discovery work rather than evaluation execution. Design them with the same
template when you reach that step.

---

## 14. Recommended Implementation Order

Do not implement these functions in the order in which the final pipeline runs.
Implement them from pure and inexpensive to effectful and integrated:

1. `validate_evaluation_batch()`;
2. `build_evaluation_report_envelope()`;
3. `parse_scenario_options()`;
4. `require_evaluation()`;
5. `create_evaluation_output_dir()`;
6. `write_json_report()`;
7. `resolve_evaluation_artifact()`;
8. `load_artifact_approximator()`;
9. `sample_posterior()`;
10. `run_evaluation()`.

This order gives fast unit-test feedback before working with files, plugins, or
large neural networks.

---

## 15. Final Modular-Thinking Check

Before implementing any proposed function, ask:

- Does its responsibility contain one coherent reason to change?
- Does its name describe an outcome rather than implementation details?
- Are its inputs the minimum information it needs?
- Is its output useful without exposing unnecessary internals?
- Are shapes, keys, and types explicit?
- Are side effects visible in its contract?
- Are errors detected by the layer with enough knowledge to explain them?
- Can it be tested without running the full evaluation pipeline?
- Does it contain any scientific term the engine should not understand?
- Am I creating a helper because it clarifies a boundary, or merely because I
  can split three lines into another function?

If the last question has no clear architectural answer, keep the code inline
and preserve the conceptual contract in the runner test instead.

