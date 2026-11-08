
Let's think in term of questions, functions and behaviors.

In this attempt, I used the following mindset:

It is better to see each question and answer in term series of functions. Therefore, we need to extract the atomic functions and then try to categorized them into modules. In other words, we have to think in term of a set of questions and their relations. Some questions should be answered in the same way regardless of particular evaluation and some depend on a particular evaluation. This is the fundamental category we have to identify to handle all the questions that should be answered to implement an evaluation pipeline.



The most fundamental questions is What behaviors remained fixed during evaluation and what dependes on a specific scientific scenario.



## Fixed Behaviors 
- Retrieved the save trained model 
- Plot recovery 
- Calculate statiscs of posterior samples, std, mean, coverage and confidence interval. 
- Save the statiscs of posterior samples
- Calculate metrics such as mse and absolute error using posterior and ground truth 
- Save metrics data 
- Run the whole pipeline of evaluation from generating true parameters and validation data to posterior samples and metric calculation and visualizaiotn.
def make_model_spec(name="test_model"):
    return ModelSpec(
        name=name,
        workflow="builtin",
        simulator_factory=lambda: object(),
        adapter_factory=lambda: object(),
        family="test",
        par_names=["a", "b"],
        version="1.0",
    )

## Dependent Behaviors 

- Generate Validation Data -> generate true parameters and observations  
- A scientific scenario might introduce new statistics for posteriors 
- A scientific approach might introduce new metrics of evaluations.
- A scientific approach might introduce new visualization.


Another approach to categorize behaviors is as the following:


## Engine-wide fixed 

- Artificat_id resolution 
- Load model 

## Recovery-specific reussable


## Scenario-specific 




## Data structure contract 

In the following, we describe the data structures that cross the boundaries of
the engine-wide fixed evaluation pipeline. Defining only a Python type is not
enough. For each structure, we must define:

- its semantic meaning;
- its concrete Python representation;
- the meaning of its keys and dimensions;
- which module creates it;
- which module consumes it;
- its invariants;
- whether it is mutable or persistent;
- what errors should be raised when its contract is violated.

The data flow is:

```text
ArtifactReference
        ↓ resolve
ArtifactMetadata + WorkflowContext
        ↓ select capability
ModelSpec + EvaluationSpec
        ↓ scenario generation
EvaluationBatch
        ↓ posterior sampling
PosteriorSamples
        ↓ evaluation-type calculation
ResultPayload
        ↓ reporting
EvaluationReport + OutputPaths
```

The structures above do not all need to be dataclasses. A dataclass is useful
when a group of fields forms one stable concept. A plain dictionary is useful
when keys are dynamic or the object must be serialized. A string remains a
string when it represents only one value.

### Dimension symbols

We should name dimensions before discussing arrays:

| Symbol | Meaning |
|---|---|
| `N` | Number of independent evaluation datasets |
| `T` | Number of trials inside one dataset |
| `F` | Number of observed features per trial |
| `C` | Number of contextual features per trial |
| `P` | Number of inferred parameters |
| `S` | Number of posterior draws for each dataset |

Dataset and trial are different concepts. One evaluation dataset may contain
many trials, but the posterior network normally infers one parameter vector for
the complete dataset.

---

### 1. `ArtifactReference`

#### Semantic meaning

A user-facing identifier for an existing trained artifact. It may be:

- the canonical artifact ID; or
- an alias such as `run-001`.

#### Python representation

```python
ArtifactReference = str
```

This does not require a dataclass because it contains one value.

#### Created by

CLI parsing or another engine caller.

#### Consumed by

`resolve_artifact_ref()` or `resolve_evaluation_artifact()`.

#### Invariants

- It is a non-empty string.
- It is not yet assumed to be a canonical artifact ID.
- Filesystem paths must not be constructed directly from it until resolution
  establishes the canonical artifact metadata.

#### Failure

If neither an exact artifact ID nor an alias resolves, artifact resolution
raises `FileNotFoundError`.

---

### 2. `ArtifactMetadata`

#### Semantic meaning

Persistent descriptive information about a trained artifact. It identifies the
model and records training history and aliases.

#### Python representation

```python
ArtifactMetadata = dict[str, Any]
```

Expected engine-relevant fields include:

```python
{
    "artifact_id": str,
    "alias": str | None,
    "spec_name": str,
    "workflow": str,
    "status": str,
    "created_at": str,
    "updated_at": str,
    "training": dict[str, Any],
}
```

Legacy metadata may also contain absolute path strings. Those strings are
descriptive historical data and must not be authoritative for current runtime
path resolution.

#### Created by

The training/resume workflow and stored as JSON.

#### Consumed by

Artifact listing, resume, recovery, and generic evaluation runners.

#### Invariants

- `artifact_id` is the canonical ID.
- `spec_name` identifies the model that produced the artifact.
- The metadata file is located through the currently configured checkpoint
  root.
- Operational checkpoint/results/log paths are reconstructed from current
  configuration, not stale stored absolute paths.

#### Mutability and persistence

It is a mutable dictionary in memory and persistent JSON on disk. Some legacy
metadata may be updated to add an alias.

#### Failure

Missing files, malformed JSON, missing required keys, or an artifact/model
mismatch must fail before loading the neural network.

---

### 3. Runtime `config`

#### Semantic meaning

The current execution environment for the engine. It tells the application
where the mounted workspace currently lives and which device/options to use.

#### Python representation

```python
RuntimeConfig = dict[str, Any]
```

The evaluation path depends on fields such as:

```python
{
    "device": device,
    "paths": {
        "checkpoints": str,
        "results": str,
        "logs": str,
    },
}
```

#### Created by

`build_runtime_config()` from defaults, current working directory, and CLI
overrides.

#### Consumed by

Artifact resolution, context construction, model loading, and reporting.

#### Invariants

- `paths` exists and is a dictionary.
- Every required path value can be converted to `Path`.
- Paths describe the current runtime workspace, including a Docker-mounted
  workspace.
- Scenario-specific facts such as `trials_per_tta` should not be inserted into
  generic runtime configuration.

#### Failure

Missing required path keys should raise `ValueError` before filesystem work.

---

### 4. `WorkflowContext`

#### Semantic meaning

The portable, resolved runtime locations for one artifact.

#### Python representation

```python
@dataclass(frozen=True)
class WorkflowContext:
    artifact_id: str
    checkpoint_path: Path
    metadata_path: Path
    results_dir: Path
    logs_dir: Path
    device: Any
```

#### Created by

`resolve_context_from_artifact_id()` for an existing artifact or
`build_workflow_context()` for a new artifact.

#### Consumed by

The evaluation runner and reporting layer.

#### Invariants

- `artifact_id` equals the canonical metadata artifact ID.
- `checkpoint_path` and `metadata_path` use the configured checkpoint root.
- `results_dir` uses the configured results root plus the artifact ID.
- `logs_dir` uses the configured logs root plus the artifact ID.
- Constructing the context does not imply that every directory should be
  created.

#### Mutability

The dataclass is frozen. The `Path` objects are immutable values, but the files
to which they refer may change.

#### Failure

The evaluation runner raises `FileNotFoundError` when the resolved checkpoint
file is absent.

---

### 5. `ModelSpec`

#### Semantic meaning

A discovered description of one scientific model and the factories needed by
the application.

#### Relevant Python representation

```python
@dataclass
class ModelSpec:
    name: str
    workflow: str
    simulator_factory: Callable
    adapter_factory: Callable
    family: str
    par_names: list[str]
    version: str
    description: str = ""
```

#### Created by

The researcher model plugin and collected by plugin discovery.

#### Consumed by

CLI selection, artifact/model validation, scenario generation, and recovery
metrics.

#### Evaluation-relevant invariants

- `name` matches `ArtifactMetadata["spec_name"]`.
- `par_names` is non-empty and contains unique strings.
- `par_names` defines both the expected true-parameter keys and the parameter
  order when a posterior is represented by one combined array.

#### Ownership

The interface is engine-owned; the scientific values are researcher-owned.

---

### 6. `EvaluationSpec`

#### Semantic meaning

A registered capability saying that one named scientific evaluation scenario
can generate evaluation data for one particular model.

#### Python representation

```python
@dataclass(frozen=True)
class EvaluationSpec:
    name: str
    model_name: str
    generate_batch: GenerateEvaluationBatch
    description: str = ""
```

The callable contract is:

```python
GenerateEvaluationBatch = Callable[
    [ModelSpec, dict[str, Any]],
    EvaluationBatch,
]
```

#### Created by

The researcher evaluation plugin.

#### Consumed by

Plugin discovery, CLI scenario selection, and `run_evaluation()`.

#### Invariants

- Registry identity is `(model_name, name)`.
- `generate_batch` is callable.
- `model_name` equals the selected `ModelSpec.name` before invocation.
- The generator returns an `EvaluationBatch` but does not load checkpoints,
  sample posteriors, calculate generic metrics, or write reports.

---

### 7. `EvaluationBatch`

#### Semantic meaning

The complete scientific evaluation input generated by one scenario before
posterior sampling. The whole object is evaluation data. Its
`posterior_conditions` field is only the part used to condition posterior
sampling.

#### Python representation

```python
@dataclass(frozen=True)
class EvaluationBatch:
    posterior_conditions: dict[str, Any]
    true_parameters: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### `posterior_conditions`

Semantic meaning:

> The exact batched dictionary accepted by
> `approximator.sample(conditions=...)`.

Generic type:

```python
dict[str, Any]
```

We cannot type every value as one `(N, T, C)` array because different
conditions have different ranks. For the current mixed-TTA models, the concrete
structure is expected to resemble:

```python
{
    "x": np.ndarray(shape=(N, T, F)),
    "tta_per_trial": np.ndarray(shape=(N, T, C)),
    "number_of_trials": np.ndarray(shape=(N,)),
}
```

In the active simulator:

```text
F = 1  # one response/choice-RT value per trial
C = 1  # one TTA value per trial
```

Therefore, a concrete batch may have:

```text
x.shape                == (N, T, 1)
tta_per_trial.shape    == (N, T, 1)
number_of_trials.shape == (N,)
```

The generic engine knows that the batch describes `N` datasets, but only the
model/scenario compatibility test knows the exact keys and inner dimensions.

#### `true_parameters`

Semantic meaning:

> Known parameter values used to generate every evaluation dataset.

More precise runtime type:

```python
dict[str, np.ndarray]
```

For each parameter:

```text
true_parameters[parameter_name].shape == (N,)
```

An array shaped `(N, 1)` may be normalized to `(N,)`, but it still represents
exactly one scalar parameter value per dataset. An array `(N, K)` with `K > 1`
violates this initial scalar-parameter contract.

Keys must satisfy:

```python
set(batch.true_parameters) == set(model_spec.par_names)
```

Dataset index establishes correspondence:

```text
posterior_conditions["x"][i]
true_parameters["b0"][i]
true_parameters["alpha"][i]
```

All describe the same independently simulated dataset `i`.

#### `metadata`

Semantic meaning:

> JSON-safe scientific facts describing how the scenario generated the batch.

Type:

```python
dict[str, JSON-compatible value]
```

Example:

```python
{
    "interpretation": "out_of_distribution_stress_test",
    "condition_values": [2.5, 3.0, 3.5, 4.0],
    "trials_per_tta": 15,
    "n_trials": 60,
    "shuffle_tta": True,
    "seed": 2026,
}
```

It has no required dataset dimension. NumPy arrays must be converted to lists
before JSON serialization.

#### Created by

`EvaluationSpec.generate_batch(model_spec, options)`.

#### Consumed by

Batch validation, posterior sampling, evaluation-type metrics, and reporting.

#### Generic invariants

- It is an `EvaluationBatch` instance.
- `posterior_conditions` is a non-empty dictionary.
- Target keys exactly match model parameters.
- Every target has the same non-zero `N`.
- Metadata is a JSON-serializable dictionary.
- After validation, downstream code treats the batch as immutable.

#### Mutability qualification

`frozen=True` prevents replacing the three attributes, but dictionaries and
NumPy arrays inside the object remain mutable. Immutability is therefore a
downstream convention, not deep enforcement.

---

### 8. `PosteriorSamples`

#### Semantic meaning

Samples from the learned posterior distribution for every evaluation dataset.

#### Python representation

For the installed BayesFlow API, normally:

```python
PosteriorSamples = dict[str, np.ndarray]
```

Parameter-keyed representation:

```text
posterior_samples[name].shape == (N, S)
```

Conceptually, all parameters can be represented as one array:

```text
(N, S, P)
```

where parameter axis order follows `model_spec.par_names`.

#### Created by

```python
approximator.sample(
    conditions=batch.posterior_conditions,
    num_samples=S,
)
```

#### Consumed by

Recovery metrics, diagnostics, plots, and array persistence.

#### Invariants

- Dataset dimension equals the validated `N` from `EvaluationBatch`.
- Posterior draw dimension equals requested `S`.
- Parameter keys or parameter dimension match `model_spec.par_names`.
- Trial dimension normally does not appear in a posterior over global model
  parameters; trials were summarized as conditions for one dataset posterior.

#### Failure

Shape mismatch should fail before calculating numerical metrics, because
averaging the wrong axis can produce plausible but invalid results.

---

### 9. `ResultPayload`

#### Semantic meaning

Evaluation-type-specific, JSON-safe calculated results. For parameter recovery,
this may contain bias, RMSE, Pearson correlation, posterior uncertainty, and
coverage. A future evaluation type may provide different keys.

#### Python representation

```python
ResultPayload = dict[str, Any]
```

Example recovery payload:

```python
{
    "metrics": {
        "b0": {
            "bias": 0.02,
            "rmse": 0.13,
            "pearson": 0.91,
            "coverage_95": 0.95,
        }
    }
}
```

#### Created by

The selected reusable evaluation-type calculation.

#### Consumed by

Generic report-envelope construction.

#### Invariants

- It is JSON-serializable.
- It contains calculated results, not large raw posterior arrays.
- NumPy arrays needed for later analysis are stored separately in NPZ output.

This payload is variable, while the report envelope containing it is fixed.

---

### 10. `OutputPaths`

#### Semantic meaning

Names and locations of durable evaluation outputs.

#### Python representation

```python
OutputPaths = dict[str, Path]
```

Example:

```python
{
    "report": output_dir / "report.json",
    "arrays": output_dir / "evaluation.npz",
    "recovery_plot": output_dir / "recovery.pdf",
}
```

#### Created by

The engine reporting layer after it creates:

```text
results/<artifact-id>/evaluations/<evaluation-name>/
```

#### Consumed by

Persistence functions and report-envelope construction.

#### Invariants

- Every path lies beneath the configured artifact results directory.
- Paths are converted to strings before JSON serialization.
- Creating these paths does not create checkpoint or log directories.

The mapping may vary by evaluation type because not every evaluation produces
the same plots, but the output-root policy is engine-fixed.

---

### 11. `EvaluationReport`

#### Semantic meaning

The final JSON-safe description of one completed evaluation execution.

#### Python representation

```python
EvaluationReport = dict[str, Any]
```

Stable engine envelope:

```python
{
    "artifact_id": str,
    "alias": str | None,
    "spec_name": str,
    "evaluation_name": str,
    "evaluation_description": str,
    "checkpoint_path": str,
    "created_at": str,
    "n_datasets": int,
    "n_posterior_samples": int,
    "scenario_metadata": dict,
    "results": ResultPayload,
    "outputs": dict[str, str],
}
```

#### Created by

`build_evaluation_report_envelope()` after calculations and output paths are
known.

#### Consumed by

`write_json_report()`, the CLI summary, and researchers inspecting results.

#### Invariants

- The complete dictionary is JSON-serializable.
- Engine identity and execution facts use stable top-level fields.
- Scientific scenario facts remain under `scenario_metadata`.
- Evaluation-type results remain under `results`.
- It contains paths as strings, not `Path` objects.
- Building the dictionary has no filesystem side effect; writing it is a
  separate operation.

---

### Contract relationships that must always hold

The engine-wide validator and runner should preserve these cross-structure
relationships:

```python
metadata["spec_name"] == model_spec.name
evaluation_spec.model_name == model_spec.name
context.artifact_id == metadata["artifact_id"]
set(batch.true_parameters) == set(model_spec.par_names)
n_datasets == len(batch.true_parameters[name])
posterior_samples[name].shape[0] == n_datasets
posterior_samples[name].shape[1] == n_posterior_samples
report["artifact_id"] == metadata["artifact_id"]
report["spec_name"] == model_spec.name
```

These relationships explain why data-structure design must happen before
function implementation. Each engine-fixed function either constructs,
validates, transforms, or persists one of these contracts.







