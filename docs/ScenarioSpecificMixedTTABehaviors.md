# Scenario-Specific Mixed-TTA Behaviors

## Purpose and boundary

This document describes the researcher-owned scenario plugin needed to
reconstruct the current mixed-TTA evaluation. It is implemented **in addition
to** the engine-fixed and reusable recovery layers.

The scenario answers the scientific question:

> Does a posterior network trained with a fixed condition per simulated trial
> recover parameters when each validation dataset contains trials with mixed
> TTA conditions?

The scenario owns simulation semantics. The engine must remain unaware of TTA,
trial-wise jitter, DDM equations, and the names of model-specific condition
variables.

## Boundary and data flow

```text
scenario options
      ↓
validate mixed-TTA options
      ↓
sample true parameters
      ↓
create per-dataset/per-trial TTA schedule
      ↓
simulate observations with trial-wise conditions
      ↓
adapt observations to the trained model input schema
      ↓
return EvaluationBatch
      ↓
engine samples posterior and recovery layer scores it
```

The plugin is registered through an `EvaluationSpec`:

```python
EVALUATIONS = [
    EvaluationSpec(
        name="mixed_tta",
        model_name="model_DC",
        generate_batch=generate_mixed_tta_batch,
        description="Trial-wise mixed-TTA parameter recovery",
    )
]
```

The same scenario can be registered separately for models with different input
adapters, or it can dispatch through a model-specific simulator registry. Do
not put the mixed-TTA evaluator into `EvaluationSpec`; its callable generates
the batch only.

## Scenario data contract

Use `N` datasets, `T` trials per dataset, and `F` observed features.

- true parameters: `{parameter_name: np.ndarray}` with one value per dataset;
- posterior conditions: the exact keys and shapes expected by the trained
  approximator, commonly `x: (N, T, F)` and `tta_per_trial: (N, T, 1)`;
- optional `number_of_trials`: `(N,)` or the shape required by the model’s
  adapter;
- metadata: JSON-safe scalar options and summaries of the generated schedule;
  do not store full large arrays in metadata.

The scenario must document whether the network was trained with TTA included as
a separate condition key or concatenated into `x`. This is an adapter contract,
not something the engine should infer.

## Function inventory and implementation order

```text
validate_mixed_tta_options()
        ↓
sample_mixed_tta_parameters()
        ↓
build_mixed_tta_schedule()
        ↓
simulate_mixed_tta_observations()
        ↓
adapt_mixed_tta_conditions()
        ↓
generate_mixed_tta_batch()
```

Implement each function independently with deterministic tests before
registering the final `EvaluationSpec`.

## 1. Validate scenario options

### Function question

Which options are required to make a mixed-TTA experiment scientifically
well-defined and reproducible?

### Proposed function

```python
def validate_mixed_tta_options(options: dict) -> dict:
    ...
```

### Responsibility

Validate and normalize `n_datasets`, `n_trials` or `number_of_trials`,
`trials_per_tta`, allowed TTA values, jitter, random seed, and optional
`shuffle_tta`. Return a new normalized options dictionary.

### Owner

Researcher scenario plugin, for example
`examples/article_workspace/plugins/mixed_tta_evaluation.py`.

### Preconditions and postconditions

- Preconditions: options is a mapping and required values are present or have
  documented defaults.
- Postconditions: counts are positive integers; TTA values are finite; the
  number of schedule entries is compatible with `n_trials`; seed is explicit.

### Errors and tests

Raise `ValueError` for zero/negative counts, impossible combinations, invalid
TTA values, or negative jitter. Test defaults and each invalid boundary.

## 2. Generate true parameters

### Function question

How are known ground-truth parameter values sampled for recovery?

### Proposed function

```python
def sample_mixed_tta_parameters(
    *, model_spec: ModelSpec, n_datasets: int,
    rng: np.random.Generator, options: dict,
) -> dict[str, np.ndarray]:
    ...
```

### Responsibility

Use the model’s scientific prior or parameter sampler to produce exactly one
value per dataset for every `model_spec.par_names`. Keep this separate from
observation simulation so the ground truth is inspectable.

### Owner

Scenario/model plugin. Reuse the model’s existing prior definitions rather than
copying them into the engine.

### Preconditions and postconditions

- Preconditions: model spec has a known parameter schema and the RNG is seeded.
- Postconditions: keys exactly equal `model_spec.par_names`; every value has
  shape `(N,)`; repeated seeds reproduce the same result.

### Tests

Assert keys, shapes, finite values, seed reproducibility, and that a different
seed changes at least one generated parameter set.

## 3. Build the trial-wise TTA schedule

### Function question

How does each dataset receive a mixture of TTA conditions across trials?

### Proposed function

```python
def build_mixed_tta_schedule(
    *, n_datasets: int, n_trials: int,
    tta_values: np.ndarray, trials_per_tta: int | None,
    rng: np.random.Generator, shuffle: bool,
    jitter: float = 0.0,
) -> np.ndarray:
    ...
```

### Responsibility

Create the base condition sequence, repeat each condition according to the
scenario policy, optionally shuffle per dataset, and apply bounded jitter if
that is part of the original experiment. Return shape `(N, T, 1)`.

### Owner

Scenario plugin.

### Preconditions and postconditions

- Preconditions: all TTA values are finite and positive; the requested schedule
  fits exactly or has an explicit remainder policy.
- Postconditions: output has `(N, T, 1)`; every condition belongs to the allowed
  range unless jitter explicitly permits an offset; the same seed reproduces it.

### Tests

Test no-shuffle order, shuffle reproducibility, counts per condition, shape,
and invalid remainder handling. Keep a regression fixture for the old
mixed-TTA sequence.

## 4. Simulate observations

### Function question

How are observations generated when the condition can vary by trial?

### Proposed function

```python
def simulate_mixed_tta_observations(
    *, model_spec: ModelSpec,
    true_parameters: dict[str, np.ndarray],
    tta_schedule: np.ndarray,
    options: dict,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    ...
```

### Responsibility

Call the model-specific trial-wise simulator with each dataset’s parameter
vector and each trial’s TTA value. Preserve observation and condition arrays;
do not hide the schedule in global state.

### Owner

Scenario/model plugin, using code such as
`DDM_DC_Pedestrain_TrialWise.py`.

### Preconditions and postconditions

- Preconditions: parameter keys and schedule dimensions are valid; simulator
  accepts trial-wise conditions.
- Postconditions: observations contain exactly `N` datasets and `T` trials;
  no observation is generated with a condition from another trial; all output
  keys are documented.

### Tests

Use a deterministic fake simulator that records its condition argument. Assert
that every trial receives the expected schedule entry and that output shapes
are `(N, T, F)` (or the model’s documented shape).

## 5. Adapt conditions to the trained model schema

### Function question

How do raw simulator outputs become the exact dictionary expected by the
posterior network?

### Proposed function

```python
def adapt_mixed_tta_conditions(
    *, observations: dict[str, np.ndarray],
    tta_schedule: np.ndarray,
    model_spec: ModelSpec,
    options: dict,
) -> dict[str, np.ndarray]:
    ...
```

### Responsibility

Apply the same concatenation, broadcasting, renaming, and feature ordering used
during training. For example, preserve `tta_per_trial` separately or
concatenate it with `x` according to the model adapter.

### Owner

Scenario/model plugin. This is where the old TrialWise adapter logic belongs.

### Preconditions and postconditions

- Preconditions: observation and schedule dimensions agree; model input schema
  is known.
- Postconditions: returned keys and shapes match the trained approximator’s
  conditions exactly; no accidental dataset/trial axis transposition occurs.

### Tests

Assert exact keys, shapes, concatenated feature order, and broadcasting of
`number_of_trials`. Add a test that intentionally supplies mismatched shapes and
expects `ValueError`.

## 6. Generate the EvaluationBatch

### Function question

How are all scenario outputs packaged into the engine-neutral interface?

### Proposed function

```python
def generate_mixed_tta_batch(
    model_spec: ModelSpec, options: dict,
) -> EvaluationBatch:
    ...
```

### Responsibility

Create a local RNG, validate options, sample targets, build the schedule,
simulate observations, adapt conditions, and return `EvaluationBatch`.

### Owner

Researcher workspace plugin. The engine calls it but does not inspect its
scientific internals.

### Preconditions and postconditions

- Preconditions: `model_spec.name` is supported by this scenario and options
  have passed validation.
- Postconditions: the batch satisfies the generic `EvaluationBatch` contract;
  target keys match model parameters; metadata contains seed and scenario
  settings; no global RNG or filesystem state is changed.

### Tests

Run with a tiny deterministic fake model and assert the complete batch keys,
shapes, metadata, and seed reproducibility. Then run one integration test using
the real DDM trial-wise simulator with very small `N` and `T`.

## 7. Registration and CLI selection

Register the scenario in the plugin module and let unified discovery collect it
once:

```python
EVALUATIONS = [
    EvaluationSpec(
        name="mixed_tta",
        model_name="model_DC",
        generate_batch=generate_mixed_tta_batch,
    )
]
```

The registry key is `(model_name, evaluation_name)`. The CLI should select the
pair explicitly and pass scenario options through `options`; it should not
import or call the old `evaluate_mixed_tta_artifact_with_bf_recovery` function
directly after migration.

## 8. Compatibility reconstruction test

After unit tests pass, write one end-to-end regression test that compares the
new pipeline with the old mixed-TTA behavior for a tiny fixed seed:

1. generate a batch with the same parameter seed and TTA policy;
2. verify the old and new observation/condition shapes and schedule values;
3. feed identical fake posterior draws to both metric paths;
4. compare metrics within a documented tolerance;
5. verify the new report contains scenario metadata and output references.

This test proves scientific equivalence without putting mixed-TTA knowledge
into the engine.

## What remains outside the scenario

- artifact lookup and checkpoint paths;
- model loading and posterior sampling coordination;
- generic batch validation;
- recovery metric formulas;
- JSON serialization and output directory policy;
- CLI discovery internals.

Those responsibilities are provided by the fixed-engine and reusable recovery
documents. The mixed-TTA plugin supplies only the scientific data-generation
and model-input adaptation needed to produce an `EvaluationBatch`.
