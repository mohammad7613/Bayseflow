# Scientific and Recovery Evaluation Behaviors

## Purpose and boundary

Implementation update (23 October 2026): the first four functions below are now
implemented and numerically tested in `evaluation/metrics.py`. Engine integration
and the persistence/plotting work remain separate tasks. The signatures below
describe the implemented numerical API.

This document covers the reusable scientific layer that interprets posterior
draws as a **parameter-recovery evaluation**. It is added on top of the
engine-fixed behaviors described in `docs/EngineFixedEvaluationBehaviors.md`.

The engine knows how to resolve an artifact, validate a batch, load a
checkpoint, sample, and write outputs. This layer knows what recovery means:
how to normalize posterior draws, compare them with true parameters, compute
summary statistics, and produce recovery artifacts.

It must not know that a scenario is mixed-TTA, DDM, pedestrian, or trial-wise.
Those details belong to the scenario plugin.

The dependency direction is:

```text
engine-fixed runner
        -> EvaluationBatch and posterior draws
        -> recovery functions in this document
        -> JSON/array/figure payloads
```

Do not add an `evaluator` field to `EvaluationSpec` merely to implement this
layer. `EvaluationSpec.generate_batch(model_spec, options)` generates data;
recovery functions consume the resulting batch and posterior samples.

## Data contract

Use `N` for datasets, `S` for posterior draws, and `P` for parameters.

- `batch.true_parameters[name]`: array with one value per dataset, normally
  shape `(N,)`; the metric function also explicitly accepts `(N, 1)`.
- `posterior_samples[name]`: posterior draws for one parameter, normalized to
  shape `(N, S)`.
- `parameter_names`: the model parameter names and therefore the required keys
  in both dictionaries.
- metric values in the report: JSON-safe Python scalars, not NumPy scalars.
- per-dataset summaries may remain NumPy arrays until they are written to NPZ;
  JSON reports must convert them to lists or scalars.

The normalization rule must be explicit. Do not silently guess between an
array shaped `(N, S)` and `(S, N)` when both dimensions happen to be equal.
Require explicit expected dataset and draw counts. A square `(N, N)` array is
valid under the declared convention; reject unsupported layouts rather than
guessing orientation.

## Function inventory and implementation order

Implement and test in this order:

```text
normalize_posterior_draws()
        ↓
summarize_posterior_draws()
        ↓
compute_recovery_metrics()
        ↓
build_recovery_result_payload()
        ↓
save_recovery_plot() / save_evaluation_arrays()
```

The fixed engine should call these only after `sample_posterior()` succeeds.

## 1. Normalize posterior draws

### Function question

How do all recovery metrics receive the same `(datasets, draws)` shape?

### Implemented function

```python
def normalize_posterior_draws(
    *,
    posterior_samples: dict[str, Any],
    parameter_names: list[str],
    n_datasets: int,
    n_posterior_samples: int,
) -> dict[str, np.ndarray]:
    ...
```

### Responsibility

Convert the supported BayesFlow output representation into one array per
parameter with shape `(N, S)`. Check parameter keys, dataset count, draw count,
and finite dimensionality. This function does not compute scientific metrics.

### Owner

Reusable recovery module, initially `bayesflow_models/evaluation/metrics.py`.

### Preconditions and postconditions

- Preconditions: posterior samples exist; every requested parameter is present;
  `n_datasets > 0`; each value has a dataset and draw axis.
- Postconditions: exactly one array exists per requested parameter; every array
  has first dimension `n_datasets` and a positive second dimension.

### Errors, side effects, variation

Raise `KeyError` for missing parameters, `TypeError` for invalid count types or
non-real values, and `ValueError` for extra parameters, wrong shapes/counts, or
non-finite values. Accept `(N, S)` and `(N, S, 1)` for each scalar parameter.
The function is pure and has no filesystem side effects. Combined arrays need
explicit splitting into parameter names before calling this function.

### Tests

Test dict input, missing parameter, wrong dataset dimension, one-draw input,
zero draws, and a square array with distinguishable values. Assert the exact
normalized shape and verify that values are not transposed.

## 2. Summarize posterior draws

### Function question

How do we derive posterior mean, spread, and credible bounds for each dataset?

### Implemented function

```python
def summarize_posterior_draws(
    *,
    normalized_draws: dict[str, np.ndarray],
    lower: float = 0.025,
    upper: float = 0.975,
) -> dict[str, dict[str, np.ndarray]]:
    ...
```

### Responsibility

For each parameter compute `mean`, `std`, `lower`, and `upper` across the draw
axis only. Validate `0 <= lower < upper <= 1`.

### Owner

Recovery module, `metrics.py`.

### Preconditions and postconditions

- Preconditions: inputs are normalized `(N, S)` arrays; each has `S > 0`.
- Postconditions: each statistic has shape `(N,)`, and no dataset/draw axis is
  accidentally collapsed.

### Errors and tests

Raise `ValueError` for invalid quantiles or non-2D arrays. Use a deterministic
small array whose expected mean and quantiles can be calculated by hand.

## 3. Compute recovery metrics

### Function question

How accurately do posterior summaries recover known simulation parameters?

### Implemented function

```python
def compute_recovery_metrics(
    *,
    normalized_draws: dict[str, np.ndarray],
    true_parameters: dict[str, Any],
) -> tuple[dict[str, dict[str, float | None]],
           dict[str, dict[str, np.ndarray]]]:
    ...
```

### Responsibility

Consume already normalized draws, summarize them, then calculate per-parameter aggregate
metrics such as bias, RMSE, Pearson correlation, posterior spread, and 95%
coverage. Keep formulas documented and stable. A constant target or estimate
has undefined Pearson correlation; return `None`, not `NaN`, in JSON output.

### Owner

Reusable recovery module. The engine invokes it; a scenario does not reimplement
these formulas.

### Preconditions and postconditions

- Preconditions: target names exactly match the normalized draw keys; each target has
  `N` values; posterior arrays normalize to `(N, S)`.
- Postconditions: metrics contain JSON-safe scalar values; summaries retain one
  value per dataset; no input arrays are mutated.

### Tests

Use perfect draws (zero bias/RMSE, full coverage), shifted draws (known bias and
RMSE), an out-of-interval target (reduced coverage), and constant arrays
(Pearson `None`). Test missing and extra target keys.

## 4. Build the recovery result payload

### Function question

How are scientific results separated from the engine report envelope?

### Implemented function

```python
def build_recovery_result_payload(
    *,
    metrics: dict,
) -> dict:
    ...
```

### Responsibility

Return `{"metrics": ...}` containing aggregate scalar metrics. Per-dataset
summaries and draws stay outside this JSON payload; save them in NPZ and
reference their files from the fixed-engine report envelope.

### Owner

Implemented in `metrics.py` alongside the pure numerical operations. It performs
no persistence; the engine/reporting layer owns file writes.

### Preconditions and postconditions

- Preconditions: metrics came from the recovery functions and contain finite
  real scalars or `None` for undefined correlation.
- Postconditions: `json.dumps(payload)` succeeds; no `Path`, NumPy scalar, or
  ndarray remains unconverted.

### Tests

Run `json.dumps(payload, allow_nan=False)` and assert that metrics are preserved
exactly. Test conversion of NumPy scalars and rejection of arrays/non-finite
values. Test that undefined correlation becomes JSON `null`.

## 5. Save recovery figures and arrays

### Function questions

How can a user inspect recovery visually and reproduce the numerical analysis?

### Proposed functions

```python
def save_recovery_plot(*, posterior_samples, true_parameters,
                       parameter_names, output_path: Path,
                       visualize: bool = False) -> None: ...

def save_evaluation_arrays(*, output_path: Path, posterior_samples,
                           true_parameters, posterior_summary) -> None: ...
```

### Responsibility and owner

`save_recovery_plot` delegates to BayesFlow’s recovery diagnostic and closes
the figure when not displaying it. `save_evaluation_arrays` writes a stable
NPZ naming convention. Both belong to reusable recovery reporting.

### Preconditions, postconditions, side effects

Output parents must be writable. On success the requested file exists and can
be read back. These functions intentionally perform filesystem I/O; they must
not mutate the batch or rerun sampling.

### Tests

Monkeypatch the BayesFlow plotting function for a fast unit test, verify
`savefig` and `close`, and test NPZ round-trip keys and shapes.

## 6. Connect this layer to the fixed engine

After the functions above pass unit tests, the fixed runner may perform:

```python
batch = evaluation_spec.generate_batch(model_spec, options)
n = validate_evaluation_batch(batch=batch, model_spec=model_spec)
approximator = load_artifact_approximator(context=ctx)
draws = sample_posterior(approximator=approximator,
                         posterior_conditions=batch.posterior_conditions,
                         n_posterior_samples=n_samples)
metrics, summaries = compute_recovery_metrics(
    posterior_samples=draws,
    true_parameters=batch.true_parameters,
    parameter_names=model_spec.par_names,
    n_datasets=n,
)
```

Only this last composition belongs in the runner. Recovery formulas and
plotting remain replaceable. Add an integration test with fake model loading
and fake posterior draws; do not require a real neural-network checkpoint for
unit tests.

## What this layer must not contain

- TTA condition sequences or shuffling;
- DDM simulation functions;
- trial counts and jitter rules;
- model-specific parameter priors;
- CLI parsing;
- artifact path reconstruction.

Those are either scenario-owned or engine-fixed responsibilities.
