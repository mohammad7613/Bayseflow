# Improving the cognitive developer experience

This roadmap is based on the current checkout reviewed on 2026-11-06. It
distinguishes working extension points from remaining work. The immediate goal
is a developer who can define a model, train it, resume it, and compare experimental
designs without editing package internals or depending on article-specific code.

The [developer guide](CognitiveDeveloperGuide.md) describes today's interface.
[Cleanup before pushing](CleaningBeforeGitPush.md) covers repository preparation.
These recommendations do not imply that the changes below have been implemented.

## 1. Fix the release blockers first

The baseline command was:

```bash
KERAS_BACKEND=torch MPLBACKEND=Agg uv run --no-sync python -m pytest \
  tests examples/cognitive_developer_workspace/tests -q
```

It produced **121 passed, 2 failed**, with 14 dependency deprecation warnings.
Separately, the active `model_DC1` homogeneous evaluation completed through the
real CLI after the trial-count shape correction; that verifies this path, not
every legacy command or installation mode.

| Priority | Current issue | Proposed change | Completion evidence |
| --- | --- | --- | --- |
| P0 | `test_run_evaluation_saves_results` still declares `generate_batch(model, options)`, while the engine calls `generate_batch(options)`. | Update the test and any obsolete docs to the actual public callback. Retain a real runner persistence test. | Test checks generation, sampling, metrics, arrays, and report with the supported signature. |
| P0 | CLI `mixed-recovery` imports an example helper that imports removed package scientific code. Its test patches the previous module path. | Prefer deprecating/removing the duplicate command in favor of registered `evaluate` scenarios; update help and tests together. If retaining it, route through the generic engine. | Advertised commands run from an installed wheel without importing repository `examples/`. |
| P0 | Fish training and smoke tooling use `model_DC`; evaluation registrations target `model_DC1`. | Choose one canonical end-to-end example, or register scientifically compatible scenarios for both. Update scripts, YAML, README and tests as a unit. | A fresh developer follows one documented sequence successfully, with no existing artifacts. |
| P0 | Built-in resume can retain an exhausted learning-rate schedule. `model_DC1` still selects `builtin`. | Expose and implement explicit optimizer/schedule continuation policies; reuse the example's demonstrated approach where appropriate. | Fresh-process resume increments steps and changes trainable weights under the declared policy. |
| P0 | Active scenario wrappers silently ignore several settings in the bundled YAML files. | Define per-scenario option schemas, reject unknown fields, and return fully resolved/applied settings. | Unsupported `dt`/`posterior_seed`/TTA settings fail clearly instead of appearing applied. |

Treat P0 items as the first development milestone before presenting the checkout
as a clean release. Separate scientific-equation changes from infrastructure fixes
so their effects remain reviewable.

## 2. Make model authoring a short, reliable workflow

### Add a small starter independent of the article

Provide a minimal Gaussian or similarly tractable workspace alongside the article
example. Include only model science, a simulator, a serializable adapter, model
registration, one evaluation generator, and a small integration check. A developer
should learn the package API before needing to understand DDM equations.

Consider a future `workspace init` command that copies this starter. It should
generate ordinary Python files the researcher owns, including a model contract
describing parameter order, observations, units, trial axes, and seed policy.
This command is proposed; it does not currently exist.

**Acceptance:** a developer can implement another model by editing workspace
science and registrations only, then run the same train/resume/evaluate commands.

### Add a preflight validation command

The engine currently checks leading dataset dimensions but cannot explain all
adapter-specific failures. A future `validate` command should:

1. Discover the requested model and evaluation pair.
2. Simulate a tiny batch and report raw/adapted keys, dtypes and shapes.
3. Initialize the adapter as training would, then adapt conditions without truths.
4. Check parameter names/order, sample output, finite values, and JSON metadata.
5. Repeat with more than one dataset to catch silent broadcasting problems.

For the recent failure, report the offending key and actual transformation:
`number_of_trials: (8, 1) -> (1, 8, 1), cannot broadcast to (8, 60, 1)`.
Suggest `(N, 1, 1)` only when the declared adapter schema warrants it, rather than
hard-coding a DDM-specific reshape into the generic engine.

**Acceptance:** incompatible inputs fail before a costly training/evaluation run,
with a key-specific explanation and a declared expected schema.

### Align the public contracts

Update `ResumeFn` and `RecoveryFn` in `interfaces.py` to match the runtime
`(spec, config)` callbacks. Keep the single-argument evaluation generator contract
consistent throughout examples, tests and documentation. Remove the obsolete
alternate `evaluation/runner.py` after checking consumers.

Maintain a small public import surface for `ModelSpec`, `Workflow`,
`EvaluationSpec`, `EvaluationBatch`, and `EvaluationResult`. Document what custom
workflows must persist for generic evaluation to consume their artifacts.

**Acceptance:** type checking and real callbacks agree; users need no knowledge
of internal module names to implement ordinary extensions.

## 3. Make experiments reproducible and configurable

### Give each source of randomness an explicit owner

`myscience.py` uses a module-level `default_rng(2023)` for priors and the global
`np.random` stream for trial noise. Mixed generation uses a separate local RNG,
but its prior still consumes the module-global generator. Active homogeneous
generation does not apply the YAML seed. Consequently the existing option named
`seed` does not describe the whole experiment.

Pass explicit RNGs to priors, design generation and trial kernels. Separate
training simulation, evaluation simulation, and posterior sampling seeds. Record
the resolved seed policy and relevant versions. Keep random-stream ownership
consistent when workers are introduced; do not rely on import order or repeated
calls resetting a module-global generator.

**Acceptance:** repeating an evaluation with the same seeds and environment
reproduces its conditions/truths, changing only the posterior seed leaves those
inputs fixed, and separate experiments do not consume each other's RNG stream.

### Resolve settings once, then persist them

Introduce validated configuration objects for training, resume, and scenarios.
Keep `--n-sim` as a compatibility alias but use `batches_per_epoch` internally and
in reports. Expose supported network/optimizer choices through stable factories
or configuration rather than requiring copied training functions.

Store defaults after resolution, not only the user-supplied mapping. Make ignored
settings impossible. Include trial count, TTA levels, integration step, shuffle,
dataset count, posterior budget, and the actual model/simulator version when
those settings apply.

**Acceptance:** a saved run contains enough configuration to explain what was
actually executed, and command-line overrides have a documented precedence.

### Make continuation an explicit experiment decision

Support policies such as restoring a schedule, restarting a schedule while
preserving optimizer moments, or resetting the optimizer. Print the selected
policy and effective learning rate. Record additional and cumulative training
steps, parent artifact, and any changed simulation settings.

The CLI currently defaults to `resume --mode old`; consider requiring explicit
overwrite or defaulting to a new child artifact. Resume dispatch currently follows
the current registered `spec.workflow`; changing that registration can change
how an old artifact is resumed. Record and validate this choice explicitly.

**Acceptance:** users can explain which state was preserved, and an ordinary
resume cannot silently overwrite a valuable artifact or perform zero learning.

## 4. Improve evaluation as an experimental tool

The current shared engine already separates batch generation, posterior sampling,
normalization, metrics, and persistence. Preserve that separation.

| Improvement | Developer benefit | Acceptance evidence |
| --- | --- | --- |
| One shared scientific trial kernel for homogeneous and mixed schedules | Changing a design does not accidentally change equations or response coding. | Tests compare both designs against the same declared generative assumptions. |
| Explicit design labels and training-support metadata | Researchers can distinguish baseline recovery from changed-design generalization. | Reports state the applied design and relevant training differences. |
| Calibration/SBC and posterior predictive evaluation | Recovery plots are supplemented by uncertainty and model-behavior checks. | Separate evaluation types produce documented metrics and appropriate figures. |
| Truth-optional real-data inference | Researchers can run joint participant inference without fabricating recovery truths. | Synthetic data through a real-data loader agree with direct joint inference. |
| Configurable plotting/output hooks | Custom evaluators do not have to produce a recovery plot. | Numeric-only and domain-specific plot modes both work without resampling. |
| Optional sampler batch size and practical progress/resource reporting | Large dataset/draw budgets are easier to run. | Chunked sampling preserves output shape/order and reports its resource settings. |
| Vector/hierarchical target schemas | More cognitive models fit the package contract. | Normalization, metrics, plots and persistence preserve declared target structure. |

When supporting joint participant parameters across conditions, pass the full
trial dataset and covariates to a compatible trained model. Do not concatenate
independent condition-specific posterior draws as a substitute for joint inference.

Also reconcile point-estimate descriptions: recovery bias/RMSE currently use
posterior means, while the evaluator payload declares median plot settings.
Ensure saved settings describe the statistic actually used by each plot/metric.

## 5. Make artifacts portable and packaging independent

Extend metadata with a manifest/schema version, model and input-schema version,
parameter order, resolved config, plugin source revision/hash, package/BayesFlow/
Keras/Torch/NumPy versions, seed policy, parent artifact and checkpoint hash.
Use relative paths within portable artifact bundles where possible.

Current model-name checks do not establish full compatibility. Validate schema
and version compatibility before resume or evaluation, and describe deliberate
migrations. Write metadata atomically, preserve failure/interruption status, and
avoid mutating artifact metadata merely to list runs. Handle alias allocation
safely if concurrent runs are supported.

For distribution, keep installed package imports independent of `examples/`.
Test a built wheel outside the repository with an external plugin workspace,
then add CI for the maintained Python/CPU environment. Add a CPU container only
after this installed workflow works. Treat accelerator variants as separately
verified environments.

**Acceptance:** moving a workspace or installing from a wheel does not break
discovery, artifact lookup, loading, or the documented CLI commands.

## 6. Deliver improvements in small milestones

| Milestone | Deliverable | Definition of done |
| --- | --- | --- |
| 1. Clean working example | P0 fixes, reconciled model/scenario wiring, maintained README and guides. | Fast tests pass and a fresh train/resume/both-evaluations sequence succeeds. |
| 2. Reliable experiment contracts | RNG ownership, validated options, adapter preflight, explicit resume policy. | Invalid options/shapes fail early; repeated seeded experiments are reproducible. |
| 3. Portable release | Clean source tree, wheel verification, manifest/schema compatibility, CI. | Installed package works from another directory without article imports. |
| 4. Broader scientific workflows | Calibration, predictive checks, joint real-data inference, richer targets. | Each workflow has a documented scientific contract and an end-to-end test. |

Keep fast interface tests, small real integration tests, and expensive scientific
validation distinct. A smoke run verifies execution; recovery accuracy,
calibration, and suitability for real participants require their own experiments.
