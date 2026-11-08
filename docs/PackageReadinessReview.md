# Review and action plan for a portable BayesFlow CLI package

Review date: 21 October 2026.

This document reviews the current working tree and proposes implementation work. It does not change the Python code, dependencies, or existing research models. Commands and layouts identified as proposed describe the intended result of future work.

## 1. Assessment

The repository has a useful foundation: `ModelSpec`, built-in workflows, external model plugins, artifact identifiers, checkpoint metadata, and the beginnings of a generic evaluation API. Keep these ideas and finish their integration.

The package is currently partway through a refactor. Its biggest blockers are incomplete evaluation execution, inconsistent scientific assumptions across older and newer models, and an unfinished real-data inference interface. Docker should package a working, tested workflow after these boundaries are established.

The product goal should be:

> A researcher supplies a prior, simulator, and a declaration of observations and experimental conditions. The package provides training, checkpoint management, recovery diagnostics, real-data inference, and reproducible execution.

For ordinary models, researchers should not need to implement a workflow, checkpoint loader, recovery plotting function, or Dockerfile. Advanced models should be able to override adapters and network construction.

### Review evidence and limits

- Inspected the CLI, packaging configuration, model discovery, workflows, training, evaluation modules, real-data helpers, article plugins, tests, and existing architecture documents.
- Parsed all 34 Python files with Python's AST parser: no syntax errors. Parsed `pyproject.toml` successfully.
- Found four imports of internal scientific modules that no longer exist at those package paths.
- The available interpreter is Python 3.14.8; `uv` is installed. This interpreter has no pytest, NumPy, BayesFlow, Keras, or PyTorch installed.
- Attempting `python -m pytest -p no:cacheprovider` stopped with `No module named pytest`. No test suite, training, dependency resolution, wheel installation, or Docker build was validated.
- Findings below distinguish direct code defects from design work. Existing staged, modified, deleted, and untracked files were treated as the current working state and left intact.

## 2. Findings to address first

P0 means a current execution or scientific correctness blocker. P1 means necessary work for a dependable first release. P2 means later improvement.

| Priority | Evidence in current code | Consequence | Action |
| --- | --- | --- | --- |
| P0 | `bayesflow_models/mixed_tta_evaluation.py:11` imports the removed package module `DDM_DC_Pedestrain_TrialWise`. | The real mixed-recovery path cannot import successfully once its scientific dependencies are available. | Move scenario ownership into the article workspace and dispatch through the evaluation registry. |
| P0 | `bayesflow_models/evaluation/fixed_engine.py:370` implements a runner, but `result_payload` is never assigned before it is written and added to the report. | A valid evaluation reaching the reporting stage raises `NameError`. | Complete the recovery calculation and output payload before wiring this runner into the CLI. |
| P0 | `bayesflow_models/evaluation/runner.py` uses bare `from metrics`/`from reporting` imports, calls `resolve_artifact_ref(artifact_ref=...)` although its argument is named `ref`, checks a nonexistent `model_name` metadata key, calls keyword-only `sample()` positionally, references undefined `n_posterior_samples`, and neither saves nor returns its report. | There are two competing runners, and this one contains several independent execution failures. | Retain one supported runner; merge useful reporting work and retire the unfinished alternative. |
| P0 | `bayesflow_models/evaluation/reporting.py:8` imports the `datetime` module, then calls `datetime.now()` at line 107. | This report builder raises `AttributeError`. | Consolidate report creation with the working UTC envelope helper in `fixed_engine.py`. |
| P0 | `cli.py:380` calls a hard-coded mixed-TTA evaluator. `registry.evaluations` is collected but never dispatched. | Evaluation plugins exist as declarations without a generic CLI execution path. | Add scenario listing and a generic `evaluate` command. |
| P0 | `mixed_tta_evaluation.py:17` fixes four parameter names and its own priors and equations. Active `article_models.py` registers a seven-parameter `model_DC` from `reproduce_models.py`. | The evaluator does not describe the currently registered model. Repairing the import alone would leave a scientific mismatch. | Generate evaluation observations through the same model kernels and parameter definitions used for training. |
| P0 | `utils_real_data.py:111` infers separately by condition, then concatenates draws or averages posterior summaries. | These operations do not produce the joint posterior for one subject's shared parameters. | Implement inference over the subject's complete trial dataset and covariates, validated for that training distribution. |
| P0 | `mixed_tta_evaluation.py:155` subtracts targets shaped `(N, 1)` from posterior means shaped `(N,)`. The generic metrics module also uses unrestricted `np.squeeze()`. | Legacy metrics can broadcast to `(N, N)`; generic metrics reject valid singleton dataset/draw cases after squeezing. | Normalize named parameter arrays to explicit dataset and draw dimensions before calculating metrics. |
| P1 | `train.py:296` selects CUDA automatically; the CLI's configured device is never passed into the trainer. | `--device cpu` can still train on a GPU. | Resolve the device once and pass it consistently through training, loading, and sampling. |
| P1 | `cli.py` parses `resume --n-sim`, but `build_runtime_config()` only copies resume epochs and batch size. | The resume simulation-budget option is ignored. | Validate and propagate all supported options; test the resulting trainer arguments. |
| P1 | `train.py:304` passes `n_sim` as `num_batches_per_epoch`. | Documentation describes simulations while the code counts batches; cost changes with batch size. | Rename to `batches_per_epoch` or explicitly convert a requested dataset budget to batches. |
| P1 | `interfaces.py:38` declares resume/recovery callable signatures that disagree with the CLI's `(spec, config)` calls. | Plugin authors cannot rely on the published type contract. | Align callback signatures and return types with actual supported execution. |
| P1 | `main.py:15`, `workflow_trialwise.py:28`, and `parameter_recovery.py:34` also import removed internal scientific modules. | Legacy entry points remain broken and obscure which interface is supported. | Archive, migrate, or deprecate them after equivalent functionality exists in the package. |
| P1 | No `uv.lock`, Dockerfile, `.dockerignore`, Compose file, or CI configuration was found. README/setup scripts reference deleted requirements and moved notebooks. | A new user lacks a verified installation and execution path. | Establish one uv workflow, test the built distribution, then provide a CPU container and updated quickstart. |

The mixed-recovery CLI test replaces the actual evaluation module in `sys.modules`. It verifies dispatch, but cannot detect the missing model import or scientific mismatch. Preserve this unit test and add an integration test that imports and runs the real registered scenario.

## 3. Make package and researcher responsibilities explicit

Use three layers, building on the contracts already present.

| Layer | Owns | Must remain independent of |
| --- | --- | --- |
| Execution engine | Configuration, discovery, artifact resolution, checkpoint loading, posterior sampling, output locations, logging. | TTA values, DDM equations, participant CSV column names. |
| Reusable scientific operations | Recovery summaries, calibration, posterior predictive orchestration, common plots and exports. | One article's prior ranges and experimental schedules. |
| Researcher workspace | Prior, simulator, experimental design, observation encoding, data mapping, domain-specific predictive statistics. | Package installation paths and checkpoint bookkeeping. |

A practical target layout is:

```text
pyproject.toml
uv.lock
.python-version
src/
  bayesflow_models/
    __init__.py
    __main__.py
    api.py
    cli.py
    config.py
    discovery.py
    artifacts.py
    training.py
    inference.py
    evaluation/
      interfaces.py
      engine.py
      recovery.py
      reporting.py
examples/
  gaussian_workspace/
  article_workspace/
    config.yaml
    plugins/
    data/
tests/
docs/
Dockerfile
.dockerignore
compose.yaml
```

This is a destination, not a requirement to rename and split everything in one commit. Keep the existing import package name during the first migration. Move the console entry point from `cli:main` to `bayesflow_models.cli:main` when the CLI moves, and update tests with it.

Use a `src/` layout to make accidental imports from a checkout less likely; still test the installed wheel outside the repository. This layout supports the packaging boundary described by the [Python Packaging User Guide](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/).

### Keep the model-author API small

The minimum contribution should contain a prior, a simulator, parameter names, and an observation/condition declaration. Initially retain the current adapter factory; later generate a default adapter from that declaration. Expose an explicit public API instead of requiring authors to import private workflow helpers.

Provide default workflow and network presets. Allow advanced authors to supply an adapter or network factory without reimplementing training and persistence. The current fixed `SetTransformer(summary_dim=10)` plus `CouplingFlow` is one useful preset, but should not be the only supported architecture.

Add a model validation command that builds a tiny simulation batch, checks the adapter, performs a minimal network forward/sample operation, and explains failures using the declared parameter and input names.

### Represent conditions independently of TTA

Distinguish observations, trial covariates, dataset covariates, and inference targets. A condition can be TTA, stimulus strength, task difficulty, group, or another declared experimental variable.

Do not impose a universal key such as `tta_per_trial` on the engine. The model schema should declare keys, feature order, shapes, dtypes, units, categorical encoding, and allowed values or ranges. Keep categorical encoding consistent between simulation and real data.

For an exchangeable trial model, an example schema is observations `(N, T, F)`, trial covariates `(N, T, C)`, dataset covariates `(N, D)`, and parameters `(N, P)`. These are examples; the engine should permit model-specific schemas.

If a model includes learning, fatigue, or sequential dependence, preserve trial order and use a suitable sequence architecture. A set encoder removes order information. For variable trial counts, initially process subjects individually or bucket equal lengths; introduce padding only with tested mask handling through adapters and networks.

## 4. Complete generic evaluation before extending mixed TTA

Use `EvaluationSpec.generate_batch(model_spec, options)` and `EvaluationBatch` as the current scenario extension point. The scenario supplies data and truths; common recovery code scores posterior draws.

Implement in this order:

1. Select one runner and one report schema. `fixed_engine.py` is the stronger starting point because artifact and batch validation are already separated into helpers.
2. Normalize posterior output to `{parameter_name: array(N, S)}`. Check the expected dataset count against validated truths, rather than deriving it from the posterior itself. The current generic metrics dataset-count check compares a value with itself.
3. Handle explicitly supported shapes such as `(N, S, 1)` without losing singleton dataset or draw axes. Reject unsupported orientation; never guess based on matching dimension lengths. Record the tested BayesFlow output convention.
4. Calculate bias, RMSE, interval coverage, posterior spread, and correlation where defined. Use `null` for undefined correlation, and write strict JSON with non-finite values handled explicitly.
5. Build `result_payload`, then persist posterior draws, truths, summaries, plots, and a report. Save the evaluated inputs or a stable reference/hash so a failure can be reproduced. Large arrays belong in array files.
6. Give each evaluation invocation its own identifier or require explicit overwrite. Current scenario output paths overwrite previous runs of the same scenario.
7. Add `evaluations list` and `evaluate --artifacts ... --scenario ...`, resolving scenarios by `(model_name, scenario_name)`. Produce an actionable error when a model has no matching scenario.
8. Route ordinary recovery through the same orchestration with a default simulator-generated batch. Keep a temporary compatibility alias for `mixed-recovery` if needed.

Keep truths separate from posterior conditions. Verify that inference succeeds with no true parameter values in the supplied input. Ground-truth fields must not become a hidden inference requirement.

Choose one adaptation boundary. In the pinned BayesFlow 2.0.8 API, `ContinuousApproximator.sample()` is keyword-only and preprocesses conditions with its adapter. Prefer raw, schema-valid condition dictionaries at this boundary, and test that real-data inputs are transformed exactly once. The current real-data helper applies the adapter before calling `sample()`. See the [versioned sampling API](https://bayesflow.org/v2.0.8/api/bayesflow.approximators.ContinuousApproximator.html).

Do not force every future diagnostic into a batch containing known parameters. Recovery needs truths; inference and posterior predictive checks need different inputs. Reuse loading, sampling, and persistence helpers while giving each operation an appropriate contract.

## 5. Implement mixed-TTA evaluation as a scientific scenario

### Reuse the actual scientific model

The active `model_DC` has parameters `tau`, `b0`, `k`, `mu_ndt`, `sigma_ndt`, `alpha`, and `theta`. The legacy mixed evaluator samples `theta`, `b0`, `mu_ndt`, and `mu_alpah`, with different ranges and fixed values for other quantities. Treat these as different scientific specifications.

Before implementing a new scenario, choose the authoritative model equations and version them. `reproduce_models.py` and `reproduce_models_utility.py` also differ in equations, including how `tau` and `theta` scale with TTA. Consolidation must preserve an explicitly chosen scientific meaning and document any change.

Extract or expose reusable prior and simulation functions that accept an explicit RNG and experimental design. Both the BayesFlow training wrapper and evaluation plugin should call those functions. Do not copy the diffusion equations into another evaluator.

### Generate one coherent dataset per participant

For each simulated participant:

1. Sample one parameter vector from the registered model's prior.
2. Construct the participant's trial-condition schedule.
3. Simulate all trials using that same parameter vector and their respective conditions.
4. Return observations and covariates in the checkpoint's input schema, with one true value per parameter per participant.

Expose scenario options for condition levels, balanced or custom counts, shuffling, jitter, dataset count, and seed. Validate positive counts, allowed values, and schedule length. Record nominal versus jittered TTA semantics; use the same definition that the available real-data columns support.

### Separate recovery from generalization claims

Putting TTA alongside each trial allows the encoder to see it. This does not establish that a network trained on homogeneous-TTA datasets is accurate on mixed-TTA datasets. The current `reproduce_models.py:249` samples one nominal TTA for an entire 60-trial dataset.

Use a recorded comparison:

| Training design | Evaluation design | Interpretation |
| --- | --- | --- |
| Homogeneous TTA per dataset | Homogeneous TTA per dataset | Baseline recovery under the training design. |
| Homogeneous TTA per dataset | Mixed TTA within each dataset | Generalization stress test under a changed design. |
| Mixed TTA within each dataset | Mixed TTA within each dataset | Recovery under the intended mixed-trial workflow. |

For real-data use, train across the condition compositions and trial counts you intend to support, then validate held-out datasets under those designs. Add unbalanced and missing-condition cases when they are part of the intended use. Label out-of-support conditions explicitly rather than implying that conditioning guarantees arbitrary extrapolation.

Preserve shuffle invariance as a meaningful test only for exchangeable models. For order-dependent models, shuffling changes the scientific dataset.

## 6. Make real-data inference a supported CLI operation

Add an `infer` command with an artifact reference, data path, dataset/group identifier mapping, condition mapping, posterior draw count, and output location. Put reusable loading and grouping in the package; keep scientific encodings in model declarations or workspace hooks.

For one subject with shared parameters and trials `(y_t, c_t)`, the intended target is:

```text
p(theta | y_1:T, c_1:T)
  proportional to p(theta) * product_t p(y_t | theta, c_t)
```

This expression assumes trials are conditionally independent given the subject parameters and covariates. Sequential models need their corresponding joint generative process.

Concatenating posterior draws from individual conditions creates a mixture of the condition-specific posteriors. Averaging their means is a summary operation. Neither operation gives the joint posterior above. Even multiplying condition-specific posteriors would require correcting repeated prior factors and a justified approximation method.

The preferred workflow is one inference call over all of a subject's trials, using an approximator trained and checked for that design. Per-condition estimates may still be exported as separate diagnostic analyses.

Required actions:

- Validate units, trial coding, column mappings, finite values, condition support, and subject grouping before sampling.
- Make exclusion and missing-data rules explicit. The current zero filling of unavailable CPP values is only valid if the model and training process intentionally support that encoding.
- Match each artifact's actual observation features. The active reproduced models emit one observed feature, while older helpers assume `[RT, CPP]`.
- Export per-subject draws and a tidy summary table with parameter names and credible intervals. Record exclusions and failed subjects; avoid silent partial success.
- Provide posterior predictive checks by simulating from inferred parameters under the observed experimental design. Researchers supply domain-specific summaries such as choice proportions and RT quantiles.
- Keep real-data model adequacy separate from parameter recovery: true real-data parameter values are unavailable.

Do not describe separate subject fits as a hierarchical population model. Hierarchical inference requires an explicit population/subject generative model and a compatible inference workflow.

## 7. Make training and artifacts reproducible

### Configuration and device handling

Replace scattered nested dictionaries with one validated configuration model. A dataclass-based implementation is sufficient initially. Define precedence as CLI overrides, then workspace configuration, then package defaults. Store the resolved configuration with every run.

Make the selected device reach the actual trainer and approximator. Support `cpu` first and a tested CUDA path next. Set the Keras backend before importing Keras/BayesFlow; postpone heavy imports until execution so help and lightweight discovery do not eagerly load the entire training stack. BayesFlow documents this backend setup in its [2.0.8 installation guide](https://bayesflow.org/v2.0.8/index.html).

Rename the training budget to reflect batches. For example, 100 batches at batch size 32 means 3,200 training datasets per epoch, excluding validation and implementation-specific setup calls. Each dataset can itself contain many trials. The meaning of `num_batches_per_epoch` is documented in the [2.0.8 workflow API](https://bayesflow.org/v2.0.8/api/bayesflow.workflows.BasicWorkflow.html).

### Checkpoint and resume semantics

Use one supported serialization path. Consolidate the legacy `.pt`, weight-only, and full `.keras` training helpers after confirming which existing artifacts require migration. Keras describes saved model configuration, weights, and optimizer state in its [serialization guide](https://keras.io/guides/serialization_and_saving/).

Verify optimizer restoration in a real save/load/resume test. `fit_online()` exposes optimizer-retention behavior; do not infer correct continuation solely from a successful load. Also verify architecture, adapter configuration, parameter order, and preprocessing compatibility before resuming.

Preserve the existing fail-if-missing behavior in `resume_from_artifact()` and make the lower-level resume operation equally explicit. The shared trainer currently starts fresh when its requested checkpoint is absent; distinguish intentional fresh training from continuation.

Record original and additional epochs, cumulative steps, parent artifact, optimizer policy, and any changed simulation settings. Write metadata atomically and retain enough status information to identify interrupted runs. Listing checkpoints should not assign aliases or rewrite metadata as a side effect.

### Minimum artifact manifest

| Category | Record |
| --- | --- |
| Identity | Artifact ID, optional display alias, manifest schema version, parent ID. |
| Scientific definition | Model version, parameter order, prior specification/version, condition design, observation schema. |
| Implementation | Package/BayesFlow/Keras/backend versions, plugin source hash or version, resolved configuration. |
| Reproduction | Seed policy, data or simulation input references, relevant RNG state for continuation, device information. |
| Persistence | Checkpoint hash, relative paths within an artifact bundle, training history and status. |

Use explicit RNG ownership. Module-global generators seeded at import time make results depend on import order and previous calls; seed global libraries alone does not reset an existing `default_rng()` instance. Separate training and evaluation streams, and define worker seeding before parallel simulation. Promise reproducibility within a documented environment; exact CPU/GPU numerical equality is a separate requirement.

Keep immutable artifact IDs authoritative. Numeric aliases should be display conveniences with collision handling if concurrent runs are supported. A copied artifact should resolve under a new workspace root without its original absolute paths.

## 8. Establish uv as the supported environment workflow

The existing `pyproject.toml` already defines a build backend, a console script, CPU/CUDA extras, and explicit PyTorch indexes. Preserve that useful starting point.

Actions:

1. Start with a tested Python 3.12 environment and commit `.python-version`. Reconcile `requires-python` with the versions actually tested; the current open-ended `>=3.12` declaration does not demonstrate compatibility with newer interpreters and pinned binary wheels.
2. Resolve and commit `uv.lock`, including development tools and supported accelerator choices. Record the uv version used by CI and Docker.
3. Retain the existing package pins for the initial baseline. After establishing compatibility tests, decide which published library dependencies can use tested ranges; keep exact deployment resolution in the lockfile.
4. Move developer tooling to a development dependency group and keep notebooks optional. Until that migration, the existing development tools are selected with `--extra dev`.
5. Audit direct dependencies. No Python source import of `torchvision` was found; verify whether it is needed by the supported dependency stack before retaining it as a direct requirement.
6. Define what an installation without an accelerator extra should do. CPU and CUDA extras currently select index sources, but neither is selected by default.
7. Test each advertised platform and accelerator. Document that uv-specific index routing is not a portable wheel metadata mechanism for every installer.
8. Replace or deprecate the setup scripts that reference removed requirements files. Add `.venv/`, `dist/`, and build output ignores as appropriate.

The CPU/CUDA index approach is supported by the [uv PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/). The current `torch==2.7.1`, `torchvision==0.22.1`, and CUDA 11.8 combination is listed by [PyTorch's versioned installation instructions](https://pytorch.org/get-started/previous-versions/); an upgrade is not required just to adopt uv. Actual platform and Python compatibility still need installation tests.

Baseline commands using the current extras, to run during implementation:

```bash
uv python pin 3.12
uv lock
uv sync --locked --extra cpu --extra dev
uv run --locked --extra cpu --extra dev pytest
uv run --locked --extra cpu bayesflow-cli --help
uv run --locked --extra cpu bayesflow-cli \
  --plugin-path examples/article_workspace/plugins models list
uv build
```

These commands have not been validated in this review. The first two create project files; execute them when beginning the implementation. After moving development dependencies to a group, replace `--extra dev` with `--group dev` in the documented workflow.

`uv lock` records exact resolution; `--locked` rejects stale lockfiles, whereas `--frozen` skips that freshness check. Use locked installs and runs in verification. See [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/).

## 9. Package the validated workflow with Docker

### First release target: a CPU CLI image

Build a CPU image before adding a separate GPU variant. Keep application installation under `/opt/bayesflow-cli` and the researcher workspace under `/workspace`, so a bind mount cannot hide the installed environment.

Implementation sequence:

1. Add `.dockerignore` excluding Git internals, local environments, caches, data, checkpoints, results, logs, and unrelated notebooks or large research files. `.gitignore` does not control Docker's build context.
2. Use a tested Python 3.12 slim base; pin the base and uv image versions/digests for releases.
3. Copy `pyproject.toml` and `uv.lock` first. Install locked CPU dependencies without development tools or the project, using a build cache.
4. Copy the required build inputs, including `README.md` and the chosen license, then install the project non-editably from the same lockfile.
5. Put the installed environment on `PATH`, set `KERAS_BACKEND=torch`, `MPLBACKEND=Agg`, and unbuffered output, and use an exec-form console-script entry point.
6. Configure a non-root runtime user and writable output/cache directories. On Linux, document a host UID/GID option for mounted output ownership.
7. Set `/workspace` as the runtime working directory. Require no dependency downloads or environment synchronization during ordinary container execution.
8. Add an optional Compose service that captures the repeated workspace mounts and paths.

Dependency-first layers and `uv sync --locked --no-editable` are supported in the [uv Docker guide](https://docs.astral.sh/uv/guides/integration/docker/). Keep the same environment path and compatible Python/base image if copying a virtual environment between build stages.

Proposed runtime example, after building the image:

```bash
docker run --rm \
  --mount type=bind,source="$PWD/examples/article_workspace",target=/workspace \
  bayesflow-cli:cpu \
  --plugin-path /workspace/plugins models list
```

With `/workspace` mounted, default checkpoints/results/logs should persist there. For shared deployments, allow plugins and input data to be mounted read-only with separate writable output mounts.

Plugins with additional Python dependencies need an explicit installation recipe. Declare them in the researcher project's uv environment or build a derived image; mounting a Python file alone does not install its dependencies. Test custom Keras serialization objects in a fresh container process with the required plugin package available.

### GPU image after CPU acceptance

Provide a separate tested accelerator variant, select the intended PyTorch index, document the host NVIDIA driver/container runtime requirements, and test actual training with GPU access. A CUDA-enabled image does not supply the host driver. Avoid advertising every backend before maintaining tests for it.

### Container acceptance

- Help and model listing work in a clean workspace and with an external model plugin.
- A tiny train/save/load/recovery/infer sequence works with persistent mounted outputs.
- A second container can read the first container's checkpoint using the same plugin definition.
- Moving the workspace and changing the host directory does not break artifact lookup.
- Save-only plotting runs without a display; `--device cpu` is respected on a GPU host.
- Expected bad input produces a readable error and nonzero exit status.

## 10. Add tests that protect scientific and packaging boundaries

Keep fast mocked CLI tests, then add a small number of real integration tests. A routing test alone cannot establish a working scientific workflow.

| Test | What it should establish |
| --- | --- |
| Analytic Gaussian example | The package supports a non-DDM model; inferred posterior moments approach the known result within a justified tolerance. |
| Installed wheel in a fresh directory | CLI imports, bundled modules, and public API work without the source checkout on `PYTHONPATH`. |
| Real runner with a tiny model | Artifact loading, batch generation, sampling, metrics, and all report files work together. |
| Shape and metric fixtures | Correct results for `N=1`, `S=1`, equal dataset/draw counts, known intervals, mismatched targets, and invalid/non-finite inputs. |
| Mixed-design scenario | One parameter vector governs each participant; declared counts and covariates align with observations; the intended model kernel is used. |
| Synthetic CSV round trip | A simulator-generated dataset can pass through the real-data loader and inference interface without truths or duplicate adaptation. |
| Save/load/resume | Parameters, adapter behavior, and optimizer policy survive a fresh process; requested device and training budget are honored. |
| Workspace portability | Artifacts and plugins work from a different directory and inside the CPU image. |

For scientific validation beyond smoke tests, include parameter recovery across prior regions, calibration/interval coverage, and posterior predictive checks. Report Monte Carlo uncertainty; 50 evaluation datasets do not establish precise nominal 95% coverage. Do not use a universal recovery correlation or R² threshold as the only acceptance criterion.

Simulation-based calibration checks posterior computation under the generative model; it does not prove the model describes real participants. See the original [SBC paper](https://arxiv.org/abs/1804.06788) and [BayesFlow diagnostics overview](https://bayesflow.org/v2.0.12/user_guide/diagnostics.html). The latter is a newer documentation version: verify the available diagnostic APIs against the package's pinned BayesFlow version before implementation.

Add CI for linting, unit tests, a tiny CPU integration run, distribution build/install checks, and the CPU container smoke test. Start with Python 3.12, then add versions and platforms only as they are verified. Keep expensive recovery/calibration experiments separate from the fast CI suite.

## 11. Deliver the work in reviewable stages

| Stage | Deliverable | Completion criterion |
| --- | --- | --- |
| 1. Reproducible baseline | Tested Python choice, uv lock, environment instructions, baseline test results. | A clean CPU installation resolves and existing test results are recorded. |
| 2. Stable package boundary | Public API, consistent callback types, installed CLI, legacy path decisions. | A built wheel runs from a fresh directory with an external example plugin. |
| 3. Complete recovery engine | One runner, shape normalization, metrics, array/report persistence. | Real evaluation completes through the CLI and fails clearly for invalid batches. |
| 4. Correct mixed-TTA scenario | Versioned model kernels, seedable design generation, external scenario registration. | Homogeneous and mixed-design evaluations use the declared model and report their interpretation. |
| 5. Real-data inference | Schema mapping, joint subject inference, exports, predictive-check hooks. | Synthetic data travel through the real-data interface and match direct inference. |
| 6. Reliable runs | Resolved config, device propagation, resume tests, manifests, isolated evaluation outputs. | A saved run can be explained, resumed, copied, and evaluated without hidden state. |
| 7. Portable distribution | CPU Docker image, optional Compose configuration, wheel/container CI. | The same tiny workflow succeeds locally with uv and in a fresh container. |
| 8. Researcher release | Starter workspace, focused tutorial, reference documentation, chosen license and metadata. | A new user can add a model and run the complete workflow without editing package internals. |

Fix directly blocking defects as soon as the baseline exposes them. Separate scientific equation changes from packaging moves so reviewers can assess each independently. Avoid combining a BayesFlow upgrade, model-equation revision, layout migration, and Docker rollout in one change.

### Documentation to keep and consolidate

The existing [portable plugin refactor plan](PortablePluginArchitectureRefactor.md), [engine behavior guide](EngineFixedEvaluationBehaviors.md), [recovery behavior guide](ScientificRecoveryEvaluationBehaviors.md), and [mixed-TTA scenario guide](ScenarioSpecificMixedTTABehaviors.md) already describe much of the intended direction. Some steps are implemented; others remain proposals. Use this review as the current status and acceptance checklist rather than restarting those plans.

Replace the outdated README with one verified path: install with uv, create/copy a workspace, validate a model, train, recover, infer, and run the same commands in Docker. Distinguish implemented commands from proposed ones. Add a small Gaussian example before the article-specific tutorial so users can learn the package independently of DDM details.

Fill in the package author, confirm the intended license and include its file, document the supported version matrix, and add release notes. Preserve useful legacy scientific scripts as clearly labeled examples or archived material until their results can be reproduced through the supported interface.

## 12. First-release definition of done

- [ ] Researchers can define a model and its input schema without implementing infrastructure.
- [ ] The core contains no required imports of article-specific scientific modules.
- [ ] Generic evaluation is available through the CLI and produces usable numerical reports and arrays.
- [ ] Mixed-TTA evaluation reuses the correct registered model and labels changed-design tests accurately.
- [ ] Real-data inference targets the declared joint dataset posterior and checks schema compatibility.
- [ ] Device selection, training budgets, resume behavior, and parameter order are tested.
- [ ] `uv.lock` and a tested Python version reproduce a clean CPU environment.
- [ ] The installed wheel and CPU container work with external plugins and portable artifacts.
- [ ] The README contains commands exercised by tests or release verification.

The most useful next implementation step is to establish the uv baseline, finish one generic recovery execution path, and prove it with the Gaussian example. That gives the mixed-TTA scenario, real-data workflow, and Docker image a dependable foundation.
