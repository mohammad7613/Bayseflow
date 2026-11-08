# Refactoring Toward a Portable Plugin-Based CLI

## Purpose

This document is an implementation plan, not an already-applied refactor. It
describes small, ordered code changes for two related workstreams:

1. **Model-plugin boundary:** remove researcher-owned scientific models from the
   installed application and load them from a mounted workspace.
2. **Evaluation boundary:** keep generic evaluation orchestration in the
   installed application while obtaining model-specific evaluation scenarios
   from that workspace.

Together, these workstreams turn the current project into a portable cognitive-
modeling CLI. Dockerization comes afterward and packages the boundary already
defined by the Python wheel.

The desired boundary is:

```text
Build-time input                         Runtime input
----------------                         -------------
pyproject.toml                           trusted model plugins
README.md                                optional custom workflows
cli.py                                   real data
generic bayesflow_models package         evaluation scenarios
generic evaluation runner                real data
                                         checkpoints
                                         results
                                         logs
```

The Docker image should not contain the article-specific DC, DF, and SF model
implementations or model-specific validation generators. It should contain the
reusable CLI, plugin interfaces, discovery, artifact management, training code,
one generic workflow, and a generic evaluation runner.

Do not perform all changes at once. Complete and validate each phase before
continuing. File line numbers refer to the current source and will move as edits
are made; prefer the named function or symbol when applying later steps.

---

## 1. Make the Two-Workstream Architecture Explicit

Add this contract to the package documentation before changing code:

> `bayesflow-cli` is a generic execution engine. The installed package provides
> a workflow named `builtin`, but provides no scientific `ModelSpec` objects.
> Users supply trusted model plugins through `--plugin-path`; when the flag is
> omitted, the CLI scans its current working directory. In Docker, the current
> working directory is `/workspace`. The installed package also provides generic
> evaluation orchestration. A researcher plugin supplies each model-specific
> evaluation scenario and the data needed to evaluate its trained posterior.

Short reason: this sentence tells developers which behavior is intentional and
prevents project-specific models from slowly returning to the core package.

The resulting ownership is:

| Component | Owner | Included in wheel/image? |
|---|---|---:|
| CLI parsing and dispatch | CLI engine | Yes |
| `ModelSpec`, `Workflow`, and context types | CLI engine | Yes |
| Plugin discovery | CLI engine | Yes |
| Generic train/resume/recovery workflow | CLI engine | Yes |
| Artifact metadata and checkpoint resolution | CLI engine | Yes |
| Evaluation interfaces, runner, metrics, and reporting | CLI engine | Yes |
| DC, DF, and SF simulators/adapters | Research workspace | No |
| Mixed-TTA scientific data generator | Research workspace | No |
| Mixed-TTA evaluation orchestration | CLI engine | Yes |
| Real data and experiment output | Research workspace | No |

The ownership test is simple:

- If code can operate through stable interfaces without knowing DC, DF, SF,
  parameter priors, or TTA values, it belongs to the engine.
- If code knows model equations, priors, parameter names, TTA conditions, or the
  exact posterior input schema, it belongs to the researcher plugin.

Short reason: ownership should follow knowledge and dependencies, not the file's
current directory.

The implementation order is:

```text
baseline tests
    |
    +--> Workstream 1: external model plugins
    |
    +--> Workstream 2: engine runner + workspace scenarios
    |
    +--> wheel boundary verification
    |
    +--> Docker build/runtime implementation
```

Use the detailed sections as follows: the first sentence of every implementation
step names the file to change, the code block shows the intended change, and the
“Short reason” states why it belongs at that boundary. Conceptual inspection
steps explicitly say that no source change is made.

### Workstream 1 at a glance

| Order | File or directory | Simple change | Reason |
|---:|---|---|---|
| 1 | `tests/` | Add plugin-boundary tests | Protect behavior before moving code |
| 2 | `bayesflow_models/builtin_workflows.py` | Register the generic workflow explicitly | Keep reusable workflow behavior in the engine |
| 3 | `bayesflow_models/discovery.py` | Stop importing internal scientific models | Discover researcher models only at runtime |
| 4 | `cli.py` | Clarify plugin paths and empty-registry behavior | Give researchers a predictable CLI |
| 5 | `examples/article_workspace/plugins/` | Move DC, DF, and SF model code here | Keep scientific definitions out of the wheel |
| 6 | `bayesflow_models/internalpluginpath/` | Delete after equivalence tests pass | Remove the duplicate internal registry |

### Workstream 2 at a glance

| Order | File or directory | Simple change | Reason |
|---:|---|---|---|
| 1 | `bayesflow_models/evaluation/interfaces.py` | Define evaluation batch and registration contracts | Give both sides a stable boundary |
| 2 | `bayesflow_models/discovery.py` | Discover evaluation scenarios with models/workflows | Load all plugin capabilities once |
| 3 | `bayesflow_models/evaluation/runner.py` | Orchestrate evaluation generically | Keep artifact and posterior mechanics in the engine |
| 4 | `bayesflow_models/evaluation/metrics.py` | Compute model-neutral recovery metrics | Avoid scientific-model coupling |
| 5 | `bayesflow_models/evaluation/reporting.py` | Save generic plots and reports | Standardize researcher outputs |
| 6 | `cli.py` | Add `evaluate --scenario` | Expose a portable evaluation command |
| 7 | workspace `reproduce_models.py` | Extract reusable kernels and seeded priors | Make training and evaluation scientifically identical |
| 8 | workspace `mixed_tta_scenarios.py` | Generate model-specific heterogeneous-condition data | Keep TTA/model knowledge in the workspace |
| 9 | `tests/` | Verify targets, shapes, seeds, and scenario ownership | Prevent invalid recovery conclusions |
| 10 | old `mixed_tta_evaluation.py` | Delete after the split passes | Remove duplicated equations and orchestration |

---

## 2. Record a Baseline Before Refactoring

### Step 2.1: Add focused tests

Create these tests before moving files:

```text
tests/
├── test_discovery.py
├── test_cli_models.py
├── test_builtin_workflow.py
├── test_evaluation_discovery.py
├── test_evaluation_runner.py
└── fixtures/
    └── plugins/
        ├── example_models.py
        ├── example_evaluations.py
        └── helper.py
```

Short reason: import-graph cleanup is unsafe if the only validation is a costly
training run.

### Step 2.2: Create a lightweight plugin fixture

In `tests/fixtures/plugins/example_models.py`, define a minimal `MODEL_SPECS`
list. The simulator and adapter factories may return small test doubles; the
discovery test does not need BayesFlow training.

The fixture should import a small value or function from `helper.py` so the test
also verifies that sibling plugin imports work through
`plugin_import_context()`.

Short reason: external plugins frequently consist of one registration file plus
helper modules, not one completely self-contained file.

### Step 2.3: Assert current and target-relevant behavior

Add tests for the following:

1. `resolve_plugin_paths([])` resolves to `Path.cwd()`.
2. An explicit plugin directory is used instead of `Path.cwd()`.
3. A missing plugin directory raises `FileNotFoundError`.
4. A non-directory plugin path raises `ValueError`.
5. A file without plugin markers is not imported.
6. A plugin containing `MODEL_SPECS` is discovered.
7. Duplicate model names raise a clear error.
8. The generic `builtin` workflow is available without a workspace plugin.
9. `models list` with no model plugins succeeds and reports an empty registry.
10. A model referencing a nonexistent workflow fails before training begins.
11. An evaluation scenario is associated with one explicit model.
12. A generic evaluation runner works with a fake scenario and fake posterior.

Short reason: these are the portable application's actual boundary conditions.

### Step 2.4: Save baseline package evidence

Before moving code, build a wheel and list its contents:

```bash
python -m pip wheel . \
    --no-deps \
    --no-build-isolation \
    --wheel-dir /tmp/bayesflow-wheel-before

python -m zipfile -l /tmp/bayesflow-wheel-before/*.whl
```

Short reason: the before/after wheel listing proves whether research modules
were actually removed from the distributable artifact.

Do not treat `--no-build-isolation` as universally required. If it is used, the
active environment must already provide `setuptools>=68`.

---

## 3. Workstream 1: Separate Built-in Workflows from Research Models

The current `internalpluginpath` mixes two different concerns:

```text
internalpluginpath/
├── trailwise_builtin.py   # article-specific models
└── workflow_builtin.py    # reusable workflow registration
```

Both discovery functions iterate over every file in that directory. Therefore,
workflow discovery imports the model registry, and model discovery imports the
workflow registry. The clean design should not dynamically scan internal Python
modules at all.

### Step 3.1: Create an explicit built-in workflow registry

Create:

```text
bayesflow_models/builtin_workflows.py
```

Move the active content of
`bayesflow_models/internalpluginpath/workflow_builtin.py` into it, but use a
constant with a precise type:

```python
from bayesflow_models.interfaces import Workflow
from bayesflow_models.workflow import (
    recovery_from_artifact,
    resume_from_artifact,
    train_from_spec,
)


BUILTIN_WORKFLOWS: tuple[Workflow, ...] = (
    Workflow(
        name="builtin",
        train_fn=train_from_spec,
        resume_fn=resume_from_artifact,
        recovery_fn=recovery_from_artifact,
    ),
)
```

Short reason: an explicit tuple is predictable, typed, and does not require
scanning and executing arbitrary internal modules.

### Step 3.2: Remove dynamic internal imports from model discovery

In `bayesflow_models/discovery.py`, inside `discover_model_specs()`, remove the
current block that imports `bayesflow_models.internalpluginpath` and iterates
over it:

```python
package = import_module("bayesflow_models.internalpluginpath")
for module_info in iter_modules(package.__path__):
    ...
```

The function should start with an empty registry and scan only resolved runtime
plugin paths:

```python
def discover_model_specs(plugin_paths=None) -> dict[str, ModelSpec]:
    specs: dict[str, ModelSpec] = {}

    for plugin_path in resolve_plugin_paths(plugin_paths):
        ...

    return specs
```

Short reason: the core package promises no scientific models; every discovered
model must come from runtime input.

### Step 3.3: Initialize workflows explicitly

At the top of `bayesflow_models/discovery.py`, add:

```python
from bayesflow_models.builtin_workflows import BUILTIN_WORKFLOWS
```

In `discover_workflows()`, replace the internal package scan with:

```python
workflows = {
    workflow.name: workflow
    for workflow in BUILTIN_WORKFLOWS
}
```

Continue scanning runtime plugin paths afterward so users may register custom
workflows.

Short reason: one reusable default workflow remains available without coupling
the wheel to article-specific models.

### Step 3.4: Remove imports made obsolete by Steps 3.2 and 3.3

From `bayesflow_models/discovery.py`, remove:

```python
from importlib import import_module
from pkgutil import iter_modules
```

Keep:

```python
from importlib.util import module_from_spec, spec_from_file_location
```

Short reason: unused imports hide the fact that internal scanning has genuinely
been removed.

### Step 3.5: Fix the mutable default argument

Change:

```python
def discover_workflows(plugin_paths: list = []) -> dict[str, Workflow]:
```

to:

```python
def discover_workflows(plugin_paths=None) -> dict[str, Workflow]:
```

Short reason: a shared mutable default can leak state across calls and is not
needed because `resolve_plugin_paths()` already handles a missing value.

### Step 3.6: Define duplicate workflow behavior

The current external workflow loop silently overwrites a workflow with the same
name. Replace that assignment with an explicit check:

```python
for workflow in _collect_workflows_from_module(module):
    if workflow.name in workflows:
        raise ValueError(f"Duplicate workflow name: {workflow.name}")
    workflows[workflow.name] = workflow
```

Use the renamed helper described in Step 4.2.

Short reason: silently replacing the built-in workflow makes behavior depend on
plugin scan order and weakens reproducibility.

If overriding built-ins is a desired feature later, implement it as an explicit
flag or policy instead of an accidental dictionary overwrite.

### Step 3.7: Remove the impossible empty-workflow check

After `BUILTIN_WORKFLOWS` always initializes the registry, this block is no
longer useful:

```python
if len(workflows) == 0:
    raise ValueError("No workflow is found")
```

Replace it with a startup assertion near the built-in registry if desired:

```python
if not BUILTIN_WORKFLOWS:
    raise RuntimeError("The package contains no default workflow")
```

Short reason: validate the invariant where it is defined rather than checking an
unreachable state at the end of every discovery call.

### Validation checkpoint A

Run only the discovery and CLI-listing tests. Confirm:

- no import of `internalpluginpath` occurs;
- an empty workspace produces zero model specs;
- `builtin` is still present in the workflow registry; and
- an explicit test plugin is discovered.

Do not delete `internalpluginpath` until this checkpoint passes.

---

## 4. Tighten the External Plugin API

### Step 4.1: Correct the `--plugin-path` help text

The CLI currently says that the option accepts a directory or `.py` file, while
`resolve_plugin_paths()` rejects non-directories.

In `cli.py`, replace the existing argument definition with formatted code like:

```python
parser.add_argument(
    "--plugin-path",
    action="append",
    default=[],
    metavar="DIRECTORY",
    help=(
        "Directory containing trusted ModelSpec or Workflow plugins. "
        "May be passed multiple times. Defaults to the current directory."
    ),
)
```

Short reason: the public CLI contract must match validation behavior.

Keep this option before the subcommand unless the parser is deliberately
redesigned. With the current parser, users write:

```bash
bayesflow-cli --plugin-path /workspace/plugins models list
```

not:

```bash
bayesflow-cli models list --plugin-path /workspace/plugins
```

### Step 4.2: Rename the workflow collector consistently

Change:

```python
def _collect_workflow_from_module(module) -> list:
```

to:

```python
def _collect_workflows_from_module(module) -> list[Workflow]:
```

Update both call sites.

Short reason: the function returns multiple workflows, and precise naming makes
the registry code easier to audit.

### Step 4.3: Correct misleading error messages

Inside the workflow collector, change messages referring to `MODEL_SPECS` or
`ModelSpec` so they refer to `WORKFLOWS` and `Workflow`:

```python
raise ValueError(
    f"All items in WORKFLOWS in {module.__name__} "
    "must be Workflow instances."
)
```

Also correct `returing` to `returning`.

Short reason: plugin authors need errors that identify the actual violated
contract.

### Step 4.4: Validate registration-function return values

Today `register_models()` and `register_workflow()` are checked only for
callability. Validate their returned collections using the same rules as
`MODEL_SPECS` and `WORKFLOWS`.

A small helper is clearer:

```python
def _validate_model_specs(value, source: str) -> list[ModelSpec]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{source} must provide a list or tuple")
    if not all(isinstance(item, ModelSpec) for item in value):
        raise ValueError(f"Every item from {source} must be a ModelSpec")
    return list(value)
```

Create an equivalent workflow validator and call it for both constant-based and
function-based registration.

Short reason: plugin entry styles should enforce the same interface.

### Step 4.5: Decide singular versus plural registration names

The marker is currently `register_workflow`, singular, even though the collector
expects an iterable. Choose one contract:

- preferred: rename to `register_workflows()` and return a collection; or
- keep `register_workflow()` and require it to return exactly one `Workflow`.

For consistency with `register_models()`, prefer:

```python
PLUGIN_MARKERS = {
    "MODEL_SPECS",
    "WORKFLOWS",
    "register_models",
    "register_workflows",
}
```

During one release, discovery may accept both names and issue a deprecation
warning for the singular spelling.

Short reason: a precise API reduces plugin-author mistakes.

### Step 4.6: Give imported plugins collision-resistant module names

The current loader uses only `py_file.stem`, so two plugin directories
containing `models.py` receive the same module name. Construct a deterministic
name from the resolved path, for example with a short SHA-256 digest:

```python
module_name = f"bayesflow_external_{py_file.stem}_{path_digest}"
mod_spec = spec_from_file_location(module_name, py_file)
```

Short reason: repeated common filenames should not create ambiguous module
identity in `sys.modules`, tracebacks, or serialization.

### Step 4.7: Document the security boundary

Add this warning to CLI/plugin documentation:

> Plugin files are imported as Python modules and execute with the permissions
> of `bayesflow-cli`. AST marker detection is an import filter, not a sandbox.
> Load only trusted plugins.

Short reason: a mounted Python plugin is executable code, not passive data.

### Step 4.8: Import each plugin module only once per CLI invocation

The current CLI calls model discovery and workflow discovery separately. Each
function scans and executes the same marked plugin files, so a registration
module may run twice. Introduce one registry result:

```python
@dataclass(frozen=True)
class PluginRegistry:
    models: dict[str, ModelSpec]
    workflows: dict[str, Workflow]


def discover_plugins(plugin_paths=None) -> PluginRegistry:
    models: dict[str, ModelSpec] = {}
    workflows = {
        workflow.name: workflow
        for workflow in BUILTIN_WORKFLOWS
    }

    # Resolve directories, load each marked module once, and collect both kinds.
    ...

    return PluginRegistry(models=models, workflows=workflows)
```

Keep thin compatibility wrappers temporarily if tests or other callers use the
old functions:

```python
def discover_model_specs(plugin_paths=None) -> dict[str, ModelSpec]:
    return discover_plugins(plugin_paths).models


def discover_workflows(plugin_paths=None) -> dict[str, Workflow]:
    return discover_plugins(plugin_paths).workflows
```

Within the CLI, call `discover_plugins()` once for commands needing both
registries. Do not call both wrappers consecutively, because that would still
import files twice.

Short reason: plugin imports can have expensive or stateful module-level effects;
one discovery pass makes execution deterministic and faster.

### Validation checkpoint B

Confirm tests cover:

- malformed `MODEL_SPECS`;
- malformed `WORKFLOWS`;
- invalid registration-function returns;
- duplicate models;
- duplicate workflows;
- sibling helper imports; and
- two plugin directories containing equally named files; and
- a registration module with an import counter executes only once per CLI
  invocation.

---

## 5. Make Empty and Missing Plugin Behavior Clear

### Step 5.1: Improve `models list` output

In `cli.py`, replace:

```python
print("No spec found.")
```

with a message that explains the runtime contract:

```python
print(
    "No model plugins found. "
    "Mount plugins in the working directory or pass --plugin-path DIRECTORY."
)
```

Keep the command successful with exit code zero.

Short reason: an empty registry is a valid state for a generic engine, not an
internal failure.

### Step 5.2: Centralize model and workflow resolution errors

Create small helpers in `cli.py` or a dedicated command-support module:

```python
def require_model(specs: dict[str, ModelSpec], name: str) -> ModelSpec:
    try:
        return specs[name]
    except KeyError as exc:
        available = ", ".join(sorted(specs)) or "none"
        raise ValueError(
            f"Unknown model {name!r}. Discovered models: {available}"
        ) from exc
```

Create an equivalent `require_workflow()` helper. Replace the repeated checks in
`train`, `resume`, and `recovery` branches.

Short reason: one error policy prevents different commands from giving
different guidance for the same missing plugin.

### Step 5.3: Discover only what each command needs

The current CLI discovers models before handling every command. Refactor dispatch
so:

- `models list` discovers models only;
- `checkpoints list` needs neither model nor workflow discovery;
- `train`, `resume`, and `recovery` discover both;
- any project-specific extension discovers only its own requirements.

Conceptually:

```python
if args.command == "models":
    specs = discover_model_specs(args.plugin_path)
    ...
    return

config = build_runtime_config(args)
setup_directories(config)

if args.command == "checkpoints":
    ...
    return

specs = discover_model_specs(args.plugin_path)
workflows = discover_workflows(args.plugin_path)
```

Short reason: `checkpoints list` should still work when a model plugin has been
temporarily removed, and commands should not execute unrelated plugin code.

### Step 5.4: Do not create every directory for read-only commands

Currently `setup_directories()` runs before `checkpoints list` and creates paths
including `real_data`. Separate directory requirements by command:

- `models list`: create nothing;
- `checkpoints list`: read the checkpoint directory and return an empty list if
  absent;
- `train`: create checkpoint, result, and log destinations;
- `resume` and `recovery`: validate required inputs, then create only output
  destinations.

Short reason: read-only inspection commands should not mutate a mounted
workspace.

### Validation checkpoint C

From an empty temporary working directory, verify:

```bash
bayesflow-cli models list
bayesflow-cli checkpoints list
```

Both should run without built-in model files and without creating unrelated
directories. Then pass the test plugin directory explicitly and verify that its
model appears.

---

## 6. Move Article-Specific Models into a Workspace Plugin

Do this only after discovery works without `internalpluginpath`.

### Step 6.1: Create a runtime workspace example outside the package

Create a directory that is not matched by
`[tool.setuptools.packages.find] include = ["bayesflow_models*"]`:

```text
examples/article_workspace/
├── plugins/
│   ├── article_models.py
│   └── reproduce_models.py
├── real_data/
├── checkpoints/
├── results/
└── logs/
```

The empty data/output directories do not need to be committed. Small `.gitkeep`
files are optional, but generated content should be ignored by Git and Docker.

Short reason: physical separation is the most reliable packaging boundary.

### Step 6.2: Move the active scientific implementation

Move:

```text
bayesflow_models/reproduce_models.py
```

to:

```text
examples/article_workspace/plugins/reproduce_models.py
```

Do not merely copy it and leave the original in the package; that would keep the
article model in the wheel.

Short reason: the build artifact should contain only engine-owned code.

### Step 6.3: Replace `trailwise_builtin.py` with an external registration file

Create `examples/article_workspace/plugins/article_models.py` containing the
active `MODEL_SPECS` definitions formerly located in
`internalpluginpath/trailwise_builtin.py`.

Change its model imports from package imports such as:

```python
from bayesflow_models.reproduce_models import ...
```

to sibling workspace imports:

```python
from reproduce_models import (
    PAR_NAMES_DC_REPRODUCED,
    PAR_NAMES_DF_REPRODUCED,
    PAR_NAMES_SF_REPRODUCED,
    get_adapter_trialwise_dc,
    get_adapter_trialwise_df,
    get_adapter_trialwise_sf,
    model_DC_reproduced,
    model_DF_reproduced,
    model_SF_reproduced,
)
```

Continue importing the stable interface from the installed engine:

```python
from bayesflow_models.interfaces import ModelSpec
```

Each model should continue to use:

```python
workflow="builtin"
```

Short reason: models belong to the researcher, while `ModelSpec` and the generic
workflow contract belong to the engine.

### Step 6.4: Remove unused imports from the registration file

The current `trailwise_builtin.py` imports symbols from
`DDM_DC_Pedestrain_TrialWise.py` even though the corresponding specs are
commented out. Do not carry those imports into `article_models.py` unless those
models are actually registered.

Short reason: import only active plugin dependencies; commented specifications
should not enlarge or destabilize runtime requirements.

### Step 6.5: Validate the mounted-workspace behavior locally

From the example workspace, test implicit current-directory discovery:

```bash
cd examples/article_workspace/plugins
bayesflow-cli models list
```

From the repository root, test explicit discovery:

```bash
bayesflow-cli \
    --plugin-path examples/article_workspace/plugins \
    models list
```

Expected models:

```text
model_DC
model_DF
model_SF
```

Short reason: these are the two plugin-path modes the container will expose.

### Step 6.6: Delete the obsolete internal model registry

After tests pass, delete:

```text
bayesflow_models/internalpluginpath/trailwise_builtin.py
```

After the explicit built-in workflow registry is active, also delete:

```text
bayesflow_models/internalpluginpath/workflow_builtin.py
bayesflow_models/internalpluginpath/
```

Short reason: leaving a second registry creates two sources of truth and invites
future accidental coupling.

### Validation checkpoint D

Search the engine package:

```bash
rg -n \
    "internalpluginpath|reproduce_models|model_DC_reproduced|model_DF_reproduced|model_SF_reproduced" \
    cli.py bayesflow_models pyproject.toml
```

The engine should not reference the article model registry. References inside
the external example workspace are expected.

---

## 7. Workstream 2: Split Evaluation Orchestration from Scientific Scenarios

### Step 7.1: Document the current mixed-TTA behavior

Files to inspect without modifying them in this step:

```text
bayesflow_models/reproduce_models.py
bayesflow_models/DDM_DC_Pedestrain_TrialWise.py
bayesflow_models/mixed_tta_evaluation.py
```

The active reproduced models choose one base TTA condition for each simulated
training dataset. Individual trials receive small jitter around that base:

```text
training dataset A: 2.53, 2.58, 2.51, ...
training dataset B: 3.04, 3.09, 3.01, ...
```

The posterior network sees a trial-level representation containing the response
and actual TTA. Mixed-TTA recovery supplies several base-condition clusters
inside one dataset:

```text
mixed test dataset: 2.53, 3.08, 4.04, 2.57, 3.52, ...
```

Therefore, mixed-TTA recovery is an out-of-training-distribution composition
test. It asks whether the network can combine evidence across heterogeneous
conditions even though training datasets were condition-homogeneous.

The current `bayesflow_models/mixed_tta_evaluation.py` is not a valid generic
runner because it contains both kinds of knowledge:

- generic artifact loading, posterior sampling, metrics, plots, and reports;
- model-specific TTA values, four parameter priors, and a duplicated DDM
  simulator.

It also generates the four-parameter simplest model while the active reproduced
DC, DF, and SF models use different parameter sets. The split must remove this
mismatch, not merely relocate the file.

Short reason: the refactor must preserve the scientific meaning of the test.

### Step 7.2: Create the engine evaluation package

Create these files:

```text
bayesflow_models/evaluation/
├── __init__.py
├── interfaces.py
├── runner.py
├── metrics.py
└── reporting.py
```

Responsibilities:

- `interfaces.py`: stable contracts shared with researcher plugins;
- `runner.py`: artifact resolution, model loading, scenario invocation, posterior
  sampling, validation, and result coordination;
- `metrics.py`: model-independent parameter-recovery metrics;
- `reporting.py`: model-independent files, plots, and metadata reports;
- `__init__.py`: re-export only the public evaluation types.

Short reason: separate scientific data generation from reusable evaluation
mechanics.

### Step 7.3: Define one small evaluation batch contract

In `bayesflow_models/evaluation/interfaces.py`, add:

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationBatch:
    posterior_conditions: dict[str, Any]
    true_parameters: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
```

Meaning:

- `posterior_conditions` is the exact structure accepted by the saved posterior
  network's `sample()` method;
- `true_parameters` contains the ground-truth values used for recovery metrics;
- `metadata` records scenario facts such as condition values, trial counts, and
  whether trials were shuffled.

Short reason: the engine needs one model-neutral object from every evaluation
scenario.

Do not name fields `x` or `tta_per_trial` in the engine interface. Those are
properties of this research model, not universal cognitive-model concepts.

### Step 7.4: Define the evaluation-scenario registration contract

In the same file, add:

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

Use the pair `(model_name, name)` as the registry key. This allows all three
models to register an evaluation named `mixed-tta` without colliding:

```text
(model_DC, mixed-tta)
(model_DF, mixed-tta)
(model_SF, mixed-tta)
```

Short reason: an evaluation scenario must be tied to the model whose posterior
and simulator semantics it matches.

### Step 7.5: Export the public evaluation API

In `bayesflow_models/evaluation/__init__.py`, add:

```python
from .interfaces import EvaluationBatch, EvaluationSpec

__all__ = ["EvaluationBatch", "EvaluationSpec"]
```

Researcher plugins should import these public types rather than engine-internal
runner functions.

Short reason: a small public API allows engine internals to change without
breaking plugins.

### Step 7.6: Extend plugin marker detection

In `bayesflow_models/discovery.py`, extend `PLUGIN_MARKERS`:

```python
PLUGIN_MARKERS = {
    "MODEL_SPECS",
    "WORKFLOWS",
    "EVALUATIONS",
    "register_models",
    "register_workflows",
    "register_evaluations",
}
```

If Workstream 1 temporarily retains `register_workflow`, keep it during the
deprecation period as well.

Short reason: the existing AST prefilter otherwise skips evaluation registration
files.

### Step 7.7: Collect and validate evaluation registrations

In `bayesflow_models/discovery.py`, import the public type:

```python
from bayesflow_models.evaluation import EvaluationSpec
```

Add a validator similar to the model and workflow validators:

```python
def _validate_evaluations(value, source: str) -> list[EvaluationSpec]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{source} must provide a list or tuple")
    if not all(isinstance(item, EvaluationSpec) for item in value):
        raise ValueError(f"Every item from {source} must be an EvaluationSpec")
    return list(value)
```

Add `_collect_evaluations_from_module()` supporting either:

```python
EVALUATIONS = [...]
```

or:

```python
def register_evaluations():
    return [...]
```

Short reason: models, workflows, and evaluations should use the same predictable
registration pattern.

### Step 7.8: Add evaluations to the single-pass plugin registry

Extend `PluginRegistry` in `bayesflow_models/discovery.py`:

```python
@dataclass(frozen=True)
class PluginRegistry:
    models: dict[str, ModelSpec]
    workflows: dict[str, Workflow]
    evaluations: dict[tuple[str, str], EvaluationSpec]
```

Within `discover_plugins()`, load each marked module once and collect all three
registration types. Reject duplicate `(model_name, evaluation_name)` keys.

Short reason: a plugin module must not execute separately for model discovery
and evaluation discovery.

### Step 7.9: Add a generic evaluation runner

In `bayesflow_models/evaluation/runner.py`, add a function with a narrow
responsibility:

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

Implement it in this order:

1. Resolve artifact metadata through the existing artifact API.
2. Check that the artifact belongs to `model_spec.name`.
3. Check that `evaluation_spec.model_name == model_spec.name`.
4. Resolve and verify the checkpoint path.
5. Call `evaluation_spec.generate_batch(model_spec, options)`.
6. Validate the returned `EvaluationBatch`.
7. Load the saved posterior network.
8. Call `sample()` with `batch.posterior_conditions`.
9. Pass posterior samples and true parameters to generic metrics/reporting.
10. Return a serializable report dictionary.

Short reason: the runner owns execution order but knows no TTA values or model
equations.

### Step 7.10: Validate a scenario batch before sampling

In `runner.py`, add checks before loading or sampling a large model:

```python
expected = set(model_spec.par_names)
actual = set(batch.true_parameters)
if actual != expected:
    raise ValueError(
        f"Evaluation targets {sorted(actual)} do not match "
        f"model parameters {sorted(expected)}"
    )
```

Also check:

- `posterior_conditions` is not empty;
- every target has the same leading batch dimension;
- the scenario returns at least one simulated dataset; and
- the metadata can be serialized to JSON.

Short reason: this catches the current four-parameter-versus-five/seven-parameter
mismatch before posterior sampling.

### Step 7.11: Move generic metrics into the engine

In `bayesflow_models/evaluation/metrics.py`, move and generalize:

- posterior mean and standard deviation;
- bias;
- RMSE;
- Pearson correlation when defined; and
- interval coverage.

Iterate over `model_spec.par_names` or the validated target keys. Remove the
module-level hard-coded `PARAMETER_NAMES` list.

Short reason: these operations depend on posterior and target arrays, not on the
scientific meaning of TTA.

### Step 7.12: Move generic reporting into the engine

In `bayesflow_models/evaluation/reporting.py`, move generalized versions of:

- JSON report writing;
- NPZ sample/target writing;
- recovery-plot creation; and
- output-directory creation.

Use a generic directory:

```text
results/<artifact-id>/evaluations/<evaluation-name>/
```

Include `EvaluationBatch.metadata` in the report instead of hard-coding fields
such as `trials_per_tta` in the engine.

Short reason: the engine controls output consistency while scenarios control
scientific metadata.

### Step 7.13: Add a generic CLI command

In `cli.py`, add:

```text
evaluate
```

with generic arguments:

```python
evaluate_parser = subparsers.add_parser(
    "evaluate",
    help="Evaluate a trained artifact with a registered scenario",
)
evaluate_parser.add_argument("--artifact", required=True)
evaluate_parser.add_argument("--scenario", required=True)
evaluate_parser.add_argument("--n-test-sims", type=int, default=100)
evaluate_parser.add_argument("--n-posterior-samples", type=int, default=1000)
evaluate_parser.add_argument("--seed", type=int, default=2026)
evaluate_parser.add_argument(
    "--scenario-option",
    action="append",
    default=[],
    metavar="KEY=VALUE",
)
```

The initial command becomes:

```bash
bayesflow-cli \
    --plugin-path /workspace/plugins \
    evaluate \
    --artifact run-001 \
    --scenario mixed-tta \
    --scenario-option trials_per_tta=15 \
    --scenario-option shuffle=true
```

Short reason: `evaluate` is portable; `mixed-tta` is a runtime capability
provided by a plugin.

For the first implementation, parse only documented scalar option types and
reject malformed `KEY=VALUE` entries. Do not use `eval()`.

### Step 7.14: Dispatch the generic evaluation command

In `cli.py`, in the `evaluate` branch:

1. Discover plugins once.
2. Resolve the artifact to learn its `spec_name`.
3. Find that `ModelSpec` in `registry.models`.
4. Look up `(spec.name, args.scenario)` in `registry.evaluations`.
5. Convert scenario options to a validated dictionary.
6. Call `run_evaluation()`.
7. Print the output directory and a concise summary.

If the key is missing, list scenarios available for that model.

Short reason: command dispatch selects capabilities; it does not implement their
scientific behavior.

### Step 7.15: Remove the old engine command only after replacement works

In `cli.py`, remove the old `mixed-recovery` parser and dispatch branch after
`evaluate --scenario mixed-tta` passes equivalence tests.

In `build_runtime_config()`, remove mixed-TTA-specific keys such as
`trials_per_tta` and `shuffle_tta`; these now live in scenario options.

Short reason: eliminate domain terminology from the generic CLI without losing
the feature during migration.

### Step 7.16: Refactor model simulation into reusable scientific kernels

This step is in the researcher workspace file:

```text
examples/article_workspace/plugins/reproduce_models.py
```

For each active model, extract a function that simulates one trial from explicit
parameters and one explicit TTA value. For example:

```python
def simulate_dc_trial(
    *,
    parameters: dict,
    tta: float,
    rng: np.random.Generator,
    dt: float = 0.005,
) -> float:
    ...
```

Then change the homogeneous training simulator to call this kernel repeatedly
with one base TTA plus per-trial jitter.

Short reason: training and evaluation must use the same equations rather than
duplicating a simulator in `mixed_tta_evaluation.py`.

Create equivalent kernels for DF and SF because their boundary/drift equations
differ.

### Step 7.17: Make parameter sampling reusable and seedable

In the same workspace file, add model-specific parameter sampling functions that
accept an RNG:

```python
def sample_dc_parameters(rng: np.random.Generator) -> dict:
    ...
```

Make the existing BayesFlow prior function delegate to it using the module RNG:

```python
def prior_DC() -> dict:
    return sample_dc_parameters(RNG)
```

Create matching functions for DF and SF.

Short reason: the evaluation scenario needs the exact training priors with a
reproducible local seed.

### Step 7.18: Create the workspace mixed-TTA scenario

Create:

```text
examples/article_workspace/plugins/mixed_tta_scenarios.py
```

This file should import:

- `EvaluationBatch` and `EvaluationSpec` from the installed engine;
- model specs or stable model names from `article_models.py`; and
- trial kernels and parameter samplers from `reproduce_models.py`.

Implement one generator per model, or one carefully dispatched generator:

```python
def generate_dc_mixed_tta(
    model_spec: ModelSpec,
    options: dict,
) -> EvaluationBatch:
    ...
```

For every simulated dataset:

1. Sample one parameter vector.
2. Repeat each requested base TTA `trials_per_tta` times.
3. Optionally shuffle the base-condition sequence.
4. Add per-trial jitter using the same rule as training.
5. Simulate every trial with the same participant parameter vector.
6. Construct the posterior conditions expected by the saved network.
7. Store the exact true parameter vector.
8. Record condition values, counts, shuffle policy, and seed in metadata.

Short reason: only the researcher plugin knows how to produce scientifically
valid mixed-condition data for its model.

### Step 7.19: Reuse the model's adapter contract

Inside `mixed_tta_scenarios.py`, do not assume every network accepts raw keys
named `x` and `tta_per_trial`. Use the same adapter contract as training to
produce `posterior_conditions`.

Add a small focused test that compares the keys, ranks, and feature dimensions
produced by:

- one normal training simulation passed through the model adapter; and
- one mixed-TTA evaluation batch prepared for posterior sampling.

Short reason: sharing equations is insufficient if evaluation sends a different
tensor contract to the posterior network.

The present reproduced simulator has one response column, so concatenating TTA
produces two trial features. Do not copy stale comments claiming three features
unless CPP is actually restored as a second observation column.

### Step 7.20: Register model-bound scenarios

At the bottom of `mixed_tta_scenarios.py`, add:

```python
EVALUATIONS = [
    EvaluationSpec(
        name="mixed-tta",
        model_name="model_DC",
        generate_batch=generate_dc_mixed_tta,
        description="Mixed-TTA recovery after homogeneous-TTA training",
    ),
    EvaluationSpec(
        name="mixed-tta",
        model_name="model_DF",
        generate_batch=generate_df_mixed_tta,
        description="Mixed-TTA recovery after homogeneous-TTA training",
    ),
    EvaluationSpec(
        name="mixed-tta",
        model_name="model_SF",
        generate_batch=generate_sf_mixed_tta,
        description="Mixed-TTA recovery after homogeneous-TTA training",
    ),
]
```

Short reason: scenario registration makes the capability discoverable without
putting scientific code in the image.

### Step 7.21: Remove the obsolete combined module

After generic-runner and scenario tests pass, delete:

```text
bayesflow_models/mixed_tta_evaluation.py
```

Do not move it unchanged to the workspace. Its generic portions now belong to
the evaluation package, and its duplicated simplest-model simulation should be
replaced by the active model kernels.

Short reason: retaining the combined module would preserve two sources of truth.

### Step 7.22: Add simple evaluation tests

Create:

```text
tests/
├── test_evaluation_discovery.py
├── test_evaluation_runner.py
└── test_mixed_tta_scenarios.py
```

Test these cases separately:

1. `EVALUATIONS` is discovered from a plugin directory.
2. Duplicate model/scenario keys fail clearly.
3. Unknown scenarios list available choices.
4. An artifact cannot be evaluated with another model's scenario.
5. Target keys must exactly match `ModelSpec.par_names`.
6. Target arrays share one batch dimension.
7. Training and evaluation conditions have compatible network input shapes.
8. A fixed seed reproduces the same evaluation batch.
9. A mixed batch contains every requested base condition.
10. Within one mixed dataset, one parameter vector is reused across all trials.
11. DC, DF, and SF scenarios use their corresponding scientific kernels.
12. The generic runner works with a fake non-TTA scenario.

Short reason: the last test proves that the engine abstraction is genuinely more
general than this article's mixed-TTA use case.

### Validation checkpoint E

Before continuing, verify:

```bash
bayesflow-cli \
    --plugin-path examples/article_workspace/plugins \
    evaluate \
    --artifact TEST_ARTIFACT \
    --scenario mixed-tta \
    --n-test-sims 2 \
    --n-posterior-samples 5 \
    --scenario-option trials_per_tta=2
```

Pass conditions:

- the engine imports no article simulator or TTA constant;
- the workspace scenario imports the scientific kernels;
- target names exactly match the selected model;
- posterior input shapes match training input shapes;
- output is stored under the mounted results directory; and
- the report labels the test as out-of-training-distribution.

Short reason: this proves both sides of the evaluation contract together.

---

## 8. Audit and Remove Legacy Modules

Removal must follow dependency evidence, not filename intuition.

### Step 8.1: Audit `reproduce_models_utility.py`

The current active Python import graph does not import this file; its reference
in `trailwise_builtin.py` is commented out.

Search all source, tests, notebooks, and documentation:

```bash
rg -n "reproduce_models_utility" . \
    -g '!bayesflow_env/**' \
    -g '!*.whl'
```

If no supported workflow imports it, move it temporarily to an archive branch or
delete it in a dedicated commit.

Short reason: isolated cleanup commits make restoration and review easy.

### Step 8.2: Audit `workflow_trialwise.py` and `main_workflow.py`

`main_workflow.py` currently imports `workflow_trialwise.py`; the packaged CLI
does not. Decide whether this legacy entrypoint is supported.

If the answer is no, delete both together after confirming that notebooks and
lab procedures do not use them.

Short reason: deleting only the imported module leaves a visibly broken script.

If the answer is yes, move both to the article workspace and remove imports of
engine-internal modules that should instead be accessed through documented
public interfaces.

### Step 8.3: Audit other article-specific modules

Classify every file currently inside `bayesflow_models/`:

```text
engine core
generic optional feature
article workspace plugin/helper
legacy/unused
```

At minimum, review:

```text
DDM_DC_Pedestrain.py
DDM_DC_Pedestrain_TrialWise.py
models.py
models_add_conditions.py
reproduce_models_utility.py
utils_real_data.py
workflow_trialwise.py
mixed_tta_evaluation.py
```

Move article-specific modules outside `bayesflow_models/`. Replace the combined
`mixed_tta_evaluation.py` with the generic engine evaluation package and the
workspace scenario described in Workstream 2. Delete only confirmed legacy
modules.

Short reason: package auto-discovery currently includes every module under
`bayesflow_models`, even when the CLI never imports it.

### Step 8.4: Keep these engine modules initially

Based on the current CLI, retain:

```text
bayesflow_models/__init__.py
bayesflow_models/interfaces.py
bayesflow_models/discovery.py
bayesflow_models/builtin_workflows.py
bayesflow_models/workflow.py
bayesflow_models/train.py
bayesflow_models/evaluation/__init__.py
bayesflow_models/evaluation/interfaces.py
bayesflow_models/evaluation/runner.py
bayesflow_models/evaluation/metrics.py
bayesflow_models/evaluation/reporting.py
```

Do not remove `workflow.py`: the CLI uses its artifact lookup/listing functions,
and the default workflow uses its train/resume/recovery handlers.

Short reason: portability means removing domain coupling, not removing required
engine behavior.

### Validation checkpoint F

After each removal:

1. search imports with `rg`;
2. run unit tests;
3. build the wheel;
4. install the wheel in a clean environment;
5. run CLI help, empty listing, explicit plugin listing, and a tiny workflow
   smoke test.

---

## 9. Clean the Package Interface

The following changes improve maintainability but should come after the
behavioral separation above.

### Step 9.1: Move the CLI module inside the package

Move:

```text
cli.py
```

to:

```text
bayesflow_models/cli.py
```

Update `pyproject.toml`:

```toml
[project.scripts]
bayesflow-cli = "bayesflow_models.cli:main"
```

Then remove:

```toml
[tool.setuptools]
py-modules = ["cli"]
```

Short reason: keeping application modules in one importable package makes wheel
contents and ownership clearer.

### Step 9.2: Consider renaming the engine package

`bayesflow_models` sounds like it owns models, while the target engine explicitly
does not. A later breaking release could rename it to something like:

```text
bayesflow_cli/
```

or a project-specific non-conflicting name.

Do not combine this rename with discovery behavior changes. Perform it only
after tests prove the new architecture.

Short reason: separate behavior changes from mechanical renames to keep reviews
understandable.

### Step 9.3: Define a small public plugin API

Re-export only stable plugin-facing types from the package root:

```python
from .evaluation import EvaluationBatch, EvaluationSpec
from .interfaces import ModelSpec, Workflow, WorkflowContext, WorkflowResult

__all__ = [
    "EvaluationBatch",
    "EvaluationSpec",
    "ModelSpec",
    "Workflow",
    "WorkflowContext",
    "WorkflowResult",
]
```

External plugins could then import:

```python
from bayesflow_models import ModelSpec
```

Short reason: plugin authors should not depend on arbitrary internal module
layout.

### Step 9.4: Align callable type aliases with actual use

The current `Workflow` aliases in `interfaces.py` should match the actual calls
made by `cli.py`. The CLI passes `(spec, config)` to train, resume, and recovery
handlers, while some aliases currently describe strings or an additional
context argument.

Update them to the implemented contract, for example:

```python
TrainFn = Callable[[ModelSpec, dict[str, Any]], WorkflowResult]
ResumeFn = Callable[[ModelSpec, dict[str, Any]], WorkflowResult]
RecoveryFn = Callable[[ModelSpec, dict[str, Any]], WorkflowResult]
```

Then add static type checking if desired.

Short reason: an inaccurate interface annotation is dangerous precisely where
third-party plugins depend on it.

---

## 10. Make Packaging Define the Build Input

### Step 10.1: Keep package discovery narrow

After article modules are moved outside the package, this rule:

```toml
[tool.setuptools.packages.find]
include = ["bayesflow_models*"]
```

should include only engine code. Consider adding an explicit exclusion as a
defense:

```toml
[tool.setuptools.packages.find]
include = ["bayesflow_models*"]
exclude = ["tests*", "examples*"]
```

Short reason: the wheel, rather than Docker-specific ignore tricks, should
define the portable Python application.

### Step 10.2: Update project metadata

Change the description from article-specific wording to the generic engine's
actual purpose. Replace placeholder license and author metadata with verified
values before distribution.

Short reason: wheel metadata is part of the distributed contract.

### Step 10.3: Revisit dependencies after moving features

For each dependency in `pyproject.toml`, use imports and runtime tests to decide
whether it belongs to:

- mandatory engine dependencies;
- an optional engine extra;
- an article workspace/plugin dependency; or
- development dependencies.

Do not remove a dependency merely because one direct import disappeared;
BayesFlow or another retained dependency may require it transitively. Conversely,
do not rely accidentally on a transitive dependency that the engine imports
directly.

Short reason: the package must declare what its own code imports, while plugin
dependencies may need their own environment contract.

### Step 10.4: Build and inspect the final wheel

Build into a clean temporary output directory:

```bash
python -m pip wheel . \
    --no-deps \
    --no-build-isolation \
    --wheel-dir /tmp/bayesflow-wheel-after

python -m zipfile -l /tmp/bayesflow-wheel-after/*.whl
```

The wheel should contain engine modules and metadata. It should not contain:

```text
internalpluginpath/
reproduce_models.py
reproduce_models_utility.py
DDM_DC_Pedestrain*.py
workflow_trialwise.py
mixed_tta_evaluation.py
article workspace plugins
notebooks
checkpoints
results
logs
real data
```

Short reason: inspecting the artifact verifies the build boundary directly.

### Step 10.5: Test the artifact outside the repository

Install the wheel into a clean environment, change to an empty temporary
directory, and run:

```bash
bayesflow-cli --help
bayesflow-cli models list
bayesflow-cli checkpoints list
```

Then point it to the example plugin directory and verify the three article
models and their model-bound evaluation scenarios appear. Use a fake checkpoint
or a tiny validated artifact to exercise the generic evaluation runner.

Short reason: commands that work only from the repository have hidden source
path dependencies and are not yet portable.

---

## 11. Give Docker a Clean Build Context

Only after the wheel boundary passes should Dockerization begin.

### Step 11.1: Define the minimal Docker build inputs

The Dockerfile should initially copy only:

```text
pyproject.toml
README.md
bayesflow_models/
```

If `cli.py` has not yet been moved into the package, it must also be copied.

Short reason: explicit copying provides a visible allowlist of application build
inputs.

### Step 11.2: Add `.dockerignore`

At minimum exclude:

```text
.git/
.agents/
.codex/
.vscode/
bayesflow_env/
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.pytest_cache/
.ruff_cache/
.ipynb_checkpoints/
tests/
examples/
notebooks/
real_data/
checkpoints/
trained_model*/
results/
logs/
*.keras
```

Do not use `.dockerignore` to exclude selected modules from inside an otherwise
unclean Python package. Move those modules outside the package first.

Short reason: packaging defines application membership; `.dockerignore` limits
context transfer and accidental disclosure.

### Step 11.3: Keep runtime plugins out of the image

Do not `COPY examples/article_workspace/plugins` into the image. Mount the
workspace when running:

```text
host experiment workspace -> /workspace
```

Use either implicit discovery:

```bash
docker run --rm \
    --mount type=bind,src=HOST_WORKSPACE,dst=/workspace \
    IMAGE models list
```

or an explicit directory:

```bash
docker run --rm \
    --mount type=bind,src=HOST_WORKSPACE,dst=/workspace \
    IMAGE --plugin-path /workspace/plugins models list
```

Short reason: the same image can execute different trusted research plugins
without being rebuilt.

### Step 11.4: Validate both sides of the boundary

Without a mount:

- `bayesflow-cli --help` works;
- `models list` reports no model plugins;
- the image contains the generic `builtin` workflow; and
- the image contains generic evaluation interfaces and the runner; and
- no article model modules exist in the filesystem.

With a workspace mount:

- DC, DF, and SF models are discovered;
- their `mixed-tta` scenarios are discovered;
- training selects the generic `builtin` workflow;
- `evaluate --scenario mixed-tta` obtains scientific data from the workspace
  scenario while orchestration remains in the engine;
- outputs appear on the host; and
- deleting the container does not delete those outputs.

Short reason: portability is proven only when one immutable image works with
multiple replaceable runtime workspaces.

---

## 12. Recommended Order of Small Commits

Keep each commit independently understandable:

1. Add discovery and CLI boundary tests.
2. Add `builtin_workflows.py` without changing behavior.
3. Make workflow discovery use the explicit built-in registry.
4. Make model discovery external-only.
5. Correct plugin argument help, validation, names, and duplicate handling.
6. Make empty registry behavior and command-specific discovery explicit.
7. Add the external article workspace plugin while retaining old files briefly.
8. Prove equivalent model discovery from the external plugin.
9. Remove `internalpluginpath` and the duplicate packaged article models.
10. Add evaluation interfaces and a fake scenario test.
11. Extend single-pass discovery to collect model-bound evaluations.
12. Add the generic evaluation runner, metrics, and reporting.
13. Add the generic `evaluate --scenario` CLI command.
14. Extract reusable, seedable DC, DF, and SF scientific kernels and priors.
15. Implement and register workspace `mixed-tta` scenarios for each model.
16. Prove training/evaluation input compatibility and parameter-name equality.
17. Remove the old combined `mixed_tta_evaluation.py` and old CLI command.
18. Remove confirmed legacy files one logical group at a time.
19. Move `cli.py` into the package and update the console-script entrypoint.
20. Clean metadata and dependency declarations.
21. Inspect and clean-install the final wheel.
22. Add `.dockerignore`, CPU Dockerfile, and Compose only after packaging passes.

Short reason: small commits make regressions easy to locate and architectural
choices easy to review.

---

## 13. Definition of Done

The refactor is complete when all of these statements are true:

- [ ] The installed wheel contains no article-specific model implementation.
- [ ] `discovery.py` never imports or scans an internal model-plugin directory.
- [ ] The generic `builtin` workflow is registered explicitly by engine code.
- [ ] `models list` succeeds meaningfully with no external plugins.
- [ ] `checkpoints list` does not require model plugin discovery.
- [ ] Explicit and current-directory plugin discovery both work.
- [ ] External plugins can import sibling helper modules.
- [ ] Duplicate and malformed plugin registrations fail clearly.
- [ ] The documentation states that plugins execute trusted Python code.
- [ ] Article DC, DF, and SF models work from an external workspace plugin.
- [ ] The wheel contains generic evaluation interfaces, runner, metrics, and
      reporting.
- [ ] Evaluation scenarios are registered by `(model_name, scenario_name)`.
- [ ] The generic evaluation engine contains no TTA constants, article priors,
      or DC/DF/SF equations.
- [ ] Each workspace mixed-TTA scenario reuses the same scientific kernel and
      prior as its corresponding training model.
- [ ] Evaluation target names exactly match `ModelSpec.par_names`.
- [ ] Training and evaluation produce compatible posterior-input structures.
- [ ] The old combined `mixed_tta_evaluation.py` no longer exists.
- [ ] A fake non-TTA scenario passes through the generic runner.
- [ ] Confirmed legacy modules have been removed or moved outside the package.
- [ ] The wheel installs and runs from outside the repository.
- [ ] Docker builds from engine source and metadata only.
- [ ] The same image works with an empty workspace and with the article
      workspace.
- [ ] Durable outputs exist outside the disposable container layer.

The central design rule is:

> Docker should package the already-clean Python application boundary. Docker
> should not be used to hide unresolved coupling inside the Python package.
