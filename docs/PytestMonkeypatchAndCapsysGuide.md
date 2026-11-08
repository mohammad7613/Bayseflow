# Understanding `monkeypatch` and `capsys` in CLI Tests

`monkeypatch` and `capsys` are built-in pytest fixtures. They let a test control
external dependencies and observe terminal output without permanently changing
the Python process.

They are especially useful for CLI tests because a CLI normally depends on:

- `sys.argv`;
- filesystem paths;
- plugin discovery;
- artifact lookup;
- workflow functions;
- imported modules; and
- standard output and standard error.

The objective of `tests/test_cli_models.py` is to test CLI dispatch, not to
perform real training.

## The Overall Test Boundary

For a command such as:

```bash
bayesflow-cli train \
    --models cli_test_model \
    --epochs 2
```

the test should verify:

```text
command-line arguments
        |
        v
argument parser
        |
        v
model and workflow selection
        |
        v
runtime configuration
        |
        v
correct workflow function called
```

The baseline test should not:

- scan real research plugins;
- create real directories;
- simulate data;
- train a neural network;
- write checkpoints; or
- produce recovery plots.

`monkeypatch` temporarily replaces those expensive or external operations with
controlled fakes. `capsys` captures what the CLI prints so the test can inspect
it.

## 1. What Is `monkeypatch`?

`monkeypatch` temporarily changes Python objects or process state during one
test.

For example:

```python
def test_something(monkeypatch):
    monkeypatch.setenv("MODE", "test")
```

Pytest automatically restores the original environment after the test, even if
the test fails.

Conceptually:

```text
original state
      |
      v
pytest starts test
      |
      v
monkeypatch changes state
      |
      v
test executes
      |
      v
pytest restores original state
```

You do not import or create `monkeypatch`. Pytest provides it because it appears
as a test parameter:

```python
def test_train(monkeypatch):
    ...
```

## 2. `monkeypatch.setattr()`

`setattr()` temporarily replaces an attribute on an object.

The general form is:

```python
monkeypatch.setattr(
    target_object,
    "attribute_name",
    replacement_value,
)
```

For example:

```python
monkeypatch.setattr(
    cli,
    "setup_directories",
    lambda config: None,
)
```

During the test, this changes:

```text
cli.setup_directories
        |
        v
lambda config: None
```

Calling this during the test:

```python
cli.setup_directories(config)
```

does nothing. After the test, pytest restores the real function.

### Why patch `cli.setup_directories`?

The real implementation creates directories:

```python
Path(path).mkdir(
    parents=True,
    exist_ok=True,
)
```

A CLI dispatch test should not modify the filesystem. Directory creation should
have its own focused test. Replacing it with a no-op removes that side effect.

## 3. Replacing Plugin Discovery

The shared test helper contains:

```python
monkeypatch.setattr(
    cli,
    "discover_plugins",
    lambda plugin_paths: registry,
)
```

This replaces real plugin discovery with a controlled registry.

Normally:

```python
registry = discover_plugins(
    plugin_paths=args.plugin_path
)
```

would:

- scan directories;
- parse Python files;
- import plugins;
- validate registrations; and
- construct model and workflow registries.

Those behaviors are tested separately in `tests/test_discovery.py`.

In a CLI dispatch test, the test supplies a known registry:

```python
PluginRegistry(
    models={
        "cli_test_model": spec,
    },
    workflows={
        "builtin": workflow,
    },
)
```

The replacement:

```python
lambda plugin_paths: registry
```

accepts the expected argument but always returns the prepared registry.

This separates responsibilities:

```text
test_discovery.py
└── Does real plugin discovery work?

test_cli_models.py
└── Does the CLI use a returned registry correctly?
```

If both tests used real discovery, a discovery defect could make the CLI test
fail even when CLI dispatch was correct.

## 4. Replacing `sys.argv`

A CLI obtains terminal arguments from:

```python
sys.argv
```

For this command:

```bash
bayesflow-cli train --models model_DC
```

`sys.argv` resembles:

```python
[
    "bayesflow-cli",
    "train",
    "--models",
    "model_DC",
]
```

The test replaces it:

```python
monkeypatch.setattr(
    sys,
    "argv",
    [
        "bayesflow-cli",
        "train",
        "--models",
        "cli_test_model",
    ],
)
```

When `cli.main()` calls:

```python
args = parser.parse_args()
```

`argparse` reads the temporary list. After the test, pytest restores the
original `sys.argv`.

### A future test-friendly alternative

The current function reads directly from process state:

```python
def main():
    args = parser.parse_args()
```

A future design could accept arguments explicitly:

```python
def main(argv=None):
    args = parser.parse_args(argv)
```

Tests could then call:

```python
cli.main(
    [
        "train",
        "--models",
        "cli_test_model",
    ]
)
```

That would reduce the need to patch `sys.argv`, but the current approach is
valid.

## 5. Patching Artifact Resolution

Resume and recovery commands call:

```python
cli.wf.resolve_artifact_ref(
    ref,
    config,
)
```

The real implementation reads checkpoint metadata files. The test replaces it:

```python
monkeypatch.setattr(
    cli.wf,
    "resolve_artifact_ref",
    lambda ref, config: metadata,
)
```

Artifact lookup now returns controlled metadata:

```python
metadata = {
    "artifact_id": "artifact-001",
    "spec_name": "cli_test_model",
}
```

This allows the test to verify:

```text
artifact metadata
        |
        v
spec_name extracted
        |
        v
correct model selected
        |
        v
correct workflow handler called
```

It does not test metadata-file reading. That belongs in workflow or artifact
tests.

### A fake that checks its input

The resume test can use:

```python
def fake_resolve_artifact_ref(ref, config):
    assert ref == "run-001"
    return metadata
```

This fake performs two jobs:

1. It supplies controlled output.
2. It verifies that the CLI passed the expected artifact reference.

This is more informative than a lambda when the arguments matter.

## 6. Patch Where the Function Is Used

This is an important general rule.

Suppose `cli.py` contains:

```python
from bayesflow_models.discovery import discover_plugins
```

This creates a reference named:

```python
cli.discover_plugins
```

Patch that reference:

```python
monkeypatch.setattr(
    cli,
    "discover_plugins",
    fake_discovery,
)
```

Do not patch only:

```python
bayesflow_models.discovery.discover_plugins
```

because `cli` may already hold its own imported reference.

In contrast, `cli.py` contains:

```python
import bayesflow_models.workflow as wf
```

and calls:

```python
wf.resolve_artifact_ref(...)
```

Therefore, patching this works:

```python
monkeypatch.setattr(
    cli.wf,
    "resolve_artifact_ref",
    fake_resolver,
)
```

The rule is:

> Patch the name at the location where the tested code looks it up.

## 7. What Is `monkeypatch.setitem()`?

`setitem()` temporarily changes one item inside a mapping.

The general form is:

```python
monkeypatch.setitem(
    mapping,
    key,
    temporary_value,
)
```

The mixed-recovery test uses:

```python
monkeypatch.setitem(
    sys.modules,
    "bayesflow_models.mixed_tta_evaluation",
    fake_evaluation_module,
)
```

### What is `sys.modules`?

`sys.modules` is Python's registry of imported modules:

```python
{
    "sys": <module sys>,
    "pathlib": <module pathlib>,
    "cli": <module cli>,
}
```

When Python executes:

```python
from bayesflow_models.mixed_tta_evaluation import (
    evaluate_mixed_tta_artifact_with_bf_recovery,
)
```

it first checks whether the module name already exists in `sys.modules`.

The test constructs a fake module:

```python
fake_evaluation_module = ModuleType(
    "bayesflow_models.mixed_tta_evaluation"
)
```

It gives the module the expected function:

```python
fake_evaluation_module.evaluate_mixed_tta_artifact_with_bf_recovery = (
    fake_evaluate
)
```

It then registers the fake module temporarily:

```python
monkeypatch.setitem(
    sys.modules,
    "bayesflow_models.mixed_tta_evaluation",
    fake_evaluation_module,
)
```

When the CLI performs its internal import, Python returns the fake module instead
of importing the real file.

### Why is that needed here?

The current real mixed-TTA module imports an article-specific model module
removed during Workstream 1. Importing it would fail before CLI dispatch could
be tested.

The baseline test asks a narrower question:

> If an evaluator is available, does the CLI forward the correct arguments to
> it?

The fake module isolates that behavior. After the test, pytest restores the
previous `sys.modules` entry or removes the temporary entry if none existed.

## 8. Recording Fake-Function Calls

Tests often use a list:

```python
calls = []
```

The fake records how it was called:

```python
def fake_train(spec, config):
    calls.append((spec, config))
```

After `cli.main()`:

```python
assert len(calls) == 1

spec, config = calls[0]

assert spec.name == "cli_test_model"
assert config["training"]["epochs"] == 2
```

This verifies:

- the function was called;
- it was called exactly once;
- the correct model was passed; and
- the parsed options reached the runtime configuration.

The fake does not implement training. It observes dispatch.

A formal mock library could do the same:

```python
from unittest.mock import Mock

fake_train = Mock()
```

Then:

```python
fake_train.assert_called_once()
```

For simple tests, a list is often easier to understand.

## 9. What Is `capsys`?

`capsys` is a built-in pytest fixture that captures text written to:

- standard output, `stdout`; and
- standard error, `stderr`.

Standard output includes ordinary `print()` calls:

```python
print("Training complete")
```

A test receives the fixture as an argument:

```python
def test_models_list(
    monkeypatch,
    capsys,
):
    ...
```

After calling the CLI:

```python
cli.main()
```

read the captured output:

```python
captured = capsys.readouterr()
```

The result has:

```python
captured.out
captured.err
```

For example:

```python
assert "cli_test_model" in captured.out
assert captured.err == ""
```

The current test uses the shorter form:

```python
output = capsys.readouterr().out
```

### Why use `capsys` in the model-list test?

The purpose of `models list` is to show model information to the user. The test
verifies that the printed table contains:

```python
assert "cli_test_model" in output
assert "test" in output
assert "Model used to test CLI output" in output
```

Without `capsys`, the test could call the CLI but would not verify its
user-visible result.

## 10. `capsys.readouterr()` Clears the Capture Buffer

Each call returns output captured since the previous read.

For example:

```python
print("first")

first_capture = capsys.readouterr()
assert first_capture.out == "first\n"

print("second")

second_capture = capsys.readouterr()
assert second_capture.out == "second\n"
```

The second capture does not include `first`. Usually, call `readouterr()` once
after the action being tested.

## 11. `capsys` Versus `-s`

By default, pytest captures printed output. `capsys` lets the test inspect it.

Running:

```bash
python -m pytest -s
```

disables normal capture so you can see debugging prints immediately.

Use:

- `capsys` for permanent output assertions; and
- `-s` for temporary human debugging.

Remove temporary debugging prints after resolving the problem.

## 12. The Purpose of `arrange_cli()`

The test file centralizes common setup:

```python
def arrange_cli(
    monkeypatch,
    argv,
    registry,
):
    monkeypatch.setattr(
        cli,
        "discover_plugins",
        lambda plugin_paths: registry,
    )

    monkeypatch.setattr(
        cli,
        "setup_directories",
        lambda config: None,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        argv,
    )
```

Every CLI dispatch test needs these three controls:

```text
known registry
no filesystem writes
known command-line arguments
```

Rather than copying those patches into every test, `arrange_cli()` expresses the
common test setup.

This follows the usual test structure:

```text
Arrange
├── prepare registry
├── patch dependencies
└── provide CLI arguments

Act
└── call cli.main()

Assert
├── handler called once
├── correct model passed
└── correct configuration passed
```

For example:

```python
def test_train_dispatches_to_selected_workflow(
    monkeypatch,
):
    calls = []

    def fake_train(spec, config):
        calls.append((spec, config))

    # Arrange
    arrange_cli(
        monkeypatch,
        [...],
        make_test_registry(train_fn=fake_train),
    )

    # Act
    cli.main()

    # Assert
    assert len(calls) == 1
```

## 13. When Should You Generally Use `monkeypatch`?

Use it when the tested unit depends on external or process-level state that you
need to control.

Good uses include:

- replacing a filesystem-writing function;
- supplying environment variables;
- controlling the current directory;
- replacing network or API calls;
- replacing database access;
- controlling `sys.argv`;
- supplying deterministic plugin discovery;
- replacing expensive model execution;
- injecting a temporary module; and
- forcing a rare error condition.

Examples:

```python
monkeypatch.setenv(
    "KERAS_BACKEND",
    "torch",
)
```

```python
monkeypatch.chdir(tmp_path)
```

```python
monkeypatch.setattr(
    api_client,
    "request",
    fake_request,
)
```

```python
monkeypatch.setitem(
    config,
    "mode",
    "test",
)
```

## 14. When Should You Not Use `monkeypatch`?

Do not patch the exact behavior the test claims to verify.

For example, a discovery test should not do this:

```python
monkeypatch.setattr(
    discovery,
    "discover_plugins",
    lambda paths: expected_registry,
)
```

That would replace the function the test intended to exercise.

Similarly, a test claiming to verify artifact-file parsing should not replace
the parser with a fake result.

The principle is:

> Patch dependencies around the unit under test, not the behavior under test.

In `tests/test_cli_models.py`, the unit under test is CLI parsing and dispatch.
Therefore, patching discovery, filesystem creation, artifact storage, workflow
computation, and scientific evaluation is appropriate.

In `tests/test_discovery.py`, the unit under test is discovery itself. Therefore,
use real temporary plugin files instead of patching discovery.

## 15. Summary of Usage in `test_cli_models.py`

| Fixture or method | Temporary change | Reason |
|---|---|---|
| `monkeypatch.setattr(cli, "discover_plugins", ...)` | Replaces real discovery | Test CLI dispatch independently |
| `monkeypatch.setattr(cli, "setup_directories", ...)` | Prevents directory creation | Keep tests side-effect-free |
| `monkeypatch.setattr(sys, "argv", ...)` | Supplies command arguments | Simulate terminal invocation |
| `monkeypatch.setattr(cli.wf, "resolve_artifact_ref", ...)` | Supplies metadata | Avoid checkpoint filesystem access |
| `monkeypatch.setitem(sys.modules, ..., fake_module)` | Supplies a fake mixed evaluator | Isolate CLI from an obsolete scientific dependency |
| `capsys.readouterr()` | Reads printed output | Verify user-visible CLI messages |
| `calls.append(...)` | Records fake-handler invocation | Verify handler, count, and arguments |

The central idea is that these fixtures make the test deterministic and narrow:

```text
test_cli_models.py verifies CLI decisions

other test files verify discovery,
artifact I/O, scientific models,
and evaluation behavior
```
