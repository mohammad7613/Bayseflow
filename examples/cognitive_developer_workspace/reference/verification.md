# Verification record

Checked on 29 October 2026, using the existing parent uv environment and CPU.

| Check | Result |
| --- | --- |
| Prior/trial equivalence with the old model_DC functions | Exact array equality for both reference seeds/conditions |
| Workspace unit tests | 19 passed |
| Entire repository test suite, including this workspace | 119 passed |
| Ruff checks and formatting | Passed for all 19 workspace Python files |
| Fish script syntax | Passed |
| CLI model and evaluation discovery | model_DC, homogeneous-tta, mixed-tta discovered |
| Real CLI training | Passed; saved model and metadata |
| Fresh-process checkpoint loading | Passed |
| Resume into a new artifact using dc_example workflow | Passed; optimizer iterations 2 -> 4, trainable weight digest changed |
| Homogeneous recovery using the package engine | Passed; arrays, raw conditions, JSON report saved |
| Mixed recovery using the package engine | Passed; same execution path, different scenario generator |
| Fish recovery scripts and PDF plotting | Both passed; mixed recovery plot visually inspected |
| Previous source/config/documentation outside this workspace | 57 file-content hashes unchanged |

The test suite reports Keras/Torch NumPy-conversion deprecation warnings. These
did not fail tests. No dependencies were installed or changed.

## Actual successful artifacts in this checkout

- Training: `run-004`, artifact `model_DC__20261029T202732__bb226315`.
- Resume: `run-005`, artifact `model_DC__20261029T202746__4c07cd98`.
- Successful full CLI check: [verification.json](../logs/smoke-7aa06ade9961/verification.json).
- Built-in resume diagnostic: [failed check](../logs/smoke-1d827e30aa7b/verification.json).

For an immediate inspection using an already generated checkpoint:

```fish
fish scripts/homogeneous_recovery.fish run-005
fish scripts/mixed_recovery.fish run-005
```

Artifacts and logs are generated local files ignored by Git. Their aliases are
specific to this checkout; the scripts/list.fish table is authoritative. A fresh
copy must run training or the smoke test to create its own artifacts.

## What the resume diagnostic established

The built-in package path loaded a finished cosine schedule with learning rate
0.0. Its optimizer iterations increased from 2 to 4 while the digest of every
trainable weight remained identical. The diagnostic deliberately marks that as
a failure rather than accepting a successful exit code alone.

The workspace's explicitly registered dc_example workflow preserves the loaded
optimizer moments and counter, but restarts its learning-rate schedule at a
constant 0.0005. Its resumed checkpoint has a different trainable-weight digest
and a nonzero rate. The package itself remains unchanged.

## Limits of these checks

These are tiny software-integration runs: two training batches of four datasets,
one initial epoch and one additional epoch, eight evaluation datasets, and 32
posterior draws per dataset. They do not establish parameter identifiability,
calibration, adequate training, or scientific validity of the legacy equations.
GPU execution and bit-identical training across hardware are not claimed.
