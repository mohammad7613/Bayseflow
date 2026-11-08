# Cleaning the repository before a Git push

This is an instruction checklist. No cleanup or push was performed to create
this document. Preserve valuable research artifacts before deleting local files.
Use the [developer guide](CognitiveDeveloperGuide.md) for the supported workflow
and the [roadmap](CognitiveDeveloperRoadmap.md) for remaining release work.

## 1. Decide what belongs in the source repository

Keep the package, CLI, environment/build definitions, cognitive developer example,
tests, and current documentation. Retain article implementations under
`examples/article_workspace/` when they are scientific references needed to
understand or verify the example.

| Keep in Git | Keep elsewhere or regenerate |
| --- | --- |
| `cli.py`, maintained `bayesflow_models/` modules | Trained `.keras`/`.pt`/`.pth` artifacts and optimizer state |
| `pyproject.toml`, `uv.lock`, `.gitignore`, maintained `README.md` | Checkpoint `.meta.json` sidecars associated with generated runs |
| Workspace registrations, scientific code, configs, scripts and tests | Generated reports, `.npz` arrays, recovery PDFs, logs |
| Small, intentional fixtures and scientific reference descriptions | Virtual environments, Python/test/lint caches, generated build metadata |
| The three current guides and useful maintained reference docs | Personal editor/assistant settings and local synchronization scripts unless intentionally shared |

Blanket deletion of `examples/`, `tests/`, `docs/`, or `uv.lock` would remove useful
source. A directory being untracked is not evidence that it is redundant: much
of the refactored package and examples is currently untracked and needs staging.

### Current generated artifact locations

The checkout contains root `checkpoints/`, `results/`, `logs/`, `trained_model/`,
`trained_model1/`, and corresponding output directories below both example
workspaces. These are local run outputs, not dependencies of training from scratch.
Approximate nonempty sizes during review were:

| Location | Size |
| --- | --- |
| Root `checkpoints/` | 22 MB |
| `trained_model/` | 150 MB |
| `trained_model1/` | 30 MB |
| Cognitive workspace `checkpoints/` | 15 MB |
| Cognitive workspace `results/` | 132 KB |
| `.venv/` | 6.9 GB; keep locally if still using the environment |

Sizes change as experiments run. Checkpoint aliases will disappear with their
metadata and may be reused after cleanup. Documentation must tell readers to
select their actual alias rather than requiring your existing `run-002`.

## 2. Review legacy and redundant source before removing it

These are candidates requiring an explicit keep/migrate/remove decision:

| Candidate | Evidence and recommended action |
| --- | --- |
| `bayesflow_models/evaluation/runner.py` | Alternate runner with obsolete imports/signature; the CLI uses `fixed_engine.py`. Remove after confirming no maintained consumer imports it. |
| `main.py`, `main_workflow.py`, `parameter_recovery.py`, `bayesflow_models/workflow_trialwise.py` | Legacy entry points still import scientific modules removed from the package. Archive under an explicitly labeled legacy example, migrate, or remove together with their obsolete documentation. |
| `bayesflow_models/utils_real_data.py` | Contains legacy article assumptions/imports and is used by the legacy trialwise workflow. Decide whether to preserve a supported real-data API before removing it. |
| `cognitive_models_codes/mixed_tta_evaluation.py` | Old helper still imported by CLI `mixed-recovery`. First remove/deprecate that route and update its test; deleting only the helper leaves a broken command. |
| Unused `_generate_batch` and misspelled unfinished homogeneous helper in `myscenarios.py` | Active registrations use the public homogeneous/mixed wrappers. Remove dead experiments after an import/reference check. |
| The two scientific implementations (`science.py` and `myscience.py` families) | Currently back different registered models and tests. Keep both or deliberately choose a canonical example and update all dependent registrations/tests/scripts. They are not interchangeable duplicates. |
| `train.py` legacy `train_amortizer`/`train_amortizer_load` | Active built-in lifecycle uses `train_amortizer_resume`. Check remaining callers before trimming older helpers. |
| Root `setup_environment.sh`, `setup_pytorch_env.sh`, environment check scripts | Review references to deleted requirements files and old paths. Keep only maintained installation routes. |
| `docs/WorkStream*.md`, older plans/reviews, moved `documentation/` content | Consolidate current facts into maintained guides; archive useful design history separately. Some old reviews describe defects already fixed. |
| `notebooks/`, root `1.pdf` | Retain only reproducible, intentional tutorials/references. Clear notebook outputs if kept. The existing `*.pdf` ignore is broad; explicitly handle any reference PDF you intend to version. |
| `.claude/`, `.vscode/`, `copy_to_pc.sh` | Inspect for user-specific configuration/paths. Ignore or remove unless intended as portable project tooling. |

Run reference checks before deleting source. For example:

```bash
rg -n 'evaluation\.runner|mixed_tta_evaluation|workflow_trialwise|utils_real_data' \
  cli.py bayesflow_models examples tests
rg -n 'requirements|parameter_recovery|main_workflow' \
  README.md docs examples setup_environment.sh setup_pytorch_env.sh
git diff --name-status --find-renames
```

Scientific files have already been removed from their original package paths.
Stage the retained article examples in the same reviewed change so the migration
does not accidentally publish only the deletions.

## 3. Make ignores explicit and check tracking

The current `.gitignore` already excludes directories named `checkpoints/`,
`results/`, `logs/`, `trained_model*/`, `real_data/`, and `*.keras`. Directory
patterns without a leading slash also apply inside the example workspaces.

Add these missing project-level rules if appropriate; do not rely on one user's
global Git ignore settings:

```gitignore
# Environments and caches
.venv/
.cache/
.pytest_cache/
.ruff_cache/

# Build products
*.egg-info/
build/
dist/

# Local agent settings; ignore editor settings only if they are personal
.claude/
# .vscode/
```

Add `*.pt`/`*.pth` only if those files are generated artifacts throughout this
project. Prefer directory-specific exclusions when you intend to retain small
fixtures of the same type. Do not ignore every JSON/YAML/NPZ file indiscriminately.

Inspect both ignored and tracked files:

```bash
git status --short
git status --short --ignored
git ls-files -ci --exclude-standard
git ls-files -- checkpoints results logs trained_model trained_model1 \
  examples/cognitive_developer_workspace/checkpoints \
  examples/cognitive_developer_workspace/results \
  examples/cognitive_developer_workspace/logs \
  examples/article_workspace/checkpoints \
  examples/article_workspace/results examples/article_workspace/logs
```

At review time, these artifact directories had no tracked files. An ignore rule
does not untrack a file already committed. If the commands above show tracked
artifacts, remove their index entries with a scoped command such as
`git rm -r --cached --ignore-unmatch -- checkpoints results logs`; this keeps local
files. Apply the same operation only to the other reviewed artifact paths.
Untracking does not remove large files from existing Git history. Inspect history
separately if an earlier commit already included research data.

## 4. Back up outputs, then preview local deletion

You do not have to delete ignored local outputs to keep them out of a push.
Delete them when you want a clean source-only working tree or need disk space.
For valuable runs, copy checkpoints **with their matching metadata**, plus
reports, arrays, configs, and the code revision, to storage outside the checkout.
Verify that backup before proceeding. Keep private participant data separately.

From the repository root, define only the generated directories you intend to
clean. The following Bash array excludes the working environment and all source:

```bash
artifact_paths=(
  checkpoints results logs trained_model trained_model1
  examples/cognitive_developer_workspace/checkpoints
  examples/cognitive_developer_workspace/results
  examples/cognitive_developer_workspace/logs
  examples/article_workspace/checkpoints
  examples/article_workspace/results
  examples/article_workspace/logs
)

# Preview ignored, untracked output deletions.
git clean -ndX -- "${artifact_paths[@]}"
```

After reviewing the preview and verifying backups, run the exact same scoped
operation without dry-run:

```bash
git clean -fdX -- "${artifact_paths[@]}"
```

`-X` removes only ignored files; it does not remove tracked outputs or unexpected
nonignored files. Inspect leftovers rather than broadening the command blindly.
Avoid repository-wide `git clean -fdx` or `git clean -fdX`: these can delete your
environment and other valuable local files. Optional cache cleanup should also
use an explicit path list and a dry-run first. Removing `.venv/` is unnecessary
for pushing and forces dependency reinstallation.

## 5. Verify before staging

Use the baseline checks from the repository root:

```bash
KERAS_BACKEND=torch MPLBACKEND=Agg uv run --no-sync python -m pytest \
  tests examples/cognitive_developer_workspace/tests -q

uv run --no-sync bayesflow-cli --help
uv run --no-sync bayesflow-cli \
  --plugin-path examples/cognitive_developer_workspace models list
uv run --no-sync bayesflow-cli \
  --plugin-path examples/cognitive_developer_workspace evaluations list
```

On 2026-11-06, the tests had **121 passes and 2 failures**: the old generator
signature in `test_fixed_eginge.py`, and the legacy mixed-recovery import in
`test_cli_models.py`. Resolve these and reconcile the example scripts before
claiming a clean release. The [roadmap](CognitiveDeveloperRoadmap.md) gives the
concrete actions. A successful prior experiment does not replace these checks.

Run a small real train/save/load/resume/evaluate cycle in temporary output roots
after the example wiring is fixed. Verify that resume changes trainable weights.
If you run it inside the example output directories instead, repeat the scoped
cleanup before packaging a source-only archive.

For distribution, also build and install a wheel into an isolated environment
and run it outside the checkout. In particular, the old CLI import from
`examples...mixed_tta_evaluation` cannot work as an installed package contract:
the current package discovery only ships `bayesflow_models*` and the CLI module.

## 6. Review, commit, and push deliberately

The checkout already contains staged and unstaged deletions plus untracked new
source. Review both `git diff` and `git diff --cached`. Do not stage only these
three docs and assume the package refactor is included.

```bash
git switch -c codex/cognitive-developer-release
git diff --stat
git diff --cached --stat

# Stage maintained/new source and refactored example code after review.
git add -- .gitignore README.md cli.py bayesflow_models examples docs tests \
  pyproject.toml uv.lock

# Review other existing tracked changes/deletions before staging them explicitly.
git status --short
git diff --cached --check
git diff --cached --stat
git diff --cached --name-status --find-renames
git diff --cached
```

Add or remove the legacy root files according to your decisions in section 2.
Check that staged files contain no checkpoint binaries, generated result arrays,
private data, local secrets, or accidental machine-specific settings. Confirm the
package license/author metadata and README installation instructions are intentional.

The current tracked source diff also reports trailing whitespace, including CRLF
line endings in `bayesflow_models/train.py` and `main.py`, plus whitespace in
`discovery.py` and `cli.py`. Review and normalize the files you intend to retain,
then rerun `git diff --cached --check`. Keep mechanical formatting separate from
scientific changes when that makes the review easier.

After the tests and staged review are satisfactory:

```bash
git commit -m "Separate cognitive models from package workflows and document usage"
git remote -v
git branch --show-current
```

This checkout has two remotes: `origin` (GitHub) and `origin1` (GitLab). Choose the
intended destination. For example, to publish the branch above to GitHub:

```bash
git push -u origin codex/cognitive-developer-release
```

Use `origin1` instead only when GitLab is your intended destination. Review and
merge according to your repository workflow; no force push or history rewrite is
needed for ordinary cleanup and source moves.
