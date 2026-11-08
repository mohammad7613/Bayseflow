# Dockerizing bayesflow-cli: Developer Thinking and Plan

## Current Starting Point

The project is now an installable Python CLI package, not only a collection of scripts. The package metadata lives in `pyproject.toml`, and the public terminal command is:

```bash
bayesflow-cli
```

That command calls `cli:main`, so Docker should run the package through the same interface users run locally. The Docker image should not depend on executing `python cli.py` from a specific source checkout.

The app is a small lab runtime around BayesFlow. Cognitive model developers provide model definitions through `MODEL_SPECS` and optionally `WORKFLOWS`; the app then handles listing models, training, resuming, recovery, mixed-TTA recovery, checkpoints, logs, and results.

## How An Expert Developer Thinks

### 1. What Is The Application Boundary?

Question:

> What exactly should the Docker image contain?

Answer for this project:

The image should contain the `bayesflow-cli` package, its Python dependencies, BayesFlow, Keras, Torch, and the built-in model/workflow registry. The image should expose one stable command:

```bash
bayesflow-cli
```

The image should not contain researcher data, trained artifacts, local notebooks, or local experiment outputs.

### 2. What Belongs Outside The Image?

Question:

> Which files are owned by the researcher rather than by the application?

Answer for this project:

These should stay outside the image and persist after the container exits:

- User plugin files containing `MODEL_SPECS`, `WORKFLOWS`, `register_models()`, or `register_workflow()`.
- `checkpoints/`
- `results/`
- `logs/`
- Future `real_data/` inputs when an inference command is implemented.

The Docker image is replaceable. The researcher workspace is not.

### 3. How Should Users Avoid Repeating Many Mounts?

Question:

> Should users mount `checkpoints/`, `results/`, and `logs/` separately?

Answer for this project:

No. The better runtime contract is one mounted workspace. The app already creates output folders relative to `Path.cwd()`, so Docker should set the current working directory to a mounted workspace.

Recommended runtime shape:

```text
host project folder
  model_specs.py
  helper_code.py
  checkpoints/
  results/
  logs/
```

Inside the container, this becomes:

```text
/workspace
  model_specs.py
  helper_code.py
  checkpoints/
  results/
  logs/
```

This means a single bind mount is enough:

```yaml
volumes:
  - .:/workspace
working_dir: /workspace
```

### 4. How Does Model Discovery Work In Docker?

Question:

> How will the container discover researcher-defined cognitive models?

Answer for this project:

Built-in models and workflows are always discovered from the package internals. External discovery works from plugin files.

If `--plugin-path` is passed, discovery scans that path. If no `--plugin-path` is passed, discovery scans the current working directory, which should be `/workspace` in Docker.

Discovery first parses Python files with `ast` and only imports files that contain plugin markers:

```python
MODEL_SPECS
WORKFLOWS
register_models
register_workflow
```

This avoids importing unrelated Python scripts during discovery.

### 5. Which Backend Must Docker Enforce?

Question:

> What runtime environment variable must be fixed before importing Keras/BayesFlow?

Answer for this project:

Keras must use the Torch backend:

```dockerfile
ENV KERAS_BACKEND=torch
```

The Python entrypoint also sets this defensively before importing modules that can import Keras or BayesFlow. Docker should still set the environment variable so the whole container has the same backend contract.

### 6. CPU Or GPU First?

Question:

> Should the first Docker image target CPU or GPU?

Answer for this project:

Start with a CPU image. It is easier to build, easier to run on any machine, and enough to validate packaging, CLI behavior, discovery, mounted workspaces, and output persistence.

GPU support should be added after the CPU image is working. It requires NVIDIA Container Toolkit and a CUDA-compatible Torch wheel/image strategy.

## Dockerization Plan

### Step 1: Add `.dockerignore`

Exclude files that should not enter the image:

```text
.git/
.vscode/
.codex
__pycache__/
*.pyc
.ipynb_checkpoints/
bayesflow_env/
checkpoints/
results/
logs/
trained_model*/
*.keras
*.pdf
notebooks/
```

The image should be built from package code and metadata, not from local caches, outputs, notebooks, or trained artifacts.

### Step 2: Add A CPU Dockerfile

The Dockerfile should:

- Start from a Python 3.12 base image.
- Set `KERAS_BACKEND=torch`.
- Set `MPLCONFIGDIR=/tmp/matplotlib` so Matplotlib does not try to write under an unwritable home config path.
- Copy package metadata and source code.
- Install the package from `pyproject.toml`.
- Use `ENTRYPOINT ["bayesflow-cli"]`.

Conceptual shape:

```dockerfile
FROM python:3.12-slim

ENV KERAS_BACKEND=torch
ENV MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY pyproject.toml README.md ./
COPY cli.py ./
COPY bayesflow_models ./bayesflow_models

RUN pip install --no-cache-dir .

WORKDIR /workspace

ENTRYPOINT ["bayesflow-cli"]
```

The exact dependency installation strategy can be refined when writing the file, especially for CPU versus CUDA Torch wheels.

### Step 3: Add `compose.yaml`

Compose should make repeated use simple:

```yaml
services:
  bayesflow:
    build: .
    working_dir: /workspace
    volumes:
      - .:/workspace
    environment:
      KERAS_BACKEND: torch
      MPLCONFIGDIR: /tmp/matplotlib
    entrypoint: ["bayesflow-cli"]
```

The important design is one workspace mount. Users should not mount `checkpoints/`, `results/`, and `logs/` separately for normal use.

### Step 4: Validate The Container Contract

Run:

```bash
docker compose build
docker compose run --rm bayesflow models list
```

Then run a tiny training smoke test:

```bash
docker compose run --rm bayesflow train --models model_SF --epochs 1 --n-sim 1 --batch-size 2
```

After the run, confirm these folders exist on the host:

```text
checkpoints/
results/
logs/
```

### Step 5: Validate Plugin Discovery

Create or use a workspace plugin file containing `MODEL_SPECS` or `register_models()`.

Then run:

```bash
docker compose run --rm bayesflow models list
```

The model list should include built-in models plus any workspace plugin models.

If a user wants a separate plugin directory:

```bash
docker compose run --rm bayesflow --plugin-path /workspace/plugins models list
```

## Expected User Experience

From a researcher workspace:

```bash
docker compose run --rm bayesflow models list
docker compose run --rm bayesflow train --models model_SF --epochs 2 --n-sim 1000 --batch-size 50
docker compose run --rm bayesflow checkpoints list
docker compose run --rm bayesflow recovery --artifacts run-001 --mode save_only
```

The image provides the application. The mounted workspace provides cognitive model plugins and persistent experiment state.

## Acceptance Criteria

- `bayesflow-cli models list` works locally.
- `docker compose build` succeeds.
- `docker compose run --rm bayesflow models list` works.
- Container runs with `KERAS_BACKEND=torch`.
- Outputs are created under the mounted host workspace, not trapped inside the image.
- Built-in models are available without plugin files.
- Workspace plugin models are discovered when plugin markers are present.
- Users do not need to manually mount `checkpoints/`, `results/`, and `logs/` separately.

## Later GPU Plan

After CPU Docker works, add GPU support with a separate Compose service or Dockerfile target.

GPU runtime requirements:

- NVIDIA driver on the host.
- NVIDIA Container Toolkit installed.
- CUDA-compatible Torch build.
- Compose GPU device reservation or `docker run --gpus all`.

The CLI command should stay the same. Only the runtime environment changes.
