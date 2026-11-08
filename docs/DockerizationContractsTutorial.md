# Dockerization by Contract: A Reusable Tutorial and This Project's Design

## Purpose

This document teaches a repeatable way to dockerize an application. The goal is
not to start by writing Docker syntax. The goal is to understand the application
well enough to state two contracts:

1. **Build-time contract:** what Docker needs in order to produce an immutable
   application image.
2. **Runtime contract:** what a container needs and promises when that image is
   started.

Once these contracts are explicit, a `Dockerfile`, `.dockerignore`, and
`compose.yaml` become implementations of decisions rather than collections of
copied commands.

Each section below contains:

- the general question to ask for any project;
- the reason the question matters;
- the answer for this `bayesflow-cli` project; and
- the Docker decision produced by that answer.

The first target in this tutorial is a reproducible **CPU image**. GPU support is
a later extension because it adds a host-driver and container-runtime contract.

---

## 1. The Mental Model

Docker separates four things that are often mixed together during local
development:

```text
source + build instructions
            |
            | docker build
            v
      immutable image
            |
            | docker run + runtime inputs
            v
    disposable container
            |
            +---- writes durable state ----> volume or bind mount
            +---- emits observations ------> stdout/stderr, metrics, exit code
```

An **image** is a versioned application artifact. A **container** is one running
or stopped instance of that image. A container's writable layer is disposable;
valuable data must not depend on it.

This gives the most important general rule:

> Put application code and fixed dependencies in the image. Supply
> environment-specific configuration, secrets, user inputs, and durable output
> at runtime.

For this project:

- The image owns the installed `bayesflow-cli` package, built-in models, and
  Python dependencies.
- The researcher owns external plugin files, real data, checkpoints, results,
  and logs.
- A container may be deleted and recreated without losing researcher-owned
  files.

---

## 2. Is Dockerization Appropriate?

### Questions to answer

1. Is there a repeatable program or service to run?
2. Are environment differences causing installation or reproducibility costs?
3. Can the process run without an interactive desktop?
4. Can durable state be separated from the application installation?
5. Are required licenses and redistribution terms compatible with the image?

### General principle

Docker is most useful when the process has a definable interface and its
environment can be reproduced. Docker does not automatically solve unclear data
ownership, secret management, hardware incompatibility, or a program that only
works inside one developer's home directory.

### Answer for this project

- The repeatable program is an installable CLI.
- Python, Keras, Torch, BayesFlow, and native wheels make environment
  reproducibility valuable.
- Training and recovery can run from a terminal.
- Outputs are already configurable and can live outside the image.
- Dependency licenses still need to be reviewed before distributing the image
  outside the team.

### Decision

Dockerization is appropriate. Treat this as a **batch CLI container**, not a
long-running web service.

---

## 3. Write the Application Boundary First

### Questions to answer

1. What is the single supported way to start the application?
2. Which source directories are part of the product?
3. Which development files are not part of the product?
4. Does the program assume a particular current working directory?
5. Which files must be installed for package discovery and imports to work?

### General principle

The container should expose the same stable interface that a user would use
after installing the application normally. Avoid making an incidental source
file or IDE command the production interface.

### Answer for this project

`pyproject.toml` declares:

```toml
[project.scripts]
bayesflow-cli = "cli:main"
```

The installed application consists of:

- `cli.py`;
- the `bayesflow_models` package, including built-in plugins; and
- package metadata such as `pyproject.toml` and `README.md` during the build.

Notebooks, the local virtual environment, Git history, generated checkpoints,
results, and logs are not required inside the application installation.

### Decision

- Install a built wheel into the image.
- Use `bayesflow-cli` as the image entrypoint.
- Install the application under an image-owned location such as `/app` or the
  Python environment, independently from the user workspace.
- Set the runtime working directory to `/workspace`.

Keeping `/app` and `/workspace` separate is important. Mounting a host directory
over `/workspace` must not hide the installed application.

---

## 4. The Build-Time Contract

The build-time contract should be readable as a short specification:

> Given an approved base image, the packaging metadata, the application source,
> dependency indexes, and network access, the build produces an immutable image
> containing an installed and invokable application. It does not contain local
> data, secrets, caches, or experiment outputs.

### 4.1 What is the build input?

#### Questions

1. What is the build context directory?
2. Which files are needed to build the package?
3. Which large, private, generated, or irrelevant files must be excluded?
4. Does the build require credentials for a private package index?

#### General principle

Docker can access files only from its build context. A smaller context makes
builds faster, reduces accidental data inclusion, and improves cache behavior.
`.dockerignore` is both a performance tool and a security boundary.

#### Project answer

The repository root is the intended build context. The minimal application
inputs are:

```text
pyproject.toml
README.md
cli.py
bayesflow_models/
```

Candidate exclusions include:

```text
.git/
.agents/
.codex/
.vscode/
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.pytest_cache/
.ruff_cache/
.ipynb_checkpoints/
bayesflow_env/
checkpoints/
trained_model*/
results/
logs/
real_data/
notebooks/
*.keras
```

Review exclusions before using them: a file must not be ignored if it becomes a
required package or test input.

#### Decision

Create a `.dockerignore` before the first meaningful image build. Never copy the
entire developer environment into the image.

### 4.2 Which platform and language version are required?

#### Questions

1. Which runtime and version does the package declare?
2. Which OS and CPU architectures must be supported?
3. Do dependencies publish compatible wheels for that combination?
4. Is a minimal base image compatible with required native libraries?

#### General principle

Choose a base image from the application contract, not from whichever tag is
shortest. Pin at least the major and minor runtime version. For stronger
reproducibility and supply-chain control, production builds can pin the image by
digest and use a deliberate update process.

#### Project answer

`pyproject.toml` requires Python 3.12 or newer. The first target is Linux,
x86-64, CPU-only. Packages such as Torch, NumPy, and SciPy depend on compatible
binary wheels.

#### Decision

Begin with a trusted Python 3.12 slim Linux image. Verify wheel availability
before deciding that an Alpine/musl image is suitable; scientific Python stacks
are generally simpler on glibc-based images.

### 4.3 How are dependencies resolved?

#### Questions

1. Where are runtime dependencies declared?
2. Are versions pinned, constrained, or locked?
3. Which package indexes provide platform-specific wheels?
4. Does the package manager understand project-specific index configuration?
5. Must the build work offline?

#### General principle

A project wheel normally contains the project's code and dependency metadata,
not the dependency code. Dependencies are separate build inputs. Reproducibility
requires controlling their versions and sources.

#### Project answer

Runtime dependencies are pinned in `pyproject.toml`. The project also contains
`[tool.uv]` CPU and CUDA source configuration. That configuration is understood
by `uv`; plain `pip` does not automatically interpret `tool.uv.sources` as pip
index instructions.

The command:

```bash
python -m pip wheel . \
    --no-deps \
    --no-build-isolation \
    --wheel-dir /tmp/bayesflow-wheel
```

builds only the application wheel. It does not put BayesFlow, Keras, Torch, or
the other dependencies inside that wheel. A later installation step must install
those dependencies from approved indexes or from a prepared wheelhouse.

#### Decision

Before finalizing the Dockerfile, choose one explicit dependency strategy:

1. **Online pip build:** install the project wheel and allow pip to resolve
   pinned dependencies from explicitly configured indexes.
2. **uv build:** use `uv` consistently so the existing source configuration is
   honored, ideally with a checked-in lock file.
3. **Offline wheelhouse:** download/build all required wheels in a controlled
   stage, then install with `--no-index --find-links=...`.

Do not claim a CPU-only image until the resulting Torch installation has been
verified. `cpu` in project metadata is not, by itself, proof of which binary pip
selected.

### 4.4 Is build isolation needed?

#### Questions

1. What build backend and version does `pyproject.toml` require?
2. Can the build environment download build-system requirements?
3. If isolation is disabled, who guarantees compatible build tools are present?

#### General principle

PEP 517 build isolation creates a clean temporary build environment and is the
safer default. `--no-build-isolation` is a controlled exception, not a general
speed flag: it transfers responsibility for installing compatible build tools
to the Dockerfile author.

#### Project answer

The project uses `setuptools.build_meta` and requires `setuptools>=68`. Earlier
offline work used `--no-build-isolation` because isolation attempted a network
download.

#### Decision

If the Docker build uses `--no-build-isolation`, explicitly install the required
build backend first and document why. Otherwise, allow the isolated build stage
to obtain the declared tools.

### 4.5 What artifact does the build produce?

#### Questions

1. Is the deployable artifact a wheel, executable, JAR, static bundle, or other
   object?
2. Can it be built once and installed into a clean runtime stage?
3. How will its version and provenance be identified?

#### General principle

Build once, test the built artifact, and promote that same artifact. Avoid
rebuilding differently for test and production.

#### Project answer

The project artifact is expected to resemble:

```text
bayesflow_cli-0.1.0-py3-none-any.whl
```

It contains project code and metadata. Native, platform-specific dependencies
remain separate packages.

#### Decision

Use a multi-stage build:

```text
builder stage: source -> application wheel and dependency installation material
runtime stage: approved runtime base + installed artifacts only
```

This separation is valuable when build compilers or package-manager caches are
not needed at runtime. It does not automatically make a Torch image small; the
runtime still needs Torch itself.

### 4.6 What must never be present at build time?

#### Questions

1. Are credentials copied into the build context?
2. Are tokens passed using `ARG` or written by a `RUN` command?
3. Could secrets remain in image layers or build logs?

#### General principle

Never bake secrets into images. Build arguments and copied secret files can be
recoverable from history or layers. If a private index is required, use a
supported BuildKit secret mount and ensure the secret is not persisted.

#### Project answer

The current public dependency list should not require application secrets. If a
private model registry or Python index is added, its credential becomes a build
secret, not an `ENV`, `ARG`, or repository file.

#### Decision

The normal build must contain no researcher data, API keys, SSH keys, or package
registry credentials.

---

## 5. The Runtime Contract

The runtime contract for this project can be summarized as:

> Starting the image runs `bayesflow-cli` as a non-root user in `/workspace`.
> The user supplies a CLI subcommand and may mount one trusted researcher
> workspace. The process reads plugins and inputs from that workspace, writes
> durable artifacts there, reports progress through standard output/error, and
> exits with a meaningful status code. Torch is the Keras backend. CPU is the
> default hardware target.

### 5.1 What process starts?

#### Questions

1. Is this a service, scheduled job, interactive tool, or one-shot command?
2. What is the executable?
3. Which arguments vary per run?
4. Should users be able to replace the command easily?

#### General principle

Use `ENTRYPOINT` for the stable executable and `CMD` for replaceable default
arguments. Prefer JSON/exec form so the application receives signals directly.

#### Project answer

The stable executable is `bayesflow-cli`; subcommands vary:

```text
models list
checkpoints list
train ...
resume ...
recovery ...
mixed-recovery ...
```

#### Decision

The conceptual image interface is:

```dockerfile
ENTRYPOINT ["bayesflow-cli"]
CMD ["--help"]
```

Then a user can run the equivalent of:

```bash
docker run ... IMAGE models list
```

For this batch application, a long-running `CMD` and a network port are not
required.

### 5.2 What configuration varies at runtime?

#### Questions

1. Which settings are command-line arguments?
2. Which settings are environment variables?
3. Which settings belong in mounted configuration files?
4. Which defaults are safe and portable?

#### General principle

Use command-line arguments for per-invocation behavior, environment variables
for deployment-wide runtime behavior, and files for structured configuration.
Do not rebuild an image to change an experiment's epoch count or input path.

#### Project answer

Per-run settings already belong to CLI arguments, including model names, epochs,
simulation counts, batch size, artifact references, device, and optional output
paths. Required container-wide environment includes:

```text
KERAS_BACKEND=torch
MPLCONFIGDIR=/tmp/matplotlib
```

`KERAS_BACKEND` must be set before Keras or BayesFlow is imported. `cli.py` also
sets a defensive default, but the container should state the backend contract
explicitly.

#### Decision

- Set stable backend defaults in the image.
- Keep experiment parameters as runtime CLI arguments.
- Do not put secrets in the image or Compose file; inject them at runtime if a
  future feature needs them.

### 5.3 What data enters the container?

#### Questions

1. Which inputs ship with the application?
2. Which inputs are supplied by the user?
3. Are inputs trusted code or passive data?
4. Should an input mount be read-only?

#### General principle

Separate image-owned resources from runtime inputs. Mount passive data read-only
when the process need not modify it. Treat dynamically imported plugins as code,
not as harmless data.

#### Project answer

- Built-in model and workflow plugins ship inside the installed package.
- External plugins are discovered from each `--plugin-path`, or from the current
  working directory when none is provided.
- Plugin files are imported and executed. The AST prefilter reduces accidental
  imports but is not a security sandbox.
- Real datasets are researcher-owned runtime inputs.

#### Decision

- Run only trusted plugins.
- Use `/workspace` as the default discovery and working directory.
- For stronger isolation later, separate read-only input/plugin mounts from a
  writable output mount. The first usability-oriented design may use one
  read-write workspace, but that tradeoff must be explicit.

### 5.4 What state must survive container deletion?

#### Questions

1. What does the process create?
2. Which outputs are valuable?
3. Which files are temporary caches?
4. Who owns backups and retention?
5. Can two containers safely write to the same location?

#### General principle

Containers are replaceable; durable state is not. Store valuable output in a
volume, bind mount, object store, or database designed for that state. Decide
concurrency rules instead of assuming shared writes are safe.

#### Project answer

`cli.py` derives these paths from `Path.cwd()` unless the user overrides them:

```text
checkpoints/
results/
logs/
```

Real data and external plugins also belong to the researcher. A convenient
initial workspace is:

```text
host experiment directory/
├── model_plugin.py
├── real_data/
├── checkpoints/
├── results/
└── logs/
```

mounted as:

```text
/workspace
```

#### Decision

Use one bind-mounted workspace first:

```yaml
working_dir: /workspace
volumes:
  - .:/workspace
```

This matches the current application behavior and keeps all experiment artifacts
visible on the host. Do not assume concurrent training containers can safely
write to the same checkpoint or result paths; use separate experiment
directories unless concurrency has been tested and designed.

### 5.5 What permissions does the process need?

#### Questions

1. Must the program run as root?
2. Which directories require write access?
3. Will host-mounted files have compatible user and group IDs?
4. Does the program need Linux capabilities or privileged mode?

#### General principle

Run as a non-root user and grant only required filesystem access. Bind mounts
introduce host/container UID and GID considerations. Do not use `--privileged`
as a generic permission fix.

#### Project answer

The CLI requires write access to its output directories and Matplotlib's config
directory. It does not inherently require root, privileged mode, or extra Linux
capabilities.

#### Decision

- Create and use a non-root runtime user.
- Make `/workspace` the intended writable location.
- Set `MPLCONFIGDIR` to a writable path such as `/tmp/matplotlib`.
- Decide how host UID/GID mapping will work before relying on bind mounts across
  different machines. Compose can pass a user mapping when needed.

### 5.6 What hardware does runtime require?

#### Questions

1. Is CPU sufficient?
2. Are memory, shared memory, or disk requirements known?
3. If a GPU is required, which vendor, driver, runtime, CUDA, and framework
   versions must agree?
4. What happens when requested hardware is unavailable?

#### General principle

Hardware is part of the runtime contract. A GPU container is not self-contained:
it relies on compatible host drivers and a container runtime capable of exposing
the device.

#### Project answer

The CLI accepts a device argument and uses Torch. CPU is sufficient for contract
validation, although real training may be slow. GPU support would additionally
require compatible Torch/CUDA artifacts, NVIDIA host drivers, and NVIDIA
Container Toolkit.

#### Decision

- Deliver and validate CPU first.
- Add GPU as a separate image target or Compose service.
- Do not mix CPU and CUDA dependency resolution implicitly in one ambiguous
  build.

### 5.7 Does the application need networking?

#### Questions

1. Does it listen for inbound connections?
2. Does it call external services at runtime?
3. Which ports and protocols are required?
4. Can runtime network access be removed?

#### General principle

Do not expose ports merely because examples do. A batch CLI that neither listens
nor calls external systems needs no published port.

#### Project answer

The current CLI does not define a server or listening port. Normal training and
recovery should use locally installed dependencies and mounted data.

#### Decision

Do not add `EXPOSE` or Compose `ports` for the current application. Document any
future remote tracking or data service as a new runtime dependency.

### 5.8 How does the process report health and failure?

#### Questions

1. What does success mean for this process?
2. Does it return a nonzero exit code on failure?
3. Where do logs go?
4. Is a service health check actually meaningful?

#### General principle

For a long-running service, define health checks. For a batch job, the primary
contract is completion, exit status, standard output/error, and expected
artifacts. A process merely being alive does not prove training is correct.

#### Project answer

This is a batch CLI. `models list` is a cheap installation/discovery smoke test.
A training command's success requires a zero exit code and expected output
artifacts. Logs and results are written under the workspace.

#### Decision

- Do not invent an HTTP health check.
- Use CLI smoke tests during image validation.
- Ensure exceptions reach the process boundary so Docker observes a nonzero
  exit status.
- Treat scientific correctness tests separately from container health.

### 5.9 How does the process stop?

#### Questions

1. Does it receive `SIGTERM` and `SIGINT` directly?
2. Can it leave partially written checkpoints?
3. Is graceful shutdown implemented?

#### General principle

Use exec-form `ENTRYPOINT`/`CMD` so the application is the signal-receiving
process. Long computations should define checkpoint and interruption semantics.

#### Project answer

The JSON entrypoint gives the Python CLI direct signal delivery. The current
container design should not assume that an interrupted training step always
leaves a resumable checkpoint; that behavior must be tested at the application
level.

#### Decision

Use exec form and document interruption/recovery behavior after testing it.

---

## 6. Convert the Contracts into Docker Artifacts

Do this only after the preceding questions have explicit answers.

### Artifact 1: `.dockerignore`

It implements the build-input boundary. Review it by asking:

- Is every included file necessary or deliberately included?
- Can any secret, dataset, model artifact, virtual environment, or cache enter
  the context?
- Did an ignore rule accidentally remove package data required at runtime?

### Artifact 2: `Dockerfile`

It implements the build contract and image-owned part of the runtime contract.
Its conceptual stages are:

```dockerfile
# Stage 1: build the application artifact
FROM python:3.12-slim AS builder
WORKDIR /source

# Install declared build tooling, then copy only required build inputs.
# Build the application wheel and prepare runtime dependencies.

# Stage 2: create the runtime image
FROM python:3.12-slim AS runtime

ENV KERAS_BACKEND=torch \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create a non-root user.
# Install the built application and its runtime dependencies.

WORKDIR /workspace
ENTRYPOINT ["bayesflow-cli"]
CMD ["--help"]
```

This is intentionally a design skeleton, not yet a production Dockerfile. The
dependency strategy must be chosen explicitly before filling in its `RUN`
commands.

### Artifact 3: `compose.yaml`

Compose describes a convenient runtime instantiation of the image. For this
project, the first CPU service concept is:

```yaml
services:
  bayesflow:
    build:
      context: .
    working_dir: /workspace
    volumes:
      - .:/workspace
    environment:
      KERAS_BACKEND: torch
      MPLCONFIGDIR: /tmp/matplotlib
    entrypoint: ["bayesflow-cli"]
```

This says that the image owns the installed application while the current host
directory supplies the researcher workspace. Compose is not required to build
or run an image, but it records repeated runtime options and makes the contract
easier to use consistently.

Before adopting this exact mount, consider whether the repository root is truly
the desired experiment workspace. A separate workspace directory can reduce the
amount of host source visible to dynamically imported plugins.

### Artifact 4: image metadata and documentation

Record at least:

- application version or source revision;
- supported architecture;
- CPU or GPU variant;
- expected entrypoint and example arguments;
- required mounts and their read/write behavior;
- environment variables;
- dependency/build provenance; and
- known resource expectations.

---

## 7. Validation: Prove the Contracts in Layers

Do not begin with a full, expensive training run. Test one boundary at a time.

### Gate 0: Verify the local application contract

Questions:

- Can the package artifact be built?
- Can that artifact be installed into a clean Python environment?
- Does `bayesflow-cli --help` work without relying on the source directory?

Reason: Docker cannot repair a package that only works accidentally from its
repository checkout.

### Gate 1: Verify the build context

Questions:

- Does `.dockerignore` exclude local environments, data, and generated output?
- Is the build context reasonably small?
- Does the image contain any forbidden files?

### Gate 2: Verify application installation

Candidate checks:

```bash
docker run --rm IMAGE --help
docker run --rm IMAGE models list
```

Pass conditions:

- the entrypoint exists;
- imports succeed;
- built-in models are found;
- Keras uses Torch; and
- commands return expected exit codes.

### Gate 3: Verify workspace behavior

Questions:

- Is the container working directory `/workspace`?
- Is a trusted external plugin discovered there?
- Are `checkpoints/`, `results/`, and `logs/` created on the host rather than
  trapped in the container layer?
- Are created files owned by the intended host user?

### Gate 4: Run a tiny functional smoke test

Use the smallest valid training configuration, not a scientifically meaningful
training job. Verify startup, one short execution path, output creation, and exit
status.

Important distinction:

> A smoke test proves that the container contract works. It does not prove model
> convergence, parameter recovery quality, or scientific validity.

### Gate 5: Test failure behavior

Try controlled invalid cases:

- an unknown model name;
- a missing plugin path;
- an unwritable output directory;
- an unavailable requested device; and
- an interrupted run.

Confirm that failures are understandable, return nonzero status, and do not
silently lose or corrupt durable state.

### Gate 6: Test reproducibility

Questions:

- Does rebuilding without source or dependency changes reuse appropriate cache?
- Can a clean machine build the same dependency set?
- Are base image and package updates deliberate?
- Can the resulting image be identified by immutable digest?

### Gate 7: Inspect security and supply-chain properties

At minimum:

- scan the image for known vulnerabilities;
- inventory installed packages, ideally with an SBOM;
- confirm the final process is non-root;
- confirm no credentials or private data exist in layers;
- review licenses before distribution; and
- remember that external plugins execute arbitrary trusted Python code.

---

## 8. The Project's Contract Worksheet

Use this table as the current design record. Change an answer only after checking
its downstream Docker consequence.

| Contract question | Current project answer | Docker consequence |
|---|---|---|
| What kind of process is it? | One-shot/batch CLI | No server port or HTTP health check |
| What is the stable executable? | `bayesflow-cli` | Exec-form `ENTRYPOINT` |
| What runtime is required? | Python 3.12+ | Python 3.12 base for first image |
| What code belongs in the image? | Installed CLI, `bayesflow_models`, built-ins | Build and install the project wheel |
| What does not belong in the image? | Data, plugins owned by researchers, notebooks, outputs, local venv | Exclude with `.dockerignore`; mount at runtime |
| How are dependencies declared? | Pinned in `pyproject.toml` | Install separately from the application wheel |
| How are CPU/CUDA indexes declared? | `tool.uv.sources` | Choose uv or explicitly configure pip indexes |
| Which Keras backend is required? | Torch | `KERAS_BACKEND=torch` before imports |
| What is the runtime working directory? | Researcher workspace | `WORKDIR /workspace` |
| How are external plugins found? | `--plugin-path` or current directory | Mount trusted code into `/workspace` |
| What persists? | Checkpoints, results, logs, real data | Bind mount or volume; never rely on container layer |
| What must be writable? | Workspace outputs and Matplotlib config | Non-root permissions plus writable `MPLCONFIGDIR` |
| Is networking required? | Not for current CLI behavior | Publish no ports |
| Is GPU required initially? | No | Validate CPU image first |
| What indicates health? | CLI exit status and expected artifacts | CLI smoke tests, not an HTTP health check |
| What is the trust boundary? | Plugins execute Python code | Load only trusted plugins; AST filtering is not sandboxing |

---

## 9. Decisions Still Required Before Writing the Final Dockerfile

The general contracts are clear, but these implementation decisions should not
be hidden inside Docker syntax:

1. **Dependency installer:** use pip or uv consistently?
2. **Reproducibility level:** are pinned direct dependencies sufficient, or will
   the project commit a complete lock file with transitive dependencies?
3. **CPU Torch source:** which approved index and exact CPU wheel strategy will
   the build use?
4. **Offline requirement:** must images build without internet, implying an
   internal package mirror or wheelhouse?
5. **Base-image pinning:** use a moving patch tag during development or an image
   digest for controlled releases?
6. **Workspace location:** mount the repository root or a separate experiment
   directory?
7. **Host ownership:** how will UID/GID mapping work for files created through a
   bind mount?
8. **Resource policy:** what CPU, memory, disk, and time limits are appropriate
   for smoke tests and real training?
9. **Distribution policy:** is the image local-only, shared within the lab, or
   published to a registry? This changes licensing and supply-chain needs.

For the first learning-oriented implementation, suitable provisional answers
are:

- CPU-only Linux x86-64;
- Python 3.12 slim base;
- one explicit package-management strategy;
- online dependency resolution from approved indexes;
- multi-stage wheel build;
- non-root runtime;
- one bind-mounted experiment workspace;
- no ports;
- `bayesflow-cli` entrypoint; and
- layered CLI and tiny-training validation.

---

## 10. A Reusable Checklist for Any Project

Before writing a Dockerfile for another project, answer these in order.

### A. Application discovery

- [ ] What process am I packaging?
- [ ] What is its stable entrypoint?
- [ ] What language/runtime and OS does it require?
- [ ] What files constitute the application?
- [ ] Does it work when installed outside the source checkout?

### B. Build-time contract

- [ ] What is the build context?
- [ ] What must `.dockerignore` exclude?
- [ ] Which base image and architectures are supported?
- [ ] Where are direct and transitive dependencies fixed?
- [ ] Which package indexes or registries are trusted?
- [ ] Are compilers or build tools required?
- [ ] What deployable artifact is produced?
- [ ] Can build and runtime stages be separated?
- [ ] How are private build credentials supplied without entering layers?
- [ ] How will the image be versioned, scanned, and reproduced?

### C. Runtime contract

- [ ] Is it a service, batch job, scheduled job, or interactive tool?
- [ ] What command and arguments start it?
- [ ] What configuration is supplied through arguments, environment, and files?
- [ ] What secrets are required, and how are they injected?
- [ ] What inputs are mounted, and are they read-only or read-write?
- [ ] What state must persist after container deletion?
- [ ] Which directories must be writable?
- [ ] Which user, group, and Linux capabilities are needed?
- [ ] What CPU, memory, disk, GPU, and shared-memory resources are required?
- [ ] What inbound and outbound networking is required?
- [ ] What are success, readiness, health, shutdown, and failure semantics?
- [ ] What logs, metrics, exit codes, and artifacts make behavior observable?

### D. Validation

- [ ] Can the image build from a clean checkout?
- [ ] Can its entrypoint run without mounting the source code?
- [ ] Are runtime inputs and persistent outputs correctly separated?
- [ ] Does it run as non-root with correct host file ownership?
- [ ] Do controlled failures return nonzero status and useful messages?
- [ ] Does interruption preserve the state the application promises to preserve?
- [ ] Are the actual CPU/GPU backend and dependency versions verified?
- [ ] Does a minimal functional smoke test pass?
- [ ] Are scientific or business correctness tests kept distinct from container
      smoke tests?
- [ ] Is the final image free of secrets, local data, and unnecessary build tools?

---

## 11. Recommended Learning and Implementation Sequence

For this project, proceed in small reviewable steps:

1. Review and approve the contract worksheet in this document.
2. Choose pip versus uv and the CPU Torch dependency source.
3. Build the project wheel locally and inspect what it contains.
4. Install that wheel in a clean temporary environment and prove the CLI is
   independent of the repository working directory.
5. Write and review `.dockerignore`.
6. Write the CPU multi-stage `Dockerfile`.
7. Build the image and validate `--help` and `models list` without a mount.
8. Run as non-root with one mounted experiment workspace.
9. Validate external plugin discovery and host-persistent outputs.
10. Run a tiny functional training smoke test and controlled failure tests.
11. Record the repeated runtime options in `compose.yaml`.
12. Scan, label, version, and document the image.
13. Only then design a separate GPU target and its host runtime requirements.

This sequence preserves the central DevOps discipline: make the contract
explicit, implement one boundary at a time, and prove each promise before adding
the next layer of complexity.
