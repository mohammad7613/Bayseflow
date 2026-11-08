from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from bayesflow_models.interfaces import ModelSpec
from bayesflow_models.interfaces import Workflow
from bayesflow_models.evaluation import EvaluationSpec
from bayesflow_models.builtin_workflows import BUILTIN_WORKFLOWS
from hashlib import sha256
from contextlib import contextmanager
import sys

import ast


PLUGIN_MARKERS = {
    "MODEL_SPECS",
    "WORKFLOWS",
    "EVALUATIONS",
    "register_models",
    "register_workflows",
    "register_evaluations",
}
    


@dataclass(frozen=True)
class PluginRegistry:
    models: dict[str, ModelSpec]
    workflows: dict[str, Workflow]
    evaluations: dict[tuple[str, str], EvaluationSpec] # This item can be reduced later to a dict[str, EvaluationSpec] if we assume that the evaluation name is unique across all models. But for now, we will keep it as a tuple of (model_name, evaluation_name) to avoid any potential conflicts.


def file_has_plugin_markers(py_file: Path) -> bool:
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"),
        filename=str(py_file))
    except SyntaxError:
        return False

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in PLUGIN_MARKERS:
                return True

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in PLUGIN_MARKERS:
                    return True

        if isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id in PLUGIN_MARKERS:
                return True
    return False



@contextmanager
def plugin_import_context(plugin_dir: Path):
    plugin_dir_str = str(plugin_dir)
    sys.path.insert(0, plugin_dir_str)
    try:
        yield
    finally:
        try:
            sys.path.remove(plugin_dir_str)
        except ValueError:
            pass

def resolve_plugin_paths(plugin_paths=None) -> list[Path]:
    if plugin_paths:
        raw_paths = plugin_paths
    else:
        raw_paths = [Path.cwd()]

    paths = []
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"Plugin path does not exist:{path}")

        if not path.is_dir():
            raise ValueError(f"Plugin path must be a directory:{path}")

        paths.append(path)

    return paths



def _validate_model_specs(
    value,
    source: str,
) -> list[ModelSpec]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{source} must provide a list or tuple of ModelSpec objects."
        )

    if not all(isinstance(item, ModelSpec) for item in value):
        raise ValueError(
            f"Every item from {source} must be a ModelSpec instance."
        )

    return list(value)


def _validate_workflows(
    value,
    source: str,
) -> list[Workflow]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{source} must provide a list or tuple of Workflow objects."
        )

    if not all(isinstance(item, Workflow) for item in value):
        raise ValueError(
            f"Every item from {source} must be a Workflow instance."
        )

    return list(value)


def _validate_evaluations(value, source: str) -> list[EvaluationSpec]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{source} must provide a list or tuple")
    if not all(isinstance(item, EvaluationSpec) for item in value):
        raise ValueError(f"Every item from {source} must be an EvaluationSpec")
    return list(value)


def _load_plugin_module(py_file: Path, plugin_dir: Path):
    # module_name = py_file.stem

    resolved_file = py_file.resolve()
    path_digest = sha256(
          str(resolved_file).encode("utf-8")
      ).hexdigest()[:12]

    module_name = (f"bayesflow_external_{py_file.stem}_{path_digest}")

    mod_spec = spec_from_file_location(
        module_name,
        resolved_file,
    )

    if mod_spec is None or mod_spec.loader is None:
        raise ImportError(
            f"Could not load plugin file: {py_file}"
        )

    module = module_from_spec(mod_spec)

    with plugin_import_context(plugin_dir):
        mod_spec.loader.exec_module(module)

    return module


def discover_plugins(plugin_paths=None) -> PluginRegistry:
    models: dict[str, ModelSpec] = {}

    workflows: dict[str, Workflow] = {
        workflow.name: workflow
        for workflow in BUILTIN_WORKFLOWS
    }

    evaluations: dict[(str,str), EvaluationSpec] = {}

    for plugin_path in resolve_plugin_paths(plugin_paths):
        for py_file in sorted(plugin_path.glob("*.py")):
            if not file_has_plugin_markers(py_file):
                continue

            module = _load_plugin_module(
                py_file,
                plugin_path,
            )

            for model in _collect_specs_from_module(module):
                if model.name in models:
                    raise ValueError(
                        f"Duplicate model name: {model.name}"
                    )

                models[model.name] = model

            for workflow in _collect_workflows_from_module(module):
                if workflow.name in workflows:
                    raise ValueError(
                        f"Duplicate workflow name: "
                        f"{workflow.name}"
                    )

                workflows[workflow.name] = workflow

            for evaluation in _collect_evaluations_from_module(module):
                if (evaluation.model_name, evaluation.name) in evaluations:
                    raise ValueError(
                        f"Duplicate evaluation name: "
                        f"{evaluation.name}"
                    )

                evaluations[(evaluation.model_name, evaluation.name)] = evaluation

    return PluginRegistry(
        models=models,
        workflows=workflows,
        evaluations=evaluations 
    )




# I have to change the the discovery startegy in a way that it does not impose the attribute name whether it Workflows or ModelSpec. It should be more type-dependent 
def _collect_specs_from_module(module):
    if hasattr(module, "MODEL_SPECS"):
        specs = module.MODEL_SPECS

        return _validate_model_specs(specs, source=f"MODEL_SPECS in {module.__name__}")

    if hasattr(module, "register_models"):
        # Use the built-in callable() check
        #if not isinstance(module.register_models, Callable[...]):
        if not callable(module.register_models):
            raise ValueError(f"register_models in {module.__name__} must be a callable.")
            
        return _validate_model_specs(module.register_models(), source=f"register_models in {module.__name__}")

    return []

def discover_model_specs(plugin_paths=None) -> dict[str,ModelSpec]:

    specs: dict[str, ModelSpec] = {}


    for plugin_path in resolve_plugin_paths(plugin_paths):
        for py_file in sorted(plugin_path.glob("*.py")):
            if not file_has_plugin_markers(py_file):
                continue
            mod_spec = spec_from_file_location(py_file.stem, py_file)
            if mod_spec is None or mod_spec.loader is None:
                raise ImportError(f"Could not load plugin file: {py_file}")

            module = module_from_spec(mod_spec)
            with plugin_import_context(plugin_path):
                mod_spec.loader.exec_module(module)

            for spec in _collect_specs_from_module(module):
                if spec.name in specs:
                    raise ValueError(f"Duplicate model spec name:{spec.name}")
                specs[spec.name] = spec

    return specs


def _collect_workflows_from_module(module) -> list[Workflow]:
    if hasattr(module, "WORKFLOWS"):
        workflows = module.WORKFLOWS
        
        # 1. Check if it is a list
        if not isinstance(workflows, list):
            raise ValueError(f"WORKFLOWS in {module.__name__} must be a list.")
            
        # 2. Check if everything INSIDE the list is a ModelSpec
        if not all(isinstance(workflow, Workflow) for workflow in workflows):
            raise ValueError(f"All items in WORKFLOWS in {module.__name__} must be Workflow instances.")
            
        return _validate_workflows(workflows, source=f"WORKFLOWS in {module.__name__}")

    if hasattr(module, "register_workflows"):
        # Use the built-in callable() check
        #if not isinstance(module.register_workflow, Callable[...,Workflow]):
        if not callable(module.register_workflows):
            raise ValueError(f"register_workflows in {module.__name__} must be a function returing workflow plugin.")
            
        return _validate_workflows(module.register_workflows(), source=f"register_workflows in {module.__name__}")
    return []

def discover_workflows(plugin_paths=None) -> dict[str,Workflow]:
    workflows: dict[str,Workflow] = {}

    if not BUILTIN_WORKFLOWS:
        raise RuntimeError("The package contains no default workflow")

    workflows = {
    workflow.name: workflow
    for workflow in BUILTIN_WORKFLOWS}
  

    for plugin_path in resolve_plugin_paths(plugin_paths):
        for py_file in sorted(plugin_path.glob("*.py")):
            if not file_has_plugin_markers(py_file):
                continue
            mod_spec = spec_from_file_location(py_file.stem, py_file)
            if mod_spec is None or mod_spec.loader is None:
                raise ImportError(f"Could not load plugin file: {py_file}")

            module = module_from_spec(mod_spec)

            with plugin_import_context(plugin_path):
                mod_spec.loader.exec_module(module)

            for workflow in _collect_workflows_from_module(module):
                if workflow.name in workflows:
                    raise ValueError(f"Duplicate workflow name: {workflow.name}")
                workflows[workflow.name] = workflow

    return workflows



def _collect_evaluations_from_module(module):
    if hasattr(module, "EVALUATIONS"):
        evaluations = module.EVALUATIONS

        return _validate_evaluations(evaluations, source=f"EVALUATIONS in {module.__name__}")

    if hasattr(module, "register_evaluations"):
        # Use the built-in callable() check
        #if not isinstance(module.register_evaluations, Callable[...]):
        if not callable(module.register_evaluations):
            raise ValueError(f"register_evaluations in {module.__name__} must be a callable.")
            
        return _validate_evaluations(module.register_evaluations(), source=f"register_evaluations in {module.__name__}")

    return []


def discover_evaluations(plugin_paths=None) -> dict[tuple[str, str], EvaluationSpec]:
    evaluations: dict[tuple[str, str], EvaluationSpec] = {}

    for plugin_path in resolve_plugin_paths(plugin_paths):
        for py_file in sorted(plugin_path.glob("*.py")):
            if not file_has_plugin_markers(py_file):
                continue
            mod_spec = spec_from_file_location(py_file.stem, py_file)
            if mod_spec is None or mod_spec.loader is None:
                raise ImportError(f"Could not load plugin file: {py_file}")

            module = module_from_spec(mod_spec)

            with plugin_import_context(plugin_path):
                mod_spec.loader.exec_module(module)

            for evaluation in _collect_evaluations_from_module(module):
                key = (evaluation.model_name, evaluation.name)
                if key in evaluations:
                    raise ValueError(f"Duplicate evaluation spec name: {evaluation.name} for model: {evaluation.model_name}")
                evaluations[key] = evaluation

    return evaluations