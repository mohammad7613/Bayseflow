from pathlib import Path

import pytest

from bayesflow_models.discovery import (
    discover_model_specs,
    resolve_plugin_paths,
    discover_plugins,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]




PLUGIN_FIXTURES = Path(__file__).parent / "fixtures" / "plugins"


def test_empty_plugin_path_defaults_to_current_directory(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    paths = resolve_plugin_paths([])

    assert paths == [tmp_path.resolve()]


def test_explicit_plugin_directory_is_resolved():
    paths = resolve_plugin_paths([PLUGIN_FIXTURES])

    assert paths == [PLUGIN_FIXTURES.resolve()]


def test_missing_plugin_directory_is_rejected(tmp_path):
    missing_path = tmp_path / "does-not-exist"

    with pytest.raises(FileNotFoundError) as exception_info:
        resolve_plugin_paths([missing_path])

    error = exception_info.value

    assert "Plugin path does not exist" in str(error)
    assert str(missing_path) in str(error)


def test_plugin_path_must_be_a_directory(tmp_path):
    plugin_file = tmp_path / "plugin.py"
    plugin_file.write_text("MODEL_SPECS = []",
    encoding="utf-8")

    with pytest.raises(ValueError, match="must be a directory"):
        resolve_plugin_paths([plugin_file])


def test_external_model_plugin_is_discovered():
    specs = discover_model_specs([PLUGIN_FIXTURES])

    assert set(specs) == {"fixture_model"}

    spec = specs["fixture_model"]

    assert spec.name == "fixture_model"
    assert spec.workflow == "builtin"
    assert spec.family == "test_family"
    assert spec.par_names == ["param1", "param2"]
    assert spec.description == "Model loaded from an external test plugin"

def test_unmarked_python_file_is_not_imported(tmp_path):
    dangerous_file = tmp_path / "unrelated.py"

    dangerous_file.write_text('raise RuntimeError("This file must not be imported")', encoding="utf-8",)
    specs = discover_model_specs([tmp_path])

    assert "unrelate" not in specs


def test_empty_workspace_has_no_model_specs(tmp_path):
      specs = discover_model_specs([tmp_path])

      assert specs == {}

# def test_current_internal_models_are_available():
#     specs = discover_model_specs([])

#     assert {"model_DC", "model_DF", "model_SF"} <= set(specs)



# def test_article_workspace_provides_article_models():
#     specs = discover_model_specs(
#         [ARTICLE_PLUGINS]
#     )

#     assert {"model_DC", "model_DF", "model_SF"} <= set(specs)


def test_plugin_module_executes_once(tmp_path):
    counter_file = tmp_path / "import-count.txt"
    plugin_file = tmp_path / "combined_plugin.py"

    plugin_file.write_text(
        f'''from pathlib import Path

from bayesflow_models.interfaces import ModelSpec

counter_path = Path({str(counter_file)!r})

if counter_path.exists():
    count = int(counter_path.read_text())
else:
    count = 0

counter_path.write_text(str(count + 1))


def create_simulator():
    return object()


def create_adapter():
    return object()


MODEL_SPECS = [
    ModelSpec(
        name="single_import_model",
        workflow="builtin",
        simulator_factory=create_simulator,
        adapter_factory=create_adapter,
        family="test",
        par_names=["parameter_a"],
        version="1.0",
    )
]
''',
        encoding="utf-8",
    )

    registry = discover_plugins([tmp_path])

    assert "single_import_model" in registry.models
    assert "builtin" in registry.workflows
    assert counter_file.read_text() == "1"




def test_plugin_module_duplicate_moudle(tmp_path):
    counter_file = tmp_path / "import-count.txt"
    plugin_file = tmp_path / "combined_plugin.py"

    plugin_file.write_text(
        f'''from pathlib import Path

from bayesflow_models.interfaces import ModelSpec

counter_path = Path({str(counter_file)!r})

if counter_path.exists():
    count = int(counter_path.read_text())
else:
    count = 0

counter_path.write_text(str(count + 1))


def create_simulator():
    return object()


def create_adapter():
    return object()


MODEL_SPECS = [
    ModelSpec(
        name="single_import_model",
        workflow="builtin",
        simulator_factory=create_simulator,
        adapter_factory=create_adapter,
        family="test",
        par_names=["parameter_a"],
        version="1.0",
    ),
    ModelSpec(
            name="single_import_model",
            workflow="builtin",
            simulator_factory=create_simulator,
            adapter_factory=create_adapter,
            family="test",
            par_names=["parameter_a"],
            version="1.0",
        )
]
''',
        encoding="utf-8",
    )


    with pytest.raises(ValueError, match= "Duplicate model name: single_import_model"):
        discover_plugins([tmp_path])

