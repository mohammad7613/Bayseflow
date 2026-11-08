import sys
from types import ModuleType

import cli

from bayesflow_models.discovery import PluginRegistry
from bayesflow_models.interfaces import ModelSpec, Workflow
from bayesflow_models.evaluation.interfaces import EvaluationSpec, EvaluationBatch

def make_test_spec():
    return ModelSpec(
        name="cli_test_model",
        workflow="builtin",
        simulator_factory=lambda: object(),
        adapter_factory=lambda: object(),
        family="test",
        par_names=["parameter_a"],
        version="1.0",
        description="Model used to test CLI output",
    )


def make_test_evaluation_spec():
    return EvaluationSpec(
        name="cli_test_evaluation",
        model_name="cli_test_model",
        generate_batch=lambda spec, config: EvaluationBatch(
            posterior_conditions={"condition_a": 1},
            true_parameters={"parameter_a": 2},
            metadata={"artifact_id": "artifact-001"},
        ),
        description="Evaluation used to test CLI output",
    )

def make_test_registry(
    *,
    train_fn=lambda spec, config: None,
    resume_fn=lambda spec, config: None,
    recovery_fn=lambda spec, config: None,
):
    spec = make_test_spec()
    evaluation_spec = make_test_evaluation_spec()
    workflow = Workflow(
        name="builtin",
        train_fn=train_fn,
        resume_fn=resume_fn,
        recovery_fn=recovery_fn,
    )
    return PluginRegistry(
        models={spec.name: spec},
        workflows={workflow.name: workflow},
        evaluations={(spec.name, evaluation_spec.name): evaluation_spec}
    )


def arrange_cli(monkeypatch, argv, registry):
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


def test_parser_accepts_plugin_path_before_subcommand():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "--plugin-path",
            "/workspace/plugins",
            "models",
            "list",
        ]
    )

    assert args.plugin_path == ["/workspace/plugins"]
    assert args.command == "models"
    assert args.models_command == "list"


def test_models_list_prints_discovered_model(
    monkeypatch,
    capsys,
):
    arrange_cli(
        monkeypatch,
        ["bayesflow-cli", "models", "list"],
        make_test_registry(),
    )

    cli.main()

    output = capsys.readouterr().out
    assert "cli_test_model" in output
    assert "test" in output
    assert "Model used to test CLI output" in output


def test_train_dispatches_to_selected_workflow(monkeypatch):
    calls = []

    def fake_train(spec, config):
        calls.append((spec, config))

    arrange_cli(
        monkeypatch,
        [
            "bayesflow-cli",
            "train",
            "--models",
            "cli_test_model",
            "--n-sim",
            "7",
            "--epochs",
            "2",
            "--batch-size",
            "3",
        ],
        make_test_registry(train_fn=fake_train),
    )

    cli.main()

    assert len(calls) == 1
    spec, config = calls[0]
    assert spec.name == "cli_test_model"
    assert config["training"] == {
        "n_sim": 7,
        "epochs": 2,
        "batch_size": 3,
        "resume_epochs": 3,
    }


def test_resume_resolves_artifact_and_dispatches_to_workflow(
    monkeypatch,
):
    calls = []
    metadata = {
        "artifact_id": "artifact-001",
        "spec_name": "cli_test_model",
    }

    def fake_resume(spec, config):
        calls.append((spec, config))

    def fake_resolve_artifact_ref(ref, config):
        assert ref == "run-001"
        return metadata

    arrange_cli(
        monkeypatch,
        [
            "bayesflow-cli",
            "resume",
            "--artifacts",
            "run-001",
            "--epochs",
            "4",
            "--batch-size",
            "5",
            "--mode",
            "new",
        ],
        make_test_registry(resume_fn=fake_resume),
    )
    monkeypatch.setattr(
        cli.wf,
        "resolve_artifact_ref",
        fake_resolve_artifact_ref,
    )

    cli.main()

    assert len(calls) == 1
    spec, config = calls[0]
    assert spec.name == "cli_test_model"
    assert config["metadata"] == metadata
    assert config["mode"] == "new"
    assert config["training"]["resume_epochs"] == 4
    assert config["training"]["batch_size"] == 5


def test_recovery_resolves_artifact_and_dispatches_to_workflow(
    monkeypatch,
):
    calls = []
    metadata = {
        "artifact_id": "artifact-001",
        "spec_name": "cli_test_model",
    }

    def fake_recovery(spec, config):
        calls.append((spec, config))

    arrange_cli(
        monkeypatch,
        [
            "bayesflow-cli",
            "recovery",
            "--artifacts",
            "run-001",
            "--n-test-sims",
            "6",
            "--n-posterior-samples",
            "7",
            "--mode",
            "visualize",
        ],
        make_test_registry(recovery_fn=fake_recovery),
    )
    monkeypatch.setattr(
        cli.wf,
        "resolve_artifact_ref",
        lambda ref, config: metadata,
    )

    cli.main()

    assert len(calls) == 1
    spec, config = calls[0]
    assert spec.name == "cli_test_model"
    assert config["metadata"] == metadata
    assert config["mode"] == "visualize"
    assert config["recovery"]["n_test_sims"] == 6
    assert config["recovery"]["n_posterior_samples"] == 7


def test_mixed_recovery_resolves_artifact_and_calls_evaluator(
    monkeypatch,
    capsys,
):
    calls = []
    metadata = {
        "artifact_id": "artifact-001",
        "spec_name": "cli_test_model",
    }

    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        return {"artifact_id": "artifact-001"}

    fake_evaluation_module = ModuleType(
        "bayesflow_models.mixed_tta_evaluation"
    )
    fake_evaluation_module.evaluate_mixed_tta_artifact_with_bf_recovery = (
        fake_evaluate
    )

    arrange_cli(
        monkeypatch,
        [
            "bayesflow-cli",
            "mixed-recovery",
            "--artifacts",
            "run-001",
            "--n-test-sims",
            "8",
            "--n-posterior-samples",
            "9",
            "--trials-per-tta",
            "10",
            "--no-shuffle-tta",
            "--seed",
            "11",
            "--mode",
            "save_only",
        ],
        make_test_registry(),
    )
    monkeypatch.setattr(
        cli.wf,
        "resolve_artifact_ref",
        lambda ref, config: metadata,
    )
    monkeypatch.setitem(
        sys.modules,
        "bayesflow_models.mixed_tta_evaluation",
        fake_evaluation_module,
    )

    cli.main()

    assert len(calls) == 1
    call = calls[0]
    assert call["artifact_ref"] == "run-001"
    assert call["spec"].name == "cli_test_model"
    assert call["n_test_sims"] == 8
    assert call["n_posterior_samples"] == 9
    assert call["trials_per_tta"] == 10
    assert call["shuffle_tta"] is False
    assert call["seed"] == 11
    assert call["config"]["mode"] == "save_only"
    assert "Mixed-TTA recovery for artifact-001" in capsys.readouterr().out
