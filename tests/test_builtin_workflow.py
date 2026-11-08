from bayesflow_models.discovery import discover_workflows


def test_builtin_workflow_is_available(tmp_path):
    workflows = discover_workflows([tmp_path])

    assert set(workflows) == {"builtin"}

    workflow = workflows["builtin"]

    assert workflow.name == "builtin"
    assert callable(workflow.train_fn)
    assert callable(workflow.resume_fn)
    assert callable(workflow.recovery_fn)