
import pytest



from bayesflow_models.interfaces import ModelSpec, Workflow
from cli import require_model, require_workflow



def make_model_spec(
    name="model_DC",
    workflow="builtin",
):
    return ModelSpec(
        name=name,
        workflow=workflow,
        simulator_factory=lambda: object(),
        adapter_factory=lambda: object(),
        family="test",
        par_names=["parameter_a"],
        version="1.0",
    )


def make_workflow(
    name="model_DC",
):
    return Workflow(name=name, train_fn=lambda spec, config: None, resume_fn=lambda spec, config: None, recovery_fn=lambda spec, config: None) 



def test_require_model_returns_requested_model():
    spec = make_model_spec(name="model_DC")
    specs = {spec.name: spec}

    result = require_model(specs, "model_DC")

    assert result is spec  


def test_require_model_raises_error_for_missing_model():
    specs = {}

    with pytest.raises(ValueError) as exc_info:
        require_model(specs, "non_existent_model")


    assert str(exc_info.value) == "Unknown model 'non_existent_model'. Available models: none. Check the mounted workspace or --plugin-path."
    print(exc_info.value)



def test_require_workflow_returns_requested_workflow():
    workflow = make_workflow(name="model_DC")
    workflows = {workflow.name: workflow}

    result = require_workflow(workflows, "model_DC", model_name="model_DC")

    assert result is workflow


def test_require_workflow_raises_error_for_missing_workflow():
    workflows = {}

    with pytest.raises(ValueError) as exc_info:
        require_workflow(workflows, "non_existent_workflow", model_name="model_DC")

    assert str(exc_info.value) == "Unknown workflow 'non_existent_workflow' required by model 'model_DC'. Available workflows: none."


        