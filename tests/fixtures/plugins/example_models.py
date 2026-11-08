
from bayesflow_models.interfaces import ModelSpec
from helper import DESCRIPTION

def create_test_simulator():
    return object()

def create_test_adapter():
    return object()


MODEL_SPECS = [
    ModelSpec(
        name="fixture_model",
        workflow="builtin",
        simulator_factory=create_test_simulator,
        adapter_factory=create_test_adapter,
        description=DESCRIPTION,
        family="test_family",
        par_names=["param1", "param2"],
        version="0.1.0",
    )]


