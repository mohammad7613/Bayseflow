from bayesflow_models.interfaces import Workflow
from bayesflow_models.workflow import (
    recovery_from_artifact,
    resume_from_artifact,
    train_from_spec,
)


BUILTIN_WORKFLOWS: tuple[Workflow, ...] = (
    Workflow(
        name="builtin",
        train_fn=train_from_spec,
        resume_fn=resume_from_artifact,
        recovery_fn=recovery_from_artifact,
    ),
)