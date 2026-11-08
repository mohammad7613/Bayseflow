"""Optional workspace workflow extension, discovered through the public registry.

This is needed only to restart an exhausted learning-rate schedule on resume.
Training itself delegates to package training; evaluate uses the package engine.
"""

from cognitive_models_codes.workflows import resume_model, train_model

from bayesflow_models.interfaces import Workflow
from bayesflow_models.workflow import recovery_from_artifact



WORKFLOWS = [
    Workflow(
        name="dc_example",
        train_fn=train_model,
        resume_fn=resume_model,
        recovery_fn=recovery_from_artifact,
    ),
]
