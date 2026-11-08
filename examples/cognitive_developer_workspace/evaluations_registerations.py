"""Discoverable scenario declarations; both use the package's SAME recovery engine."""

# from cognitive_models_codes.scenarios import (
#     generate_homogeneous_tta_batch,
#     generate_mixed_tta_batch,
# )

from cognitive_models_codes.myscenarios import generate_mixed_tta_batch, generate_homogeneous_tta_batch

from bayesflow_models.evaluation import EvaluationSpec

EVALUATIONS = [
    EvaluationSpec(
        name="homogeneous-tta",
        model_name="model_DC1",
        generate_batch=generate_homogeneous_tta_batch,
        description="Baseline recovery with one nominal TTA per participant.",
    ),
    EvaluationSpec(
        name="mixed-tta",
        model_name="model_DC1",
        generate_batch=generate_mixed_tta_batch,
        description="Generalization to mixed nominal TTAs within each participant.",
    ),
]
