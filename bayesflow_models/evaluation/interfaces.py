
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

from bayesflow_models.interfaces import ModelSpec


from typing import Protocol, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure



@dataclass(frozen=True)
class EvaluationBatch:
    posterior_conditions: dict[str, Any] # it is the data given to approximate posterior network 
    true_parameters: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


GenerateEvaluationBatch = Callable[
    [dict[str, Any]],
    EvaluationBatch,
]


@dataclass(frozen=True)
class EvaluationSpec:
    name: str
    model_name: str
    generate_batch: GenerateEvaluationBatch
    description: str = ""
    evaluator: Evaluator | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """Numerical results; arrays remain separate from JSON data."""

    result_payload: dict[str, Any]
    posterior_summary: dict[str, dict[str, np.ndarray]]


class Evaluator(Protocol):
    """Contract for assessing already-generated posterior samples."""

    name: str

    def compute(
        self,
        *,
        posterior_samples: dict[str, np.ndarray],
        batch: EvaluationBatch,
        parameter_names: list[str],
    ) -> EvaluationResult:
        ...