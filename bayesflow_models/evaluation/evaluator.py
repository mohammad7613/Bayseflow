from .interfaces import EvaluationResult
from .metrics import (
    build_recovery_result_payload,
    compute_recovery_metrics,
)


class RecoveryEvaluator:
    name = "recovery"

    def compute(self, *, posterior_samples, batch, parameter_names):
        metrics, summaries = compute_recovery_metrics(
            normalized_draws=posterior_samples,
            true_parameters=batch.true_parameters,
        )

        payload = build_recovery_result_payload(metrics=metrics)
        payload["plot_settings"] = {
            "point_estimate": "median",
            "credible_interval": 0.95,
        }

        return EvaluationResult(
            result_payload=payload,
            posterior_summary=summaries,
        )