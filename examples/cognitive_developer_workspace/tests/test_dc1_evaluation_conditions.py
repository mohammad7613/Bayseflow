"""Evaluation conditions must work with model_DC1's checkpoint adapter."""

import numpy as np
import pytest

from cognitive_models_codes.myadapter import adopt_TrialWise
from cognitive_models_codes.myscenarios import (
    generate_homogeneous_tta_batch,
    generate_mixed_tta_batch,
)


@pytest.mark.parametrize("n_datasets", [1, 8])
@pytest.mark.parametrize(
    "generate", [generate_homogeneous_tta_batch, generate_mixed_tta_batch]
)
def test_conditions_preserve_batch_and_trial_axes(generate, n_datasets):
    batch = generate({"n_datasets": n_datasets})
    conditions = batch.posterior_conditions
    n_trials = conditions["x"].shape[1]

    adapter = adopt_TrialWise()
    # Training initializes the parameter-concatenation indices saved in checkpoints.
    adapter(
        {
            **conditions,
            **{
                name: np.asarray(values).reshape(n_datasets, 1)
                for name, values in batch.true_parameters.items()
            },
        }
    )
    adapted = adapter(conditions, strict=False)

    assert "inference_variables" not in adapted
    assert adapted["summary_variables"].shape == (n_datasets, n_trials, 2)
    np.testing.assert_array_equal(
        adapted["summary_variables"],
        np.concatenate([conditions["x"], conditions["tta_per_trial"]], axis=-1),
    )
    assert adapted["number_of_trials"].shape == (n_datasets, n_trials, 1)
    np.testing.assert_allclose(adapted["number_of_trials"], np.sqrt(n_trials))
