"""Connect the scientific model to BayesFlow's online-training interface.

The wrapper creates data; the existing package's built-in workflow trains the
network. Each factory call owns a fresh RNG instead of a shared module RNG.
"""

import os

import bayesflow as bf
import numpy as np

from .designs import homogeneous_schedule
from .science import (
    PARAMETER_NAMES,
    TRAINING_TRIALS,
    TTA_LEVELS,
    sample_prior,
    simulate_dataset,
)



def make_training_simulator():
    """Return a BayesFlow simulator exposing .sample(batch_size).

    Like the original meta_fn, choose one nominal TTA for each training batch;
    participants in that batch have separate parameters and trial-level jitter.
    Keep 60 trials to reproduce the original training design.
    """
    rng = np.random.default_rng(int(os.environ.get("MODEL_DC_SIMULATION_SEED", "2023")))

    def sample_batch(batch_shape):
        if len(batch_shape) != 1 or batch_shape[0] < 1:
            raise ValueError(
                "This example expects a positive one-dimensional dataset batch."
            )
        n_datasets = batch_shape[0]
        nominal_tta = float(rng.choice(TTA_LEVELS))
        schedule = homogeneous_schedule(TRAINING_TRIALS, nominal_tta)
        parameters = {name: [] for name in PARAMETER_NAMES}
        observations, covariates = [], []

        for _ in range(n_datasets):
            participant = sample_prior(rng)
            simulated = simulate_dataset(participant, schedule, rng)
            observations.append(simulated["x"])
            covariates.append(simulated["tta_per_trial"])
            for name in PARAMETER_NAMES:
                parameters[name].append([participant[name]])

        # Scalar parameters retain a feature axis, which the adapter concatenates.
        return {
            **{name: np.asarray(values) for name, values in parameters.items()},
            "x": np.asarray(observations),
            "tta_per_trial": np.asarray(covariates),
            "number_of_trials": np.full((n_datasets, 1), TRAINING_TRIALS),
            "tta_condition": np.full((n_datasets, 1), nominal_tta),
        }
    # return bf.simulators.make_simulator([prior_DC_simplest_model,ddm_DC_TwoBoundary_simplest],meta_fn=meta1)
    return bf.simulators.LambdaSimulator(sample_batch, is_batched=True)



def make_simulator_simple():
    
    def meta():
        """Choose one nominal TTA shared by the training batch."""
        return {
            "number_of_trials": TRAINING_TRIALS,
            "tta_condition": float(np.random.default_rng(2023).choice(TTA_LEVELS)),
        }
    # Create the simulator with trial-wise TTA tracking
    model_DC = bf.simulators.make_simulator(
        [sample_prior(np.random.default_rng(2023)),simulate_dataset],
        meta_fn=meta,)

    return model_DC
    