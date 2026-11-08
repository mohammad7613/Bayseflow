"""Protect participant sharing, trial designs, and the package batch contract."""

import json

import numpy as np
import pytest
from cognitive_models_codes import scenarios
from cognitive_models_codes.science import PARAMETER_NAMES, TTA_LEVELS
from models_registeration import MODEL_SPECS


@pytest.mark.parametrize("mixed", [False, True])
def test_scenarios_are_seeded_and_match_the_batch_contract(mixed):
    generate = (
        scenarios.generate_mixed_tta_batch
        if mixed
        else scenarios.generate_homogeneous_tta_batch
    )
    options = {"n_datasets": 2, "seed": 7, "posterior_seed": 8}
    first, second = generate(MODEL_SPECS[0], options), generate(MODEL_SPECS[0], options)
    assert set(first.true_parameters) == set(PARAMETER_NAMES)
    assert not set(first.true_parameters).intersection(first.posterior_conditions)
    assert first.posterior_conditions["x"].shape == (2, 60, 1)
    assert first.posterior_conditions["tta_per_trial"].shape == (2, 60, 1)
    assert first.posterior_conditions["number_of_trials"].shape == (2, 1)
    for key, values in first.posterior_conditions.items():
        np.testing.assert_array_equal(values, second.posterior_conditions[key])
        assert np.all(np.isfinite(values))
    for key, values in first.true_parameters.items():
        assert values.shape == (2,)
        np.testing.assert_array_equal(values, second.true_parameters[key])
    json.dumps(first.metadata, allow_nan=False)
    assert first.metadata["resolved_options"]["dt"] == 0.005


def test_mixed_generator_uses_one_parameter_vector_for_the_entire_participant(
    monkeypatch,
):
    calls = []

    def record_simulation(parameters, schedule, rng, *, dt):
        calls.append((dict(parameters), schedule.copy()))
        return {"x": np.zeros((len(schedule), 1)), "tta_per_trial": schedule[:, None]}

    monkeypatch.setattr(scenarios, "simulate_dataset", record_simulation)
    batch = scenarios.generate_mixed_tta_batch(
        MODEL_SPECS[0], {"n_datasets": 2, "seed": 7}
    )
    assert (
        len(calls) == 2
    )  # One complete dataset per participant, not separate fits per TTA.
    for index, (parameters, schedule) in enumerate(calls):
        for name in PARAMETER_NAMES:
            assert parameters[name] == batch.true_parameters[name][index]
        levels, counts = np.unique(schedule, return_counts=True)
        np.testing.assert_array_equal(levels, TTA_LEVELS)
        np.testing.assert_array_equal(counts, [15, 15, 15, 15])


def test_homogeneous_generator_repeats_one_nominal_condition(monkeypatch):
    schedules = []

    def record_simulation(parameters, schedule, rng, *, dt):
        schedules.append(schedule)
        return {"x": np.zeros((len(schedule), 1)), "tta_per_trial": schedule[:, None]}

    monkeypatch.setattr(scenarios, "simulate_dataset", record_simulation)
    scenarios.generate_homogeneous_tta_batch(MODEL_SPECS[0], {"n_datasets": 3})
    assert len(schedules) == 3
    assert all(len(np.unique(schedule)) == 1 for schedule in schedules)


@pytest.mark.parametrize(
    "options",
    [
        {"n_datasets": 0},
        {"n_datasets": True},
        {"trials_per_tta": -1},
        {"tta_levels": [1.0]},
        {"tta_levels": [2.5, 2.5]},
        {"seed": -1},
        {"dt": 0},
        {"shuffle": "yes"},
        {"typo_option": 3},
    ],
)
def test_bad_scenario_settings_fail_clearly(options):
    with pytest.raises((ValueError, TypeError)):
        scenarios.generate_mixed_tta_batch(MODEL_SPECS[0], options)
