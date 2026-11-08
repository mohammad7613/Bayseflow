"""Verify that restarting the schedule does not discard optimizer state."""

import keras
import numpy as np
import pytest
from cognitive_models_codes.workflows import restart_learning_rate


def test_restarting_schedule_keeps_moments_and_iteration_counter():
    with keras.device("cpu"):
        variable = keras.Variable([1.0, 2.0])
        optimizer = keras.optimizers.Adam(
            learning_rate=keras.optimizers.schedules.CosineDecay(0.001, decay_steps=2)
        )
        optimizer.build([variable])
        # Pretend a completed run has nonzero moment buffers and two steps.
        for state in optimizer.variables:
            state.assign(keras.ops.ones_like(state) * 2)
        before = [
            keras.ops.convert_to_numpy(state).copy() for state in optimizer.variables
        ]
        assert float(keras.ops.convert_to_numpy(optimizer.learning_rate)) == 0
        restart_learning_rate(optimizer, 0.0005)
        assert float(
            keras.ops.convert_to_numpy(optimizer.learning_rate)
        ) == pytest.approx(0.0005)
        assert len(before) == len(optimizer.variables)
        for expected, actual in zip(before, optimizer.variables, strict=True):
            np.testing.assert_array_equal(expected, keras.ops.convert_to_numpy(actual))
