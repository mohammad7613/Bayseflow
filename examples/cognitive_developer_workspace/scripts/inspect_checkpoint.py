"""Inspect a real checkpoint in a fresh process; write only the requested JSON.

Called by smoke_test.py. The weight digest covers trainable parameters only,
so changing running statistics alone cannot masquerade as resumed learning.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("KERAS_BACKEND", "torch")

    # Import BayesFlow to register its serialized networks before Keras loads.
    import bayesflow  # noqa: F401
    import keras
    import numpy as np

    with keras.device("cpu"):
        model = keras.saving.load_model(args.checkpoint)
    digest = hashlib.sha256()
    for variable in model.trainable_variables:
        values = np.asarray(keras.ops.convert_to_numpy(variable))
        digest.update(str((values.shape, values.dtype.str)).encode())
        digest.update(values.tobytes())
    optimizer = model.optimizer
    state = {
        "checkpoint": str(args.checkpoint.resolve()),
        "trainable_weights_sha256": digest.hexdigest(),
        "trainable_variable_count": len(model.trainable_variables),
        "optimizer_iterations": int(keras.ops.convert_to_numpy(optimizer.iterations)),
        "learning_rate": float(keras.ops.convert_to_numpy(optimizer.learning_rate)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
