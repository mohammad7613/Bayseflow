"""CLI entry point for model discovery; scientific code lives in the package below.

The existing filename is retained. Pass THIS directory to --plugin-path; the
package does not automatically discover registrations inside nested folders.
"""

import os

import keras
from cognitive_models_codes.adapters import make_adapter
from cognitive_models_codes.science import PARAMETER_NAMES
from cognitive_models_codes.simulators import make_training_simulator
from cognitive_models_codes.mysimulator import make_simulator
from cognitive_models_codes.myadapter import adopt_TrialWise
from cognitive_models_codes.myscience import PAR_NAMES
from bayesflow_models.interfaces import ModelSpec

# Explicitly requested by the supplied training scripts, not an unconditional
# import-time RNG reset. Simulations have their own independent local seed.
if "MODEL_DC_NETWORK_SEED" in os.environ:
    keras.utils.set_random_seed(int(os.environ["MODEL_DC_NETWORK_SEED"]))

MODEL_SPECS = [
    ModelSpec(
        name="model_DC",
        # Explicit workspace extension: standard training, nonzero-LR resume.
        workflow="dc_example",
        simulator_factory=make_training_simulator,
        adapter_factory=make_adapter,
        family="ddm",
        par_names=list(PARAMETER_NAMES),
        version="1.0.0-legacy-dc-example",
        description="Seven-parameter article DC reproduction; homogeneous training, RT/TTA inputs.",
    ),
        ModelSpec(
        name="model_DC1",
        # Explicit workspace extension: standard training, nonzero-LR resume.
        workflow="builtin",
        simulator_factory=make_simulator,
        adapter_factory=adopt_TrialWise,
        family="ddm",
        par_names=list(PAR_NAMES),
        version="1.0.0-legacy-dc-example",
        description="Seven-parameter article DC reproduction; homogeneous training, RT/TTA inputs.",
    ),
]



