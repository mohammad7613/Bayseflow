"""Serializable BayesFlow encoding, shared by all designs of this model."""

import bayesflow as bf

from .science import PARAMETER_NAMES


def make_adapter():
    """Encode each trial as [response/RT, actual jittered TTA].

    Training supplies named parameters (N, 1), observations (N, T, 1), and
    covariates (N, T, 1). The summary network receives (N, T, 2).
    There is no CPP feature. Trial count is supplied for bookkeeping, but is
    not a network feature. The old unused trial-count broadcast is unnecessary.

    Only built-in transforms are used, so the saved adapter loads in a fresh
    process. After training has initialized its parameter split indices, the
    checkpoint can adapt conditions without any true parameter fields.
    """
    return (
        bf.Adapter()
        .convert_dtype("float64", "float32")
        .concatenate(PARAMETER_NAMES, into="inference_variables")
        .concatenate(["x", "tta_per_trial"], into="summary_variables")
        .as_set("summary_variables")
    )
