from .myscience import ddm_DC_TwoBoundary_TrialWise_simplest, prior_DC_simplest_model, CONDITIONS,RNG

import bayesflow as bf

def meta():
    """
    Meta-function generates simulation context.
    
    Randomly selects ONE TTA condition per simulation.
    This gets passed to simulator, which generates data for that block
    and also returns the actual TTA values for each trial (with jitter).
    """
    tta_flag = RNG.choice(CONDITIONS)
    return {
        "number_of_trials": 60,
        "tta_condition": tta_flag,
    }




def make_simulator():

    model_DC_TrialWise = bf.simulators.make_simulator(
    [prior_DC_simplest_model, ddm_DC_TwoBoundary_TrialWise_simplest],
    meta_fn=meta,
    )

    return model_DC_TrialWise