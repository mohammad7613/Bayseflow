"""Serializable BayesFlow encoding, shared by all designs of this model."""

import bayesflow as bf

from .myscience import prior_DC_simplest_model, PAR_NAMES

def adopt_TrialWise():
    """
    Adapter for TRIAL-WISE concatenation approach.
    
    Key differences from original:
    1. Concatenates x (RT, CPP) with tta_per_trial on axis=1
       Result: (N_trials, 3) with [RT, CPP, TTA] per trial
    2. Treats this augmented array as an exchangeable set
    3. No separate condition_variables (TTA is in summary_variables)
    
    This enables the Summary Network to see [RT, CPP, TTA] together,
    allowing it to learn conditional features.
    
    Args:
        p: Prior dict (to get parameter names)
    
    Returns:
        BayesFlow Adapter configured for trial-wise concatenation
    """
    adapter = (
        bf.Adapter()
        # Step 1: Broadcast trial count to both x and tta_per_trial
        .broadcast("number_of_trials", to="x")
        .broadcast("number_of_trials", to="tta_per_trial")
        
        # Step 2: CONCATENATE x and tta_per_trial on axis 1
        # Input: x=(N,2), tta_per_trial=(N,1)
        # Output: x_augmented=(N,3) with [RT, CPP, TTA] per row
        .concatenate(
            ["x", "tta_per_trial"], 
            into="x_augmented"
        )
        
        # Step 3: Treat augmented trials as exchangeable set
        .as_set("x_augmented")  # Now operating on (N, 3) data
        
        # Step 4: Standardize with mean/std
        # IMPORTANT: BayesFlow standardization applies across all features
        # This will normalize RT, CPP, and TTA together
        .standardize("x_augmented", mean=0.0, std=1.0)
        
        # Step 5: Feature engineering
        .sqrt("number_of_trials")
        
        # Step 6: Type conversion
        .convert_dtype("float64", "float32")
        
        # Step 7: Concatenate parameter values (inference variables)
        .concatenate(PAR_NAMES, into="inference_variables")
        
        # Step 8: Rename for BayesFlow
        # summary_variables now contains [RT, CPP, TTA] augmented data
        .rename("x_augmented", "summary_variables")
        
        # NOTE: No condition_variables anymore!
        # All context (TTA) is embedded in summary_variables
    )
    return adapter


def make_adaptor():
    return adopt_TrialWise(prior_DC_simplest_model())
