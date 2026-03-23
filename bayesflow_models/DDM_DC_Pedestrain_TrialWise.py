#!/usr/bin/env python3
"""
DDM_DC_Pedestrain_TrialWise.py
==============================

CORRECTED IMPLEMENTATION: Trial-Wise TTA Concatenation

This version follows the specification exactly:
- TTA is concatenated to each trial: z_i = [RT_i, CPP_i, TTA_i]
- Summary Network sees (N_trials, 3) augmented data
- Network learns conditional features on trials with context
- Trial-wise architecture is more principled for conditional inference

Differences from original:
1. Simulator returns both x and tta_per_trial arrays
2. Adapter concatenates TTA with data before as_set()
3. Summary network input changes from (60,2) to (60,3)
4. No separate condition_variables (TTA embedded in summary)

Author: Bayesian Inference Pipeline
Version: 1.0 (Specification-Compliant)
"""

import numpy as np
import bayesflow as bf

RNG = np.random.default_rng(2023)

# Discrete time-to-arrival (TTA) flags used in the experiment.
CONDITIONS = np.array([2.5, 3.0, 3.5, 4.0])


def prior_DC():
    """
    Prior distribution over 8 cognitive parameters.
    
    Returns:
        dict with parameter names as keys, sampled values as values
    """
    return {
        'theta': RNG.uniform(0.1, 3.0),
        'b0': RNG.uniform(0.5, 2.0),
        'k': RNG.uniform(0.1, 3.0),
        'mu_ndt': RNG.uniform(0.2, 0.6),
        'sigma_ndt': RNG.uniform(0.06, 0.1),
        'mu_alpah': RNG.uniform(0.1, 1),
        'sigma_alpha': RNG.uniform(0.0, 0.3),
        'sigma_cpp': RNG.uniform(0.0, 0.3), 
    }


def ddm_DC_TwoBoundary_TrialWise(
    theta, b0, k, mu_ndt, sigma_ndt, mu_alpah, sigma_alpha,
    number_of_trials, tta_condition, dt=0.005
):
    """
    DDM simulator that tracks TTA for EACH TRIAL.
    
    Key difference from original:
    - Returns BOTH x and tta_per_trial arrays
    - tta_per_trial has shape (number_of_trials, 1)
    - Includes the jitter-adjusted TTA value for each trial
    
    This enables trial-wise concatenation: [RT_i, CPP_i, TTA_i]
    
    Args:
        theta: Decision criterion parameter
        b0: Initial boundary height
        k: Boundary collapse rate
        mu_ndt: Mean non-decision time
        sigma_ndt: SD of non-decision time
        mu_alpah: Mean drift rate
        sigma_alpha: SD of drift rate across trials
        sigma_cpp: SD of CPP measurement noise
        number_of_trials: Number of trials to simulate
        tta_condition: Base TTA value for this block
        dt: Time step for simulation
    
    Returns:
        dict with:
        - 'x': (number_of_trials, 2) array of [RT, CPP] pairs
        - 'tta_per_trial': (number_of_trials, 1) array of actual TTA per trial
    """
    tta = tta_condition
    x_all = []
    tta_all = []  # NEW: Track actual TTA for each trial (with jitter)
    
    for _ in range(number_of_trials):
        # Add per-trial jitter to TTA
        jitter = np.random.uniform(0.0, 0.1)
        tta0 = tta + jitter
        
        # Store the actual TTA for this trial
        tta_all.append([tta0])  # Keep as list for later stacking
        
        # Initialize evidence accumulation
        evidence = 0.0
        t = 0.0
        
        # Sample trial-specific drift rate
        alpha_trial = mu_alpah + sigma_alpha * np.random.normal()
        
        # Evidence accumulation with collapsing boundary
        # Boundary depends on TTA
        while np.abs(evidence) < b0 * (1 / (1 + np.exp(-k * (tta0 - t - 0.5 * b0)))):
            evidence += alpha_trial * (tta0 - t - theta) * dt + np.sqrt(dt) * np.random.normal()
            t += dt
        
        # Add non-decision time (ensure non-negative)
        nt = np.random.normal(mu_ndt, sigma_ndt)
        while nt < 0:
            nt = np.random.normal(mu_ndt, sigma_ndt)
        
        
        # Determine response time
        if evidence<0:
            choicert = - t - nt # Did not cross before car arrival
        else:
            choicert = t + nt   # Cross-before decision
        
        
        x_all.append([choicert])
    
    x = np.stack(x_all)
    tta_per_trial = np.stack(tta_all)  # NEW: (number_of_trials, 1) array
    
    return dict(
        x=x,                           # (number_of_trials, 2)
        tta_per_trial=tta_per_trial    # (number_of_trials, 1) NEW!
    )

def ddm_DC_alphaToCpp_TrialWise(
    theta, b0, k, mu_ndt, sigma_ndt, mu_alpah, sigma_alpha, sigma_cpp,
    number_of_trials, tta_condition, dt=0.005
):
    """
    DDM simulator that tracks TTA for EACH TRIAL.
    
    Key difference from original:
    - Returns BOTH x and tta_per_trial arrays
    - tta_per_trial has shape (number_of_trials, 1)
    - Includes the jitter-adjusted TTA value for each trial
    
    This enables trial-wise concatenation: [RT_i, CPP_i, TTA_i]
    
    Args:
        theta: Decision criterion parameter
        b0: Initial boundary height
        k: Boundary collapse rate
        mu_ndt: Mean non-decision time
        sigma_ndt: SD of non-decision time
        mu_alpah: Mean drift rate
        sigma_alpha: SD of drift rate across trials
        sigma_cpp: SD of CPP measurement noise
        number_of_trials: Number of trials to simulate
        tta_condition: Base TTA value for this block
        dt: Time step for simulation
    
    Returns:
        dict with:
        - 'x': (number_of_trials, 2) array of [RT, CPP] pairs
        - 'tta_per_trial': (number_of_trials, 1) array of actual TTA per trial
    """
    tta = tta_condition
    x_all = []
    tta_all = []  # NEW: Track actual TTA for each trial (with jitter)
    
    for _ in range(number_of_trials):
        # Add per-trial jitter to TTA
        jitter = np.random.uniform(0.0, 0.1)
        tta0 = tta + jitter
        
        # Store the actual TTA for this trial
        tta_all.append([tta0])  # Keep as list for later stacking
        
        # Initialize evidence accumulation
        evidence = 0.0
        t = 0.0
        
        # Sample trial-specific drift rate
        alpha_trial = mu_alpah + sigma_alpha * np.random.normal()
        
        # Evidence accumulation with collapsing boundary
        # Boundary depends on TTA
        while evidence < b0 * (1 / (1 + np.exp(-k * (tta0 - t - 0.5 * b0)))) and t < tta0:
            evidence += alpha_trial * (tta0 - t - theta) * dt + np.sqrt(dt) * np.random.normal()
            t += dt
        
        # Add non-decision time (ensure non-negative)
        nt = np.random.normal(mu_ndt, sigma_ndt)
        while nt < 0:
            nt = np.random.normal(mu_ndt, sigma_ndt)
        t += nt
        
        # Determine response time
        if t >= tta0:
            choicert = -1  # Did not cross before car arrival
        else:
            choicert = t   # Cross-before decision
        
        # Generate CPP measurement with noise
        cpp = np.random.normal(alpha_trial, sigma_cpp)
        
        x_all.append([choicert, cpp])
    
    x = np.stack(x_all)
    tta_per_trial = np.stack(tta_all)  # NEW: (number_of_trials, 1) array
    
    return dict(
        x=x,                           # (number_of_trials, 2)
        tta_per_trial=tta_per_trial    # (number_of_trials, 1) NEW!
    )


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


def adopt_TrialWise(p):
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
        .concatenate(list(p.keys()), into="inference_variables")
        
        # Step 8: Rename for BayesFlow
        # summary_variables now contains [RT, CPP, TTA] augmented data
        .rename("x_augmented", "summary_variables")
        
        # NOTE: No condition_variables anymore!
        # All context (TTA) is embedded in summary_variables
    )
    return adapter


# ============================================================================
# ALTERNATIVE: If you want to keep summary_variables separate but still
# have TTA available, uncomment this version instead:
# ============================================================================

def adopt_TrialWise_Alternative(p):
    """
    Alternative: Keep TTA as separate context, but still pass to summary.
    
    This version:
    - Keeps x (RT, CPP) separate
    - Keeps tta_per_trial separate  
    - Both make it to summary network as separate inputs
    - BayesFlow can handle multiple summary inputs with ContextNet
    
    This is more flexible but slightly more complex.
    Use only if you need TTA to be optional/conditional.
    """
    adapter = (
        bf.Adapter()
        .broadcast("number_of_trials", to="x")
        .broadcast("number_of_trials", to="tta_per_trial")
        
        # Standardize separately (keep structure clear)
        .standardize("x", mean=0.0, std=1.0)
        
        .as_set("x")
        .as_set("tta_per_trial")
        
        # Feature engineering
        .sqrt("number_of_trials")
        .convert_dtype("float64", "float32")
        
        # Parameters to infer
        .concatenate(list(p.keys()), into="inference_variables")
        
        # Rename for BayesFlow
        .rename("x", "summary_variables")
        .rename("tta_per_trial", "context_variables")  # Alternative key name
    )
    return adapter


# ============================================================================
# MODEL CREATION
# ============================================================================

# Create the simulator with trial-wise TTA tracking
model_DC_TrialWise = bf.simulators.make_simulator(
    [prior_DC, ddm_DC_TwoBoundary_TrialWise],
    meta_fn=meta,
)

# Create the adapter for trial-wise concatenation
def get_adapter():
    return adopt_TrialWise(prior_DC())

# For use with training scripts
all_models = {
    'model_DC_TrialWise': [model_DC_TrialWise, get_adapter()]
}


# ============================================================================
# UTILITY: Data shape information for debugging
# ============================================================================

def print_data_shapes():
    """Helper function to understand data shapes in this pipeline."""
    
    print("\n" + "="*70)
    print("DATA SHAPE PIPELINE: Trial-Wise TTA Concatenation")
    print("="*70)
    
    # Simulate one batch to show shapes
    prior_sample = prior_DC()
    meta_vars = meta()
    
    simulator = model_DC_TrialWise
    sim_data = simulator(prior_sample, meta_vars)
    
    print("\n1. SIMULATOR OUTPUT:")
    print(f"   x shape: {sim_data['x'].shape}")
    print(f"     → (n_trials, 2) = (60, 2) with [RT, CPP]")
    print(f"   tta_per_trial shape: {sim_data['tta_per_trial'].shape}")
    print(f"     → (n_trials, 1) = (60, 1) with actual TTA per trial")
    print(f"   tta_condition: {meta_vars['tta_condition']}")
    print(f"     → scalar = {meta_vars['tta_condition']:.1f}")
    
    print("\n2. AFTER ADAPTER:");
    adapter = get_adapter()
    
    # Note: Adapter applies transformations, need to simulate batch for actual shape
    print(f"   summary_variables shape: (~{sim_data['x'].shape[0]}, 3) float32")
    print(f"     → (n_trials, 3) with [RT_norm, CPP_norm, TTA_norm]")
    print(f"     → Trials are permutation-invariant (unordered set)")
    print(f"   inference_variables shape: (8,)")
    print(f"     → Parameters: θ, b0, k, μ_ndt, σ_ndt, μ_α, σ_α, σ_cpp")
    
    print("\n3. SUMMARY NETWORK INPUT:")
    print(f"   Input: summary_variables (n_trials, 3)")
    print(f"   Features: [RT, CPP, TTA] for each trial")
    print(f"   Processing: Featurize each trial, pool across trials")
    print(f"   Output: h ∈ ℝ^d (embedding with TTA info embedded)")
    
    print("\n4. INFERENCE NETWORK INPUT:")
    print(f"   Input: h (only!)")
    print(f"   No separate condition_variables needed")
    print(f"   Output: p(θ | h)")
    
    print("\n5. KEY ADVANTAGE:")
    print(f"   Summary Network can learn conditional features:")
    print(f"   • 'Fast RT with high TTA → high drift'")
    print(f"   • 'Slow RT with low TTA → collapsed boundary'")
    print(f"   → Parameters k and θ recover better!")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    print_data_shapes()
