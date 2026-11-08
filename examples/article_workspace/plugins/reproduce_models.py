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

import logging

# Tell the bayesflow logger to ignore warnings and info messages
logging.getLogger("bayesflow").setLevel(logging.ERROR)

import numpy as np
import bayesflow as bf

RNG = np.random.default_rng(2023)

# Discrete time-to-arrival (TTA) flags used in the experiment.
CONDITIONS = np.array([2.5, 3.0, 3.5, 4.0])

PAR_NAMES_DC_REPRODUCED = ['tau','b0','k','mu_ndt','sigma_ndt','alpha','theta']
PAR_NAMES_DF_REPRODUCED = ['b0','mu_ndt','sigma_ndt','alpha','theta']
PAR_NAMES_SF_REPRODUCED = ['b0','mu_ndt','sigma_ndt','alpha','theta']

## model names:
#1) DDM base models:
    # dynamic drift rate collapsing bound model (DC)
    # dynamic drift rate fixed bound model (DF) 
    # static drift rate fixed bound (SF) 
#2) Utility base model(U)

## parameters of DDM base models:
    # alpah: rate of attention to kenimatical parameter like tta
    # theta: time of big change of accumulation from cross to not cross and 
        #  is fraction of tta because it is dependent on time distance of car arrived
        # from 0(because probably some one does not any risky point) to 1(risky point of every body does not less of start)
    # b0: Initial boundary height
    # k: Boundary collapse rate(shape of collapse)
    # tau: time of boundary really starts to collapse and 
        #  is fraction of tta from 0(because at before of start does not get mean) to 1(after tta does not mean)
    # mu_ndt: Mean non-decision time
    # sigma_ndt: SD of non-decision time
    # number_of_trials: Number of trials to simulate
    # tta_condition: Base TTA value for this block
    # dt: Time step for simulation  
     
def prior_DC() -> dict:
    """
    Prior distribution over 7 cognitive parameters.
    
    Returns:
        dict with parameter names as keys, sampled values as values
    """
    return {
        'tau':RNG.uniform(0.0,1.0),
        'b0': RNG.uniform(0.1, 2.0),
        'k': RNG.uniform(1.0, 5.0),
        'mu_ndt': RNG.uniform(0.2, 0.6),
        'sigma_ndt': RNG.uniform(0.06, 0.1),
        'alpha': RNG.uniform(0.0, 1.0),
        'theta': RNG.uniform(0.0, 1.0),
    }

def prior_DF() -> dict:
    """
    Prior distribution over 5 cognitive parameters.
    
    Returns:
        dict with parameter names as keys, sampled values as values
    """
    return {
        'b0': RNG.uniform(0.1, 2.0),
        'mu_ndt': RNG.uniform(0.2, 0.6),
        'sigma_ndt': RNG.uniform(0.06, 0.1),
        'alpha': RNG.uniform(0.0, 1.0),
        'theta': RNG.uniform(0.0, 1.0),
    }

def prior_SF() -> dict:
    """
    Prior distribution over 5 cognitive parameters.
    
    Returns:
        dict with parameter names as keys, sampled values as values
    """
    return {
        'b0': RNG.uniform(0.1, 2.0),
        'mu_ndt': RNG.uniform(0.2, 0.6),
        'sigma_ndt': RNG.uniform(0.06, 0.1),
        'alpha': RNG.uniform(0.0, 1.0),
        'theta': RNG.uniform(0.0, 1.0),
    }


def simulator_DC(
    tta_condition: int,
    number_of_trials: int,
    b0: float,
    tau: float,
    k: float,
    mu_ndt: float, 
    sigma_ndt: float,
    alpha: float,
    theta: float, 
    dt: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one participant/session with different TTA values per trial."""
    tta = tta_condition
    x_all = []
    tta_all = []

    for _ in range(number_of_trials):
        jitter = RNG.uniform(0.0, 0.1)
        tta0 = float(tta + jitter)
        tta_all.append([tta0])
        evidence = 0.0
        t = 0.0
        boundary = b0 * (1 / (1 + np.exp(-k * (tta0 - t - tau))))
        while np.abs(evidence) < boundary and t<tta0:
            evidence += alpha * (tta0 - t - theta) * dt + np.sqrt(dt) * RNG.normal()
            t += dt

        nt = RNG.normal(mu_ndt, sigma_ndt)
        while nt < 0:
            nt = RNG.normal(mu_ndt, sigma_ndt)

        if evidence < 0 or t>tta0:
            choicert = -1.0
        else:
            choicert = t + nt

        x_all.append([choicert])


    x = np.stack(x_all)
    tta_per_trial = np.stack(tta_all)  # NEW: (number_of_trials, 1) array
    # return np.asarray(x_all, dtype=np.float32), np.asarray(tta_all, dtype=np.float32)
    return dict(
        x=x,                           # (number_of_trials, 1)
        tta_per_trial=tta_per_trial    # (number_of_trials, 1) NEW!
    )

def simulator_DF(
    tta_condition: int,
    number_of_trials: int,
    b0: float,
    mu_ndt: float, 
    sigma_ndt: float,
    alpha: float,
    theta: float, 
    dt: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one participant/session with different TTA values per trial."""
    tta = tta_condition
    x_all = []
    tta_all = []

    for __ in range(number_of_trials):
        jitter = RNG.uniform(0.0, 0.1)
        tta0 = float(tta + jitter)
        tta_all.append([tta0])
        evidence = 0.0
        t = 0.0

        while np.abs(evidence) < b0 and t<tta0:
            evidence += alpha * (tta0 - t - theta) * dt + np.sqrt(dt) * RNG.normal()
            t += dt

        nt = RNG.normal(mu_ndt, sigma_ndt)
        while nt < 0:
            nt = RNG.normal(mu_ndt, sigma_ndt)

        if evidence < 0 or t>tta0:
            choicert = -1.0
        else:
            choicert = t + nt

        x_all.append([choicert])
    
    x = np.stack(x_all)
    tta_per_trial = np.stack(tta_all)  # NEW: (number_of_trials, 1) array
    # return np.asarray(x_all, dtype=np.float32), np.asarray(tta_all, dtype=np.float32)
    return dict(
        x=x,                           # (number_of_trials, 1)
        tta_per_trial=tta_per_trial    # (number_of_trials, 1) NEW!
    )

def simulator_SF(
    number_of_trials: int,
    tta_condition: int,
    b0: float,
    mu_ndt: float,
    sigma_ndt: float,
    alpha: float,
    theta: float,
    dt: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one participant/session with different TTA values per trial."""
    tta = tta_condition
    x_all = []
    tta_all = []

    for __ in range(number_of_trials):
        jitter = RNG.uniform(0.0, 0.1)
        tta0 = float(tta + jitter)
        tta_all.append([tta0])
        evidence = 0.0
        t = 0.0

        while np.abs(evidence) < b0 and t<tta0:
            evidence += alpha * (tta0 - theta) * dt + np.sqrt(dt) * RNG.normal()
            t += dt

        nt = RNG.normal(mu_ndt, sigma_ndt)
        while nt < 0:
            nt = RNG.normal(mu_ndt, sigma_ndt)

        if evidence < 0 or t>tta0:
            choicert = -1.0
        else:
            choicert = t + nt

        x_all.append([choicert])
    
    x = np.stack(x_all)
    tta_per_trial = np.stack(tta_all)  # NEW: (number_of_trials, 1) array
    # return np.asarray(x_all, dtype=np.float32), np.asarray(tta_all, dtype=np.float32)
    return dict(
        x=x,                           # (number_of_trials, 1)
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







## Adapt Trial Wise tta
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
        # .broadcast("number_of_trials", to="tta_per_trial")
        
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
        # .standardize("x_augmented", mean=0.0, std=1.0)
        
        # Step 5: Feature engineering
        # .sqrt("number_of_trials")
        
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
# MODEL CREATION
# ============================================================================


model_DF_reproduced = bf.simulators.make_simulator(
    [prior_DF,simulator_DF],
    meta_fn=meta,
)


model_SF_reproduced = bf.simulators.make_simulator(
    [prior_SF,simulator_SF],
    meta_fn=meta,
)


model_DC_reproduced = bf.simulators.make_simulator(
    [prior_DC,simulator_DC],
    meta_fn=meta,
)


def get_adapter_trialwise_dc():
    return adopt_TrialWise(prior_DC())

def get_adapter_trialwise_sf():
    return adopt_TrialWise(prior_SF())

def get_adapter_trialwise_df():
    return adopt_TrialWise(prior_DF())



# ============================================================================
# UTILITY: Data shape information for debugging
# ============================================================================

# def print_data_shapes():
#     """Helper function to understand data shapes in this pipeline."""
    
#     print("\n" + "="*70)
#     print("DATA SHAPE PIPELINE: Trial-Wise TTA Concatenation")
#     print("="*70)
    
#     # Simulate one batch to show shapes
#     prior_sample = prior_DC()
#     meta_vars = meta()
    
#     simulator = model_DC_TrialWise
#     sim_data = simulator(prior_sample, meta_vars)
    
#     print("\n1. SIMULATOR OUTPUT:")
#     print(f"   x shape: {sim_data['x'].shape}")
#     print(f"     → (n_trials, 2) = (60, 2) with [RT, CPP]")
#     print(f"   tta_per_trial shape: {sim_data['tta_per_trial'].shape}")
#     print(f"     → (n_trials, 1) = (60, 1) with actual TTA per trial")
#     print(f"   tta_condition: {meta_vars['tta_condition']}")
#     print(f"     → scalar = {meta_vars['tta_condition']:.1f}")
    
#     print("\n2. AFTER ADAPTER:");
#     adapter = get_adapter()
    
#     # Note: Adapter applies transformations, need to simulate batch for actual shape
#     print(f"   summary_variables shape: (~{sim_data['x'].shape[0]}, 3) float32")
#     print(f"     → (n_trials, 3) with [RT_norm, CPP_norm, TTA_norm]")
#     print(f"     → Trials are permutation-invariant (unordered set)")
#     print(f"   inference_variables shape: (8,)")
#     print(f"     → Parameters: θ, b0, k, μ_ndt, σ_ndt, μ_α, σ_α, σ_cpp")
    
#     print("\n3. SUMMARY NETWORK INPUT:")
#     print(f"   Input: summary_variables (n_trials, 3)")
#     print(f"   Features: [RT, CPP, TTA] for each trial")
#     print(f"   Processing: Featurize each trial, pool across trials")
#     print(f"   Output: h ∈ ℝ^d (embedding with TTA info embedded)")
    
#     print("\n4. INFERENCE NETWORK INPUT:")
#     print(f"   Input: h (only!)")
#     print(f"   No separate condition_variables needed")
#     print(f"   Output: p(θ | h)")
    
#     print("\n5. KEY ADVANTAGE:")
#     print(f"   Summary Network can learn conditional features:")
#     print(f"   • 'Fast RT with high TTA → high drift'")
#     print(f"   • 'Slow RT with low TTA → collapsed boundary'")
#     print(f"   → Parameters k and θ recover better!")
    
#     print("="*70 + "\n")


# if __name__ == "__main__":
#     print_data_shapes()