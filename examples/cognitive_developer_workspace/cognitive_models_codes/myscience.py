import logging

# Tell the bayesflow logger to ignore warnings and info messages
logging.getLogger("bayesflow").setLevel(logging.ERROR)

import numpy as np
import bayesflow as bf


RNG = np.random.default_rng(2023)

# Discrete time-to-arrival (TTA) flags used in the experiment.
CONDITIONS = np.array([2.5, 3.0, 3.5, 4.0])

PAR_NAMES = ['theta','b0','mu_ndt','mu_alpha']

TTA_LEVELS = (2.5, 3.0, 3.5, 4.0)
TRAINING_TRIALS = 60
DEFAULT_DT = 0.005


def prior_DC_simplest_model():
    """
    Prior distribution over 8 cognitive parameters.
    
    Returns:
        dict with parameter names as keys, sampled values as values
    """
    return {
        'theta': RNG.uniform(0.1, 3.0),
        'b0': RNG.uniform(2, 5),
        'mu_ndt': RNG.uniform(0.2, 0.6),
        'mu_alpha': RNG.uniform(0.1, 1),
    }


# The simplest model
def ddm_DC_TwoBoundary_TrialWise_simplest(
    theta, b0,mu_ndt, mu_alpha,
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
    sigma_ndt = 1
    sigma_alpha = 1
    k = 1
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
        alpha_trial = mu_alpha + sigma_alpha * np.random.normal()
        
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
            choicert = - 1 # Did not cross before car arrival
        else:
            choicert = t + nt   # Cross-before decision
        
        
        x_all.append([choicert])
    
    x = np.stack(x_all)
    tta_per_trial = np.stack(tta_all)  # NEW: (number_of_trials, 1) array
    
    return dict(
        x=x,                           # (number_of_trials, 1)
        tta_per_trial=tta_per_trial    # (number_of_trials, 1) NEW!
    )