import numpy as np
import bayesflow as bf

RNG = np.random.default_rng(2023)
# Discrete time-to-arrival (TTA) flags used to condition the simulators/adapters.
# These match the actual experiment: 4 TTA conditions as described in the task
CONDITIONS = np.array([2.5, 3.0, 3.5, 4.0])


def prior_DC():
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

# def ddm_DC_alphaToCpp( theta, b0, k, mu_ndt, sigma_ndt,mu_alpah, sigma_alpha,sigma_cpp,number_of_trials, tta_condition, dt=0.005):
#     tta = tta_condition  
#     x_all = []
#     x = []
#     for _ in range(number_of_trials):
#         jitter = np.random.uniform(0.0,0.1)
#         tta0 = tta + jitter
#         n_steps = 0.0
#         evidence = 0.0
#         t = 0.0
#         alpha_trial = mu_alpah + sigma_alpha * np.random.normal()
#         while evidence < b0 * (1 / (1 + np.exp(-k * (tta0 - t - 0.5 * b0)))) and t < tta0:
#             evidence += alpha_trial * (tta0 - t - theta)* dt + np.sqrt(dt) * np.random.normal()
#             n_steps += 1.0
#             t += dt

#         nt =np.random.normal(mu_ndt,sigma_ndt)
#         while nt < 0:
#             nt =np.random.normal(mu_ndt,sigma_ndt)
#         t += nt

#         if t >= tta0:
#             choicert = -1
#         else:
#             choicert =  t

#         cpp = np.random.normal(alpha_trial,sigma_cpp)
#         x = [choicert,cpp]
#         x_all.append(x)
#     x = np.stack(x_all)
#     return dict(x=x)

def ddm_DC_alphaToCpp(theta, b0, k, mu_ndt, sigma_ndt, mu_alpah, sigma_alpha, sigma_cpp, number_of_trials, tta_condition, dt=0.005):
    """
    DDM simulator for pedestrian crossing with collapsing boundaries.
    
    Returns data for a SINGLE TTA condition (as specified by tta_condition parameter).
    This is the correct format for BayesFlow conditional inference.
    
    Args:
        theta: Decision criterion parameter
        b0: Initial boundary height
        k: Boundary collapse rate
        mu_ndt: Mean non-decision time
        sigma_ndt: SD of non-decision time
        mu_alpah: Mean drift rate (note: typo preserved for compatibility)
        sigma_alpha: SD of drift rate across trials
        sigma_cpp: SD of CPP measurement noise
        number_of_trials: Number of trials to simulate
        tta_condition: Single TTA value for this simulation (e.g., 2.5, 3.0, 3.5, or 4.0)
        dt: Time step for simulation
    
    Returns:
        dict with key 'x': array of shape (number_of_trials, 2) containing [RT, CPP] pairs
    """
    tta = tta_condition
    x_all = []
    
    for _ in range(number_of_trials):
        # Add per-trial jitter to TTA
        jitter = np.random.uniform(0.0, 0.1)
        tta0 = tta + jitter
        
        # Initialize evidence accumulation
        evidence = 0.0
        t = 0.0
        
        # Sample trial-specific drift rate
        alpha_trial = mu_alpah + sigma_alpha * np.random.normal()
        
        # Evidence accumulation with collapsing boundary
        while evidence < b0 * (1 / (1 + np.exp(-k * (tta0 - t - 0.5 * b0)))) and t < tta0:
            evidence += alpha_trial * (tta0 - t - theta) * dt + np.sqrt(dt) * np.random.normal()
            t += dt
        
        # Add non-decision time (ensure non-negative)
        nt = np.random.normal(mu_ndt, sigma_ndt)
        while nt < 0:
            nt = np.random.normal(mu_ndt, sigma_ndt)
        t += nt
        
        # Determine response time (cross-before decisions only; cross-after coded as -1)
        if t >= tta0:
            choicert = -1  # Did not cross before car arrival
        else:
            choicert = t   # Cross-before decision
        
        # Generate CPP measurement with noise
        cpp = np.random.normal(alpha_trial, sigma_cpp)
        
        x_all.append([choicert, cpp])
    
    x = np.stack(x_all)
    return dict(x=x)

def meta():
    """
    Meta-function that generates simulation context.
    
    For conditional inference in BayesFlow:
    - Randomly selects ONE TTA condition per simulation
    - This TTA will be passed to the simulator and becomes the conditioning variable
    - During training, the network learns p(θ | data, TTA)
    """
    tta_flag = RNG.choice(CONDITIONS)
    return {
        "number_of_trials": 60,  # 64 trials per condition in real experiment, using 60 here
        "tta_condition": tta_flag,
    }

def adopt(p):
    """
    Adapter that transforms simulator output into format expected by BayesFlow networks.
    
    Key operations for conditional inference:
    1. Treats trials as exchangeable sets (permits variable-length data)
    2. Standardizes the data (important for neural network training)
    3. Concatenates parameters into 'inference_variables' (what we want to infer)
    4. Renames 'x' to 'summary_variables' (observed data)
    5. Renames 'tta_condition' to 'condition_variables' (conditioning context)
    
    This structure enables the network to learn p(parameters | data, TTA_condition)
    """
    adapter = (
        bf.Adapter()
        .broadcast("number_of_trials", to="x")  # Align trial count with data
        .as_set("x")  # Treat trials as exchangeable (permits variable length)
        .standardize("x", mean=0.0, std=1.0)  # Normalize data for neural network (BayesFlow 2.0.8+ requires mean/std)
        .sqrt("number_of_trials")  # Feature engineering
        .convert_dtype("float64", "float32")  # PyTorch compatibility
        .concatenate(list(p.keys()), into="inference_variables")  # Parameters to infer
        .rename("x", "summary_variables")  # Observed data
        .rename("tta_condition", "condition_variables")  # Conditioning variable
    )
    return adapter



model_DC = bf.simulators.make_simulator([prior_DC, ddm_DC_alphaToCpp], meta_fn=meta)


all_models = {'model_DC': [model_DC, adopt(prior_DC())]}