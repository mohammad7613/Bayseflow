# Conditional Inference with BayesFlow: Complete Guide

## Overview

This guide explains how to use BayesFlow for **conditional inference** in your pedestrian crossing experiment where trials have different Time-To-Arrival (TTA) conditions.

## The Challenge

In your experiment:
- Each subject completes 256 trials
- Trials are distributed across 4 TTA conditions (2.5s, 3.0s, 3.5s, 4.0s)
- Each TTA condition has 64 trials
- You want to estimate **subject-level parameters** that explain behavior across all conditions

**The key insight**: The TTA condition affects observed behavior, so the model needs to learn:

$$p(\theta | \text{data}, \text{TTA condition})$$

This is called **conditional inference** or **conditional density estimation**.

---

## How BayesFlow Handles Conditions

### 1. During Training

**What happens in each training iteration:**

1. **Sample parameters** from prior: `θ ~ prior_DC()`
2. **Sample a TTA condition**: `TTA ~ Uniform({2.5, 3.0, 3.5, 4.0})`
3. **Simulate data** given parameters and TTA: `data ~ simulator(θ, TTA)`
4. **Train network** to learn: `network(data, TTA) → θ`

The network learns that the SAME parameters can produce DIFFERENT data depending on the TTA condition.

### 2. The Data Flow

```python
# Sample from simulator
sample = model.sample(1)

# What you get:
{
    'x': array([[...]]),              # Observed data: [RT, CPP] pairs
    'theta': 0.523,                   # Parameter values
    'b0': 1.234,
    'k': 1.456,
    # ... other parameters
    'tta_condition': 3.5,             # The TTA for this simulation
    'number_of_trials': 60
}

# Adapter transforms it to:
{
    'summary_variables': [...],       # Standardized observed data
    'inference_variables': [...],     # Parameters to infer (theta, b0, k, ...)
    'condition_variables': [3.5],     # TTA condition (this is KEY!)
}
```

### 3. During Inference on Real Data

For a real subject with 256 trials across 4 TTA conditions:

```python
# You provide data in the same format:
{
    'x': array([64, 2]),              # 64 trials for TTA=2.5
    'tta_condition': 2.5,
    'number_of_trials': 64
}

# The network uses the TTA=2.5 information when inferring parameters
posterior_tta25 = approximator.sample(conditions=data_tta25, num_samples=1000)

# Repeat for each TTA condition
# Then combine posteriors across all conditions to get subject-level estimates
```

---

## Code Structure Explained

### 1. The Simulator (`ddm_DC_alphaToCpp`)

**Purpose**: Generate synthetic data given parameters and ONE TTA condition

```python
def ddm_DC_alphaToCpp(theta, b0, k, mu_ndt, sigma_ndt, mu_alpah, 
                      sigma_alpha, sigma_cpp, number_of_trials, 
                      tta_condition, dt=0.005):
    """
    CRITICAL: This function receives a SINGLE TTA value, not multiple!
    
    Args:
        theta, b0, k, ...: Model parameters
        number_of_trials: How many trials to simulate
        tta_condition: SINGLE TTA value (e.g., 2.5 or 3.0 or 3.5 or 4.0)
    
    Returns:
        dict with key 'x': array of shape (number_of_trials, 2)
                          containing [RT, CPP] pairs
    """
    # Simulate number_of_trials with the given TTA
    # Return format: {'x': array([RT, CPP] pairs)}
```

**Why only one TTA per call?**
- During training, each simulation represents data from ONE experimental condition
- The network learns the mapping: (parameters, TTA) → data distribution
- This is how the network learns that TTA affects behavior

### 2. The Meta Function

**Purpose**: Specify the simulation context

```python
def meta():
    """
    Called before each simulation during training.
    
    Returns:
        Dictionary specifying:
        - number_of_trials: How many trials per simulation
        - tta_condition: Which TTA to use (randomly selected)
    """
    tta_flag = RNG.choice(CONDITIONS)  # Randomly pick ONE TTA
    return {
        "number_of_trials": 60,
        "tta_condition": tta_flag,
    }
```

**Why random selection?**
- Training data should cover all TTA conditions
- Random selection ensures balanced representation
- Each simulation explores a different (parameters, TTA) combination

### 3. The Adapter

**Purpose**: Transform simulator output into neural network input format

```python
def adopt(p):
    adapter = (
        bf.Adapter()
        .broadcast("number_of_trials", to="x")
        .as_set("x")                    # Treat trials as exchangeable
        .standardize("x")               # Normalize data
        .sqrt("number_of_trials")       # Feature engineering
        .convert_dtype("float64", "float32")
        .concatenate(list(p.keys()), into="inference_variables")  # θ
        .rename("x", "summary_variables")                         # data
        .rename("tta_condition", "condition_variables")           # TTA ← KEY!
    )
    return adapter
```

**The critical line**:
```python
.rename("tta_condition", "condition_variables")
```

This tells BayesFlow: "tta_condition is a CONDITIONING variable, not something to infer!"

---

## Training Workflow

### Option 1: Train from Scratch

```python
from DDM_DC_Pedestrain import all_models
from train import train_amortizer

model_name = 'model_DC'
model = all_models[model_name]

# Train for 50 epochs with 10,000 simulations per epoch
history = train_amortizer(
    model=model,
    model_name=model_name,
    n_sim=10000,
    epochs=50
)
```

### Option 2: Resume Training

```python
from train import train_amortizer_resume

# Continue training from a checkpoint
history = train_amortizer_resume(
    model=model,
    model_name='model_DC',
    n_sim=10000,
    epochs=20,  # 20 more epochs
    checkpoint_dir="trained_model1/checkpoints"
)
```

**Training recommendations**:
- Start with 50 epochs minimum
- Use 10,000 simulations per epoch
- Monitor validation loss
- Check parameter recovery before applying to real data

---

## Evaluation Workflow

### 1. Overall Parameter Recovery

```python
import keras

# Load trained model
approximator = keras.saving.load_model("trained_model1/checkpoints/model_DC.keras")

# Generate validation data (mixture of all TTA conditions)
val_sims = model.sample(200)

# Get posterior samples
post_draws = approximator.sample(conditions=val_sims, num_samples=1000)

# Check recovery
bf.diagnostics.plots.recovery(
    estimates=post_draws,
    targets=val_sims,
    variable_names=['theta', 'b0', 'k', 'mu_ndt', 'sigma_ndt', 
                    'mu_alpah', 'sigma_alpha', 'sigma_cpp']
)
```

### 2. Condition-Specific Recovery

```python
# Filter by condition
def filter_by_condition(data_dict, tta_value):
    mask = np.abs(data_dict['tta_condition'] - tta_value) < 0.01
    return {key: val[mask] if len(val) == len(mask) else val 
            for key, val in data_dict.items()}

# Check each TTA separately
for tta in [2.5, 3.0, 3.5, 4.0]:
    cond_data = filter_by_condition(val_sims, tta)
    cond_posteriors = approximator.sample(conditions=cond_data, num_samples=1000)
    
    # Plot recovery for this condition
    bf.diagnostics.plots.recovery(
        estimates=cond_posteriors,
        targets=cond_data,
        variable_names=[...]
    )
```

---

## Applying to Real Data

### Expected Data Format

Your experimental data should have this structure:

```csv
subject_id,trial_num,block,TTA,direction,RT,collision,CPP
S001,1,1,2.5,left,1.245,0,0.034
S001,2,1,3.0,right,1.678,0,0.042
S001,3,1,3.5,left,-1,0,0.028  # cross-after (RT=-1)
...
```

### Step 1: Prepare Data

```python
import pandas as pd

# Load your data
df = pd.read_csv("pedestrian_experiment_data.csv")

# Select one subject
subject_id = 'S001'
subject_data = df[df['subject_id'] == subject_id].copy()

# Group by TTA condition
data_by_condition = {}
for tta in [2.5, 3.0, 3.5, 4.0]:
    condition_trials = subject_data[subject_data['TTA'] == tta]
    
    # Extract RT and CPP
    rts = condition_trials['RT'].values
    cpps = condition_trials['CPP'].values  # Use zeros if CPP not available
    
    # Stack into [n_trials, 2] format
    data_by_condition[tta] = np.column_stack([rts, cpps])
```

### Step 2: Run Inference

```python
def infer_subject_parameters(approximator, adapter, data_by_condition, num_samples=2000):
    """Infer subject-level parameters from all TTA conditions."""
    
    all_posteriors = []
    
    for tta, trials in data_by_condition.items():
        # Prepare in BayesFlow format
        data_dict = {
            'x': trials[np.newaxis, :, :],      # [1, n_trials, 2]
            'tta_condition': np.array([tta]),
            'number_of_trials': np.array([len(trials)])
        }
        
        # Apply adapter
        adapted_data = adapter(data_dict)
        
        # Get posterior
        posterior = approximator.sample(
            conditions=adapted_data,
            num_samples=num_samples
        )
        
        all_posteriors.append(posterior)
    
    # Combine posteriors from all conditions
    # Option 1: Concatenate (treat as independent estimates)
    combined = {}
    for param in all_posteriors[0].keys():
        combined[param] = np.concatenate(
            [p[param] for p in all_posteriors], axis=0
        )
    
    return combined

# Run inference
posterior = infer_subject_parameters(
    approximator=approximator,
    adapter=adapter,
    data_by_condition=data_by_condition,
    num_samples=2000
)

# Extract parameter estimates
for param in ['theta', 'b0', 'k', 'mu_ndt']:
    print(f"{param}: {np.mean(posterior[param]):.4f} "
          f"(95% CI: [{np.percentile(posterior[param], 2.5):.4f}, "
          f"{np.percentile(posterior[param], 97.5):.4f}])")
```

### Step 3: Process All Subjects

```python
all_results = []

for subject_id in df['subject_id'].unique():
    print(f"Processing {subject_id}...")
    
    # Prepare data
    subject_data = df[df['subject_id'] == subject_id]
    data_by_condition = prepare_data(subject_data)  # Your function
    
    # Infer parameters
    posterior = infer_subject_parameters(
        approximator, adapter, data_by_condition
    )
    
    # Store results
    result = {'subject_id': subject_id}
    for param in ['theta', 'b0', 'k', 'mu_ndt', 'sigma_ndt', 
                  'mu_alpah', 'sigma_alpha', 'sigma_cpp']:
        result[f"{param}_mean"] = np.mean(posterior[param])
        result[f"{param}_std"] = np.std(posterior[param])
        result[f"{param}_ci_lower"] = np.percentile(posterior[param], 2.5)
        result[f"{param}_ci_upper"] = np.percentile(posterior[param], 97.5)
    
    all_results.append(result)

# Save results
results_df = pd.DataFrame(all_results)
results_df.to_csv("inferred_parameters_all_subjects.csv", index=False)
```

---

## Common Issues and Solutions

### Issue 1: "KeyError: 'x'"

**Problem**: Simulator returns wrong format

**Solution**: Ensure simulator returns `dict(x=array)`, not nested dictionaries

```python
# WRONG:
return {'tta_condition=2.5': data}

# CORRECT:
return {'x': data}
```

### Issue 2: "Shape mismatch"

**Problem**: Data dimensions don't match expectations

**Solution**: Check shapes:
- Simulator output: `(number_of_trials, 2)` for RT and CPP
- Real data input: `(1, number_of_trials, 2)` - note the batch dimension!

### Issue 3: "Poor parameter recovery"

**Possible causes**:
1. Not enough training (try 50-100 epochs)
2. Not enough simulations per epoch (try 10,000)
3. Prior ranges too wide/narrow
4. Simulator has bugs

**Diagnosis**:
```python
# Check if simulator works
test_sim = ddm_DC_alphaToCpp(
    theta=0.5, b0=1.0, k=1.5,  # reasonable values
    mu_ndt=0.3, sigma_ndt=0.08,
    mu_alpah=0.5, sigma_alpha=0.1, sigma_cpp=0.1,
    number_of_trials=100,
    tta_condition=3.0
)
print(test_sim['x'].shape)  # Should be (100, 2)
print(test_sim['x'][:5])    # Check values look reasonable
```

### Issue 4: "Can't apply to real data"

**Checklist**:
1. ✓ Real data has same features as simulated (RT, CPP)?
2. ✓ Real data has TTA condition labels?
3. ✓ Data grouped by TTA condition?
4. ✓ Adapter applied to real data?
5. ✓ Batch dimension added: `data[np.newaxis, :, :]`?

---

## Model Interpretation

### What do the parameters mean?

```python
# Parameter descriptions for DDM with collapsing boundary

theta       # Decision criterion: urgency/bias parameter
b0          # Initial boundary height: caution/impulsivity
k           # Boundary collapse rate: time pressure effect
mu_ndt      # Mean non-decision time: sensory/motor delays
sigma_ndt   # Variability in non-decision time
mu_alpah    # Mean drift rate: evidence accumulation speed
sigma_alpha # Variability in drift across trials
sigma_cpp   # CPP measurement noise (if you have EEG)
```

### Typical parameter ranges

Based on your priors:
- `theta`: 0.1 - 3.0 (moderate urgency)
- `b0`: 0.5 - 2.0 (moderate caution)
- `k`: 0.1 - 3.0 (some time pressure)
- `mu_ndt`: 0.2 - 0.6 seconds (typical motor delays)
- `sigma_ndt`: 0.06 - 0.1 seconds
- `mu_alpah`: 0.1 - 1.0 (positive = bias to cross)
- `sigma_alpha`: 0.0 - 0.3
- `sigma_cpp`: 0.0 - 0.3

---

## Frequently Asked Questions

### Q: Why does each simulation only use one TTA?

**A**: This is how the network learns conditional distributions. If each simulation contained all TTAs, the network couldn't learn how TTA affects behavior - it would just see average behavior across conditions.

### Q: How do I combine posteriors across TTA conditions?

**A**: Two approaches:

1. **Concatenate** (treat as independent):
```python
all_samples = np.concatenate([posterior_tta25, posterior_tta30, ...])
```

2. **Average** (assume conditions measure same parameters):
```python
mean_estimate = np.mean([posterior_tta25.mean(), posterior_tta30.mean(), ...])
```

The first approach is more conservative and recommended.

### Q: What if I don't have CPP data?

**A**: Use zeros for CPP. The model will still work but may be less accurate. Consider:
- Training a model variant without CPP
- Using only RT data in the simulator
- Modifying the adapter to handle variable-length features

### Q: How many training epochs do I need?

**A**: Start with 50 epochs. Check parameter recovery plots. If recovery is poor (R² < 0.8), train longer. Typical range: 50-100 epochs.

### Q: Can I use this for other conditions (not TTA)?

**A**: Yes! The same framework works for any conditioning variable:
- Different task difficulties
- Different stimulus types  
- Different experimental blocks
- Time-varying parameters

Just replace `tta_condition` with your condition variable.

---

## References and Resources

- **BayesFlow Documentation**: https://bayesflow.readthedocs.io/
- **Conditional Inference Examples**: Look for "conditional" in BayesFlow examples
- **DDM Resources**: Ratcliff & McKoon (2008) for DDM theory

---

## Quick Start Checklist

- [ ] Understand your data structure (256 trials, 4 TTAs, RT + CPP)
- [ ] Fix simulator to return single-TTA format
- [ ] Update adapter to handle `condition_variables`
- [ ] Train model (50+ epochs, 10k sims/epoch)
- [ ] Validate with parameter recovery plots
- [ ] Check condition-specific recovery
- [ ] Prepare real data (group by TTA)
- [ ] Run inference on real subjects
- [ ] Interpret and analyze results

---

## Summary

The key insight for conditional inference in BayesFlow:

1. **Training**: Each simulation has ONE condition → network learns p(θ|data, condition)
2. **Evaluation**: Test on multiple conditions → verify recovery per condition
3. **Application**: Provide real data with known conditions → get subject parameters

The adapter's `.rename("tta_condition", "condition_variables")` is what makes this work!
