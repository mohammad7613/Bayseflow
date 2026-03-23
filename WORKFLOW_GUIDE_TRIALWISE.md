# Trial-Wise DDM-DC Model: Complete Workflow Guide

## Overview

This guide walks you through the complete workflow for training, evaluating, and using your trial-wise DDM-DC models with BayesFlow.

**Three Main Phases:**
1. **TRAINING**: Train amortizers on simulated DDM-DC data
2. **RECOVERY ANALYSIS**: Validate parameter recovery performance
3. **REAL DATA INFERENCE**: Estimate 8 parameters from subject behavior (CPP + RT)

---

## Quick Start

### 1. Fix Applied
✅ **Fixed**: `Adapter.concatenate()` now works correctly (removed unsupported `along_axis` parameter)
- See file: `DDM_DC_Pedestrain_TrialWise.py` line 178

### 2. Run Complete Workflow

```bash
# Train all models
python workflow_trialwise.py train

# Run recovery analysis  
python workflow_trialwise.py recovery

# Run inference (ready for real data)
python workflow_trialwise.py inference
```

Or run all phases:
```bash
python workflow_trialwise.py
```

---

## Detailed Phases

### PHASE 1: TRAINING THE AMORTIZERS

#### What Happens
- Simulates DDM-DC trial data using your prior distribution
- Trains amortizers to map from summary statistics (RT, CPP) → posterior over 8 parameters
- Saves checkpoints every epoch to allow resuming

#### Configuration (in `workflow_trialwise.py`)
```python
CONFIG["training"] = {
    "n_sim": 1000,          # Simulations per epoch (batch)
    "epochs": 100,          # Total training epochs
    "batch_size": 32,        # Mini-batch size for optimizer
    "resume_epochs": 50,    # Additional epochs when resuming training
}
```

#### Training Code
```python
from workflow_trialwise import train_single_model, train_all_models

# Train one model
history = train_single_model("model_DC_TrialWise", resume=False)

# Train all models sequentially
histories = train_all_models(resume_existing=True)
```

#### Checkpoints
- **Location**: `trained_model1/checkpoints/`
- **File format**: `{model_name}.keras` (Keras 3 with PyTorch backend)
- **What's saved**: Full trained network weights + optimizer state
- **Resume**: Automatically detects and loads existing checkpoints

#### Outputs
- `trained_model1/checkpoints/{model_name}.keras` - Trained amortizer
- `logs/workflow.log` - Training log with timestamps

---

### PHASE 2: RECOVERY ANALYSIS

#### What Happens
1. **Generate test data**: Simulates new trials with known true parameters
2. **Run amortized inference**: Your trained network estimates posterior over parameters
3. **Calculate metrics**: Compares inferred vs true parameters
4. **Save results**: Posterior samples and recovery statistics

#### Configuration
```python
CONFIG["recovery"] = {
    "n_test_sims": 5000,    # Test simulations for evaluation
    "n_posterior_samples": 5000  # MCMC samples for posterior
}
```

#### Recovery Analysis Code
```python
from workflow_trialwise import run_recovery_analysis, load_trained_model

# Recovery for one model
results = run_recovery_analysis("model_DC_TrialWise", n_test=5000)

# Recovery for all models
all_results = run_recovery_all_models()

# Access results
posterior_samples = results['posterior_samples']  # Shape: (5000, 8)
true_params = results['true_params']             # True parameter values
```

#### Recovery Outputs
- `results/{model_name}_recovery.npz` - Binary file with posterior samples
- `results/{model_name}_recovery.json` - Metadata (shapes, timestamps)

#### Interpreting Results
- **Good recovery**: Posterior means should correlate highly with true parameter values
- **Poor recovery**: May indicate insufficient training or data
- **Posterior spread**: Tight posteriors = confident estimates

---

### PHASE 3: REAL DATA INFERENCE

#### What Happens
Given a subject's behavioral data (CPP + RT), your trained model estimates the posterior distribution over the 8 DDM-DC parameters.

#### Parameters Estimated (8 total)
1. `v` - Drift rate (decision signal strength)
2. `a` - Boundary separation (decision threshold)
3. `t0` - Non-decision time (encoding + response latency)
4. `z` - Relative start point (bias)
5. `ndt` - Non-decision time (if separate)
6. `dc` - CPP-based decision criterion
7. `sv` - Drift rate variability (trial-to-trial)
8. `st` - Start point variability

#### Single Subject Inference
```python
from workflow_trialwise import infer_subject_parameters

# Prepare data
subject_data = {
    'subject_id': 'S001',
    'cpp': np.array([0.5, 0.8, 0.3, ...]),        # CPP values per trial
    'reaction_times': np.array([0.45, 0.52, ...])  # RT per trial
}

# Run inference
result = infer_subject_parameters(subject_data, model_name='model_DC_TrialWise')

# Access results
posterior_samples = result['posterior_samples']    # Shape: (5000, 8)
posterior_mean = result['posterior_mean']          # Point estimate
posterior_std = result['posterior_std']            # Uncertainty
param_names = result['param_names']                # ['v', 'a', 't0', ...]
```

#### Batch Inference (Multiple Subjects)
```python
from workflow_trialwise import infer_batch_subjects

# Load your subjects' data
subject_list = [
    {
        'subject_id': 'S001',
        'cpp': np.array([...]),
        'reaction_times': np.array([...])
    },
    {
        'subject_id': 'S002',
        'cpp': np.array([...]),
        'reaction_times': np.array([...])
    },
    # ... more subjects
]

# Run batch inference
results_df = infer_batch_subjects(subject_list, model_name='model_DC_TrialWise')

# Save and analyze
results_df.to_csv('estimated_parameters.csv')
print(results_df.head())
```

#### Inference Outputs
- DataFrame with columns: `subject_id`, `v_mean`, `v_std`, `v_median`, `a_mean`, ... (for all 8 params)
- CSV file: `results/batch_inference_{model_name}_{timestamp}.csv`

#### Data Format Requirements

Your real data must have:
- **CPP values**: Neural marker of decision processes (per trial)
- **Reaction Times**: Behavioral response latency (per trial)
- Same number of trials for both CPP and RT
- Data type: `np.ndarray` or list of floats

Example data structure:
```python
subject_data = {
    'cpp': np.array([0.45, 0.52, 0.38, ..., 0.61]),           # n_trials
    'reaction_times': np.array([0.450, 0.520, 0.380, ..., 0.615])  # n_trials
}
```

---

## Advanced Usage

### Interactive Training with Progress Monitoring
```python
from workflow_trialwise import train_single_model, load_trained_model

# Start fresh training
history = train_single_model("model_DC_TrialWise", resume=False)

# Check checkpoint exists
import os
checkpoint = os.path.join("trained_model1/checkpoints", "model_DC_TrialWise.keras")
print(f"Checkpoint exists: {os.path.exists(checkpoint)}")

# Resume training later
history = train_single_model("model_DC_TrialWise", resume=True)

# Load trained model for custom use
workflow = load_trained_model("model_DC_TrialWise")
```

### Custom Inference with Posterior Analysis
```python
from workflow_trialwise import infer_subject_parameters
import matplotlib.pyplot as plt

# Run inference
result = infer_subject_parameters(subject_data, model_name='model_DC_TrialWise')

# Plot posterior distributions
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
axes = axes.flatten()

for i, param_name in enumerate(result['param_names']):
    axes[i].hist(result['posterior_samples'][:, i], bins=50, alpha=0.7)
    axes[i].axvline(result['posterior_mean'][i], color='r', label='Mean')
    axes[i].axvline(result['posterior_median'][i], color='g', linestyle='--', label='Median')
    axes[i].set_title(f'{param_name}')
    axes[i].legend()

plt.tight_layout()
plt.savefig('posterior_distributions.png')
```

### Comparing Models
```python
from workflow_trialwise import run_recovery_all_models

# Run recovery for all models
results = run_recovery_all_models()

# Compare which model recovers parameters best
for model_name, result in results.items():
    print(f"\n{model_name}:")
    print(f"  Posterior shape: {result['posterior_samples'].shape}")
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'bayesflow'"
**Solution**: Activate the environment
```bash
source bayesflow_env/bin/activate.fish  # fish shell
# or
source bayesflow_env/bin/activate       # bash/zsh
```

### Issue: "CUDA out of memory"
**Solutions**:
1. Reduce batch size: `CONFIG["training"]["batch_size"] = 16`
2. Reduce simulations per epoch: `CONFIG["training"]["n_sim"] = 500`
3. Use CPU: The code automatically falls back to CPU if CUDA unavailable

### Issue: "Checkpoint not found" during recovery
**Solution**: Make sure training completed and saved checkpoint
```bash
ls -lh trained_model1/checkpoints/
```

### Issue: Poor parameter recovery
**Reasons**:
1. Insufficient training epochs - increase `CONFIG["training"]["epochs"]`
2. Insufficient simulations - increase `CONFIG["training"]["n_sim"]`
3. Data quality - ensure CPP and RT values are reasonable ranges

---

## Pipeline Architecture

```
Real Data (CPP + RT per trial)
        ↓
[Adapter: Concatenate & Normalize]
        ↓
[Summary Network: Extract features from trials]
        ↓
[Inference Network: Coupling flows mapping to posterior]
        ↓
Posterior Distribution over 8 Parameters
        ↓
Point Estimates (mean, median, std)
```

---

## File Organization

```
train_joint_models/
├── DDM_DC_Pedestrain_TrialWise.py     # Model definitions with adapter
├── workflow_trialwise.py               # Complete workflow (NEW)
├── train.py                            # Training utilities
├── utils_real_data.py                  # Real data utilities
├── trained_model1/
│   └── checkpoints/
│       ├── model_DC_TrialWise.keras
│       ├── model_1a.keras
│       └── ...
├── results/
│   ├── model_DC_TrialWise_recovery.npz
│   ├── model_DC_TrialWise_recovery.json
│   └── batch_inference_model_DC_TrialWise_*.csv
└── logs/
    └── workflow.log
```

---

## Next Steps

1. **Run training**: `python workflow_trialwise.py train` (30-60 min per model)
2. **Evaluate recovery**: `python workflow_trialwise.py recovery` (10-15 min per model)
3. **Prepare real data**: Format your subject data (CPP + RT)
4. **Run inference**: `infer_batch_subjects(your_data, model_name)`
5. **Analyze posteriors**: Check parameter estimates, uncertainty, correlations

---

## Reference

- **BayesFlow Docs**: https://github.com/stefanradev93/BayesFlow
- **DDM Review**: Ratcliff & McKoon (2008)
- **CPP Literature**: Carbine et al., Twomey et al.

---

**Last Updated**: 2026-02-13
**Model Type**: DDM-DC Trial-Wise with Neural Constraints
