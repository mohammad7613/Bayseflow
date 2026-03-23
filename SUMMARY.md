# Summary: Conditional Inference Implementation for Pedestrian Crossing Experiment

## What Was Changed

### 1. Fixed Simulator Structure (`DDM_DC_Pedestrain.py`)

**Before**: Simulator returned nested dictionary with keys like `'tta_condition=2.5'`
```python
# OLD (incorrect)
return {'tta_condition=2.5': data_for_tta_25, 'tta_condition=3.0': data_for_tta_30, ...}
```

**After**: Simulator returns single-TTA data in correct format
```python
# NEW (correct)
def ddm_DC_alphaToCpp(..., tta_condition, ...):
    # Simulate trials for ONE TTA condition
    return {'x': array([RT, CPP] pairs)}  # Shape: (n_trials, 2)
```

**Why**: BayesFlow learns p(θ|data, TTA) by seeing many examples of (parameters, TTA, data) tuples. Each simulation should represent ONE condition.

### 2. Updated CONDITIONS to Match Experiment

```python
# OLD: CONDITIONS = np.array([2.0, 3.5, 5.0])
# NEW: 
CONDITIONS = np.array([2.5, 3.0, 3.5, 4.0])  # Matches actual experiment
```

### 3. Enhanced Adapter with Documentation

Added clear comments explaining each transformation step, especially:
```python
.rename("tta_condition", "condition_variables")  # This is the KEY line!
```

This tells BayesFlow that `tta_condition` is a conditioning variable (context) not a parameter to infer.

## New Files Created

### 1. `evaluation_conditional.ipynb`
Comprehensive notebook demonstrating:
- Parameter recovery evaluation (overall and per-condition)
- How to prepare real data for inference
- Complete workflow from loading data to extracting parameters
- Visualization of posterior distributions

### 2. `utils_real_data.py`
Production-ready utility functions:
- `load_and_validate_data()`: Load and check experimental data
- `prepare_subject_data()`: Format data for BayesFlow
- `infer_subject_parameters()`: Run inference for one subject
- `process_all_subjects()`: Batch processing for full dataset
- `plot_subject_posteriors()`: Visualize results
- `export_for_statistical_analysis()`: Export for SPSS/R/etc.

### 3. `CONDITIONAL_INFERENCE_GUIDE.md`
Complete documentation covering:
- Conceptual explanation of conditional inference
- Detailed code walkthrough
- Training and evaluation workflows
- Troubleshooting common issues
- FAQ section

## How to Use

### Step 1: Train the Model

```bash
# From the workspace directory
python main.py
```

Or in Python:
```python
from DDM_DC_Pedestrain import all_models
from train import train_amortizer

model = all_models['model_DC']
train_amortizer(model, 'model_DC', n_sim=10000, epochs=50)
```

### Step 2: Evaluate Parameter Recovery

Open `evaluation_conditional.ipynb` and run cells to:
- Check overall parameter recovery
- Verify recovery for each TTA condition separately
- Ensure the model is working correctly before applying to real data

### Step 3: Apply to Real Data

```python
import keras
from DDM_DC_Pedestrain import all_models
from utils_real_data import process_all_subjects, load_and_validate_data

# Load data
df = load_and_validate_data('your_data.csv')

# Load trained model
model, adapter = all_models['model_DC']
approximator = keras.saving.load_model('trained_model1/checkpoints/model_DC.keras')

# Process all subjects
results = process_all_subjects(
    df=df,
    approximator=approximator,
    adapter=adapter,
    num_samples=2000
)

# Results are saved automatically to 'inferred_parameters_all_subjects.csv'
```

## Key Concepts

### What the Model Learns

During training, the model sees:
- Parameters: θ, b0, k, μ_ndt, σ_ndt, μ_α, σ_α, σ_cpp
- Condition: TTA ∈ {2.5, 3.0, 3.5, 4.0}
- Data: [RT, CPP] pairs

It learns: **p(parameters | data, TTA)**

### Why This Matters for Your Data

In your experiment:
- Each subject has 256 trials
- Distributed across 4 TTA conditions (64 each)
- The SAME subject parameters should explain behavior across ALL conditions
- But behavior DIFFERS by condition (e.g., faster RTs at longer TTAs)

The conditional inference framework allows the model to:
1. Account for TTA when interpreting the data
2. Infer subject-level parameters that generalize across conditions

### Data Structure

**Training data** (simulated):
```python
{
    'x': (60, 2),              # 60 trials, [RT, CPP]
    'theta': 0.523,            # Parameters (same for all trials)
    'b0': 1.234,
    # ...
    'tta_condition': 3.5,      # ONE TTA for this simulation
    'number_of_trials': 60
}
```

**Real data** (per subject, per condition):
```python
# For TTA=2.5s
{
    'x': (64, 2),              # 64 trials from real subject
    'tta_condition': 2.5,
    'number_of_trials': 64
}
# Repeat for TTA ∈ {3.0, 3.5, 4.0}
```

## Backward Compatibility

The changes maintain compatibility with your existing code:

- ✓ `models.py` still works (for non-conditional models)
- ✓ `train.py` functions unchanged
- ✓ `evaluation.ipynb` still works for old models
- ✓ New conditional code is in separate files

## Validation Checklist

Before applying to real data, verify:

- [ ] Model trains without errors
- [ ] Training loss decreases over epochs
- [ ] Parameter recovery plots show good correlation (R² > 0.8)
- [ ] Recovery is good for ALL TTA conditions
- [ ] Posterior credible intervals have ~95% coverage
- [ ] Simulator produces realistic data (check histograms)

## Expected Results

After training (50 epochs, 10k sims/epoch):
- Parameter recovery R² should be > 0.80 for all parameters
- 95% credible intervals should contain true value ~95% of the time
- Recovery should be consistent across all 4 TTA conditions

For real data:
- Posterior distributions should be reasonably narrow (not flat)
- Parameter estimates should be within prior ranges
- Estimates should be similar for repeated measurements of same subject

## Troubleshooting

### "Poor parameter recovery"
→ Train longer (50-100 epochs)
→ Check simulator for bugs
→ Verify prior ranges are reasonable

### "KeyError: 'x'"
→ Check simulator returns `{'x': data}` not nested dict
→ Verify data shape is (n_trials, 2)

### "Can't apply to real data"
→ Ensure data has 'TTA', 'RT', (optionally 'CPP') columns
→ Check data is grouped by TTA condition
→ Verify adapter is applied to real data

### "Results look strange"
→ Check data preprocessing (units, scaling, missing values)
→ Verify TTA values match CONDITIONS
→ Check for outliers or coding errors in real data

## Files Summary

```
train_joint_models/
├── DDM_DC_Pedestrain.py          # ✓ Updated simulator & adapter
├── train.py                       # (unchanged)
├── models.py                      # (unchanged - for other models)
├── evaluation_conditional.ipynb   # ✓ NEW: Full evaluation workflow
├── utils_real_data.py            # ✓ NEW: Helper functions
├── CONDITIONAL_INFERENCE_GUIDE.md # ✓ NEW: Complete documentation
└── SUMMARY.md                     # ✓ This file
```

## Next Steps

1. **Train**: Run `python main.py` or train via notebook
2. **Validate**: Run all cells in `evaluation_conditional.ipynb`
3. **Apply**: Use `utils_real_data.py` functions on your real data
4. **Analyze**: Export results and run statistical analyses
5. **Publish**: Include parameter estimates and recovery plots in paper

## Citation

If you use this code in your research, consider citing:
- BayesFlow framework: Radev et al. (2020)
- DDM with collapsing bounds: Hawkins et al. (2015)
- Your paper (once published!)

## Questions?

Refer to:
1. `CONDITIONAL_INFERENCE_GUIDE.md` for detailed explanations
2. `evaluation_conditional.ipynb` for working examples
3. BayesFlow documentation: https://bayesflow.readthedocs.io/

## Contact

For issues specific to this implementation, check:
- Simulator output format
- Adapter configuration
- Data preparation steps

For BayesFlow questions, see official docs and examples.

---

**Last updated**: January 2026
**Status**: Ready for training and evaluation
**Compatibility**: BayesFlow >=1.0, Python >=3.8
