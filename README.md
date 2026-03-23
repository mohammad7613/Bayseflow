# Conditional Inference for Pedestrian Crossing DDM - Implementation Complete

## Overview

This implementation provides a complete solution for applying BayesFlow's conditional inference framework to your pedestrian crossing experiment with multiple Time-To-Arrival (TTA) conditions.

## What Was Done

### Core Changes to Existing Code

**`DDM_DC_Pedestrain.py` - Fixed simulator and adapter**:
- ✓ Updated `CONDITIONS` to match your experiment: `[2.5, 3.0, 3.5, 4.0]`
- ✓ Rewrote simulator to return single-TTA format: `{'x': array}` instead of nested dicts
- ✓ Added comprehensive documentation to simulator, meta, and adapter functions
- ✓ Verified adapter correctly handles `condition_variables`

### New Files Created

1. **`evaluation_conditional.ipynb`** - Complete evaluation workflow
   - Parameter recovery testing (overall + per-condition)
   - Real data preparation examples
   - Posterior visualization
   - Full workflow from data loading to inference

2. **`utils_real_data.py`** - Production-ready helper functions
   - `load_and_validate_data()` - Load experimental data
   - `prepare_subject_data()` - Format for BayesFlow
   - `infer_subject_parameters()` - Run inference
   - `process_all_subjects()` - Batch processing
   - Visualization and export functions

3. **`CONDITIONAL_INFERENCE_GUIDE.md`** - Comprehensive documentation
   - Conceptual explanation of conditional inference
   - Detailed code walkthrough
   - Training and evaluation instructions
   - Troubleshooting guide
   - FAQ section

4. **`SUMMARY.md`** - Quick reference
   - What changed and why
   - How to use the new code
   - Validation checklist
   - File organization

5. **`test_setup.py`** - Automated testing
   - Validates simulator output format
   - Checks adapter configuration
   - Tests model sampling
   - Verifies conditional structure

6. **`ENVIRONMENT_SETUP.md`** - Complete environment setup guide
   - Python environment creation
   - Dependency installation
   - GPU/CPU configuration
   - Troubleshooting common issues

7. **`requirements.txt`** - Python dependencies
   - All required packages with versions
   - Tested and compatible versions

8. **`setup_environment.sh`** - Automated setup script
   - One-command environment setup
   - Automatic hardware detection
   - Dependency installation

## Quick Start

### 🚀 Automated Setup (Recommended)

```bash
# Navigate to project directory
cd "/media/mohammad/New Volume/DoctoralSharif/Articles/Matin/train_joint_models"

# Run automated setup (installs everything)
bash setup_environment.sh
```

This will:
- Create virtual environment
- Install all dependencies
- Detect GPU/CPU
- Run verification tests

### 📋 Manual Setup

```bash
# Create and activate virtual environment
python3 -m venv bayesflow_env
source bayesflow_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python test_setup.py
```

**See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for detailed instructions and troubleshooting.**

---

## Usage After Setup

### 1. Activate Environment (Every Session)

```bash
cd "/media/mohammad/New Volume/DoctoralSharif/Articles/Matin/train_joint_models"
source bayesflow_env/bin/activate
```

### 2. Test the Setup

```bash
python main.py
```

Or in Python:
```python
from DDM_DC_Pedestrain import all_models
from train import train_amortizer

model = all_models['model_DC']
train_amortizer(model, 'model_DC', n_sim=10000, epochs=50)
```

### 4. Evaluate Parameter Recovery

Open `evaluation_conditional.ipynb` in Jupyter and run all cells.

### 5. Apply to Real Data

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

# Results saved to: inferred_parameters_all_subjects.csv
```

## File Structure

```
train_joint_models/
├── DDM_DC_Pedestrain.py          ✓ UPDATED - Fixed simulator & adapter
├── models.py                      ✓ UNCHANGED - For other models
├── train.py                       ✓ UNCHANGED - Training functions
├── main.py                        ✓ UNCHANGED - Training script
├── evaluation.ipynb               ✓ UNCHANGED - Old evaluation
│
├── evaluation_conditional.ipynb   ✓ NEW - Conditional inference evaluation
├── utils_real_data.py            ✓ NEW - Helper functions for real data
├── test_setup.py                  ✓ NEW - Automated testing
│
├── CONDITIONAL_INFERENCE_GUIDE.md ✓ NEW - Complete documentation
├── SUMMARY.md                     ✓ NEW - Quick reference
└── README.md                      ✓ NEW - This file
```

## Key Concepts

### The Problem

Your experiment has:
- 256 trials per subject
- 4 TTA conditions (2.5s, 3.0s, 3.5s, 4.0s)
- 64 trials per condition
- Need to estimate subject-level parameters that explain behavior across ALL conditions

### The Solution: Conditional Inference

Train a neural network to learn:
$$p(\theta | \text{data}, \text{TTA})$$

During training:
1. Sample parameters: θ ~ prior
2. Sample TTA: TTA ~ Uniform({2.5, 3.0, 3.5, 4.0})
3. Simulate data: data ~ simulator(θ, TTA)
4. Train network: network(data, TTA) → θ

For real data:
1. Group trials by TTA condition
2. For each condition: posterior ~ network(real_data, TTA)
3. Combine posteriors → subject-level estimate

### Why This Works

The network learns that:
- The SAME parameters produce DIFFERENT data depending on TTA
- When inferring parameters, it uses TTA to interpret the data correctly
- Result: More accurate parameter estimates that generalize across conditions

## Data Format

### Your Real Data Should Look Like

```csv
subject_id,trial_num,block,TTA,direction,RT,collision,CPP
S001,1,1,2.5,left,1.245,0,0.034
S001,2,1,3.0,right,1.678,0,0.042
S001,3,1,3.5,left,-1,0,0.028
...
```

Required columns:
- `subject_id`: Subject identifier
- `TTA`: Time-to-arrival (2.5, 3.0, 3.5, or 4.0)
- `RT`: Reaction time in seconds (-1 for cross-after)
- `CPP`: CPP measure (use zeros if not available)

### What the Code Produces

**Format 1**: Wide format (one row per subject)
```csv
subject_id,theta_mean,theta_std,theta_ci_lower,theta_ci_upper,b0_mean,...
S001,0.523,0.084,0.374,0.682,1.234,...
S002,0.612,0.091,0.443,0.781,1.156,...
...
```

**Format 2**: Long format (one row per parameter per subject)
```csv
subject_id,parameter,mean,median,std,ci_lower,ci_upper
S001,theta,0.523,0.518,0.084,0.374,0.682
S001,b0,1.234,1.229,0.142,0.967,1.501
S002,theta,0.612,0.605,0.091,0.443,0.781
...
```

## Validation Checklist

Before applying to real data:

- [ ] Test setup passes: `python test_setup.py`
- [ ] Model trains without errors
- [ ] Training loss decreases over epochs
- [ ] Parameter recovery R² > 0.8 for all parameters
- [ ] Recovery good for ALL TTA conditions
- [ ] 95% credible intervals cover true values ~95% of the time
- [ ] Posterior distributions are not flat (well-identified)

## Expected Training Time

Hardware dependent, but approximate:
- CPU: ~2-4 hours for 50 epochs (10k sims/epoch)
- GPU: ~20-40 minutes for 50 epochs (10k sims/epoch)

Monitor training progress and check validation loss is decreasing.

## Troubleshooting

### "Syntax error in DDM_DC_Pedestrain.py"
→ Fixed! File has been updated with correct syntax.

### "Poor parameter recovery"
→ Train longer (50-100 epochs)
→ Check simulator for bugs
→ Verify prior ranges

### "Can't apply to real data"
→ Check data format matches expected structure
→ Verify TTA values are [2.5, 3.0, 3.5, 4.0]
→ Ensure adapter is applied to real data

### "Results look strange"
→ Check data preprocessing (units, scaling)
→ Look for outliers or coding errors
→ Verify CPP measurements (if available)

## Documentation

For detailed information, see:

1. **`CONDITIONAL_INFERENCE_GUIDE.md`** - Comprehensive guide
   - Theory and concepts
   - Code walkthrough
   - Training instructions
   - Evaluation methods
   - FAQ

2. **`evaluation_conditional.ipynb`** - Working examples
   - Parameter recovery
   - Data preparation
   - Inference workflow
   - Visualization

3. **`SUMMARY.md`** - Quick reference
   - What changed
   - How to use
   - File organization

## Parameters Explained

```python
theta       # Decision criterion: urgency/bias
b0          # Initial boundary: caution
k           # Collapse rate: time pressure
mu_ndt      # Mean non-decision time
sigma_ndt   # Variability in NDT
mu_alpah    # Mean drift rate
sigma_alpha # Drift variability
sigma_cpp   # CPP noise (if available)
```

## Next Steps

1. **Train**: `python main.py` (or use notebook)
2. **Validate**: Run `evaluation_conditional.ipynb`
3. **Apply**: Use functions from `utils_real_data.py`
4. **Analyze**: Export results and run statistical tests
5. **Publish**: Include parameter estimates and recovery plots

## Support

For issues:
1. Check `CONDITIONAL_INFERENCE_GUIDE.md` for detailed explanations
2. Run `test_setup.py` to diagnose problems
3. Review examples in `evaluation_conditional.ipynb`
4. Check BayesFlow documentation: https://bayesflow.readthedocs.io/

## Citation

If you use this code, please cite:
- BayesFlow: Radev et al. (2020)
- DDM with collapsing bounds: Hawkins et al. (2015)

---

## Status: ✓ Ready to Use

All core functionality implemented and tested.
Backward compatible with existing code.
Comprehensive documentation provided.

**Last Updated**: January 30, 2026
