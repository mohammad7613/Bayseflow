# Visual Workflow Guide: Conditional Inference for Pedestrian Crossing

## The Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                                │
└─────────────────────────────────────────────────────────────────┘

1. SAMPLE PARAMETERS FROM PRIOR
   ┌──────────────────────────────┐
   │ θ ~ prior_DC()              │
   │ • theta = 0.523             │
   │ • b0 = 1.234                │
   │ • k = 1.456                 │
   │ • ... (8 parameters)        │
   └──────────────────────────────┘
                 ↓
2. SAMPLE TTA CONDITION (randomly)
   ┌──────────────────────────────┐
   │ TTA ~ Uniform({2.5, 3.0,    │
   │               3.5, 4.0})    │
   │ → Selected: TTA = 3.5       │
   └──────────────────────────────┘
                 ↓
3. SIMULATE DATA
   ┌──────────────────────────────┐
   │ ddm_DC_alphaToCpp(θ, TTA)   │
   │                             │
   │ Generates 60 trials:        │
   │ [RT₁, CPP₁]                 │
   │ [RT₂, CPP₂]                 │
   │ ...                         │
   │ [RT₆₀, CPP₆₀]               │
   │                             │
   │ Returns: {'x': (60, 2)}     │
   └──────────────────────────────┘
                 ↓
4. ADAPTER TRANSFORMS
   ┌──────────────────────────────┐
   │ adopt(simulation_output)    │
   │                             │
   │ Creates:                    │
   │ • summary_variables         │
   │   (standardized data)       │
   │ • inference_variables       │
   │   (parameters)              │
   │ • condition_variables ← KEY!│
   │   (TTA = 3.5)              │
   └──────────────────────────────┘
                 ↓
5. NEURAL NETWORK LEARNS
   ┌──────────────────────────────┐
   │  Network(data, TTA) → θ     │
   │                             │
   │  Learns: p(θ | data, TTA)   │
   │                             │
   │  Repeat 10,000 times        │
   │  per epoch × 50 epochs      │
   └──────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   EVALUATION PHASE                               │
└─────────────────────────────────────────────────────────────────┘

6. VALIDATION DATA
   ┌──────────────────────────────┐
   │ Generate new simulations    │
   │ (unseen during training)    │
   │                             │
   │ 200 simulations with        │
   │ known parameters            │
   └──────────────────────────────┘
                 ↓
7. GET POSTERIORS
   ┌──────────────────────────────┐
   │ approximator.sample(         │
   │   conditions=val_data,      │
   │   num_samples=1000          │
   │ )                           │
   │                             │
   │ Returns: posterior samples  │
   │ for each parameter          │
   └──────────────────────────────┘
                 ↓
8. CHECK RECOVERY
   ┌──────────────────────────────┐
   │ Compare:                    │
   │ • True parameters (known)   │
   │ • Estimated parameters      │
   │   (posterior means)         │
   │                             │
   │ Plot recovery:              │
   │ R² > 0.8 for all params? ✓  │
   └──────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              APPLICATION TO REAL DATA                            │
└─────────────────────────────────────────────────────────────────┘

9. PREPARE REAL DATA (per subject)
   ┌──────────────────────────────────────────────┐
   │ Subject S001: 256 trials                    │
   │                                             │
   │ Group by TTA:                               │
   │ ┌─────────────────────────────────────────┐ │
   │ │ TTA = 2.5s: 64 trials [RT, CPP]        │ │
   │ │   [1.23, 0.03]                         │ │
   │ │   [1.45, 0.04]                         │ │
   │ │   ... (64 total)                       │ │
   │ └─────────────────────────────────────────┘ │
   │ ┌─────────────────────────────────────────┐ │
   │ │ TTA = 3.0s: 64 trials [RT, CPP]        │ │
   │ └─────────────────────────────────────────┘ │
   │ ┌─────────────────────────────────────────┐ │
   │ │ TTA = 3.5s: 64 trials [RT, CPP]        │ │
   │ └─────────────────────────────────────────┘ │
   │ ┌─────────────────────────────────────────┐ │
   │ │ TTA = 4.0s: 64 trials [RT, CPP]        │ │
   │ └─────────────────────────────────────────┘ │
   └──────────────────────────────────────────────┘
                 ↓
10. INFER PARAMETERS (per TTA condition)
   ┌──────────────────────────────────────────────┐
   │ For TTA = 2.5s:                             │
   │   posterior₂.₅ = approximator.sample(       │
   │     conditions={                            │
   │       'x': subject_data_tta_2.5,           │
   │       'tta_condition': 2.5                 │
   │     }                                       │
   │   )                                         │
   │                                             │
   │ Repeat for TTA ∈ {3.0, 3.5, 4.0}           │
   └──────────────────────────────────────────────┘
                 ↓
11. COMBINE POSTERIORS
   ┌──────────────────────────────────────────────┐
   │ Concatenate posteriors from all TTAs:       │
   │                                             │
   │ posterior_all = concat(                     │
   │   posterior₂.₅,                             │
   │   posterior₃.₀,                             │
   │   posterior₃.₅,                             │
   │   posterior₄.₀                              │
   │ )                                           │
   │                                             │
   │ → 8,000 total samples (2,000 per TTA)      │
   └──────────────────────────────────────────────┘
                 ↓
12. EXTRACT ESTIMATES
   ┌──────────────────────────────────────────────┐
   │ For each parameter:                         │
   │                                             │
   │ theta:                                      │
   │   mean  = 0.567                            │
   │   95% CI = [0.421, 0.713]                  │
   │                                             │
   │ b0:                                         │
   │   mean  = 1.189                            │
   │   95% CI = [1.023, 1.355]                  │
   │                                             │
   │ ... (all 8 parameters)                     │
   └──────────────────────────────────────────────┘
                 ↓
13. REPEAT FOR ALL SUBJECTS
   ┌──────────────────────────────────────────────┐
   │ Process subjects: S001, S002, ..., S030     │
   │                                             │
   │ Save results to CSV:                        │
   │ subject_id | theta_mean | theta_std | ...  │
   │ -----------|------------|-----------|-----  │
   │ S001       | 0.567      | 0.074     | ...  │
   │ S002       | 0.612      | 0.081     | ...  │
   │ ...        | ...        | ...       | ...  │
   └──────────────────────────────────────────────┘
```

## Code Mapping to Pipeline

### Training (Steps 1-5)

```python
# This is what train_amortizer() does internally

from DDM_DC_Pedestrain import all_models
from train import train_amortizer

model = all_models['model_DC']  # Contains prior, simulator, meta, adapter
train_amortizer(model, 'model_DC', n_sim=10000, epochs=50)

# Internally:
# - Calls prior_DC() → parameters (Step 1)
# - Calls meta() → randomly selects TTA (Step 2)
# - Calls ddm_DC_alphaToCpp() → simulates data (Step 3)
# - Calls adopt() → transforms data (Step 4)
# - Trains network (Step 5)
```

### Evaluation (Steps 6-8)

```python
# evaluation_conditional.ipynb

import keras
import bayesflow as bf

# Load trained model
approximator = keras.saving.load_model('trained_model1/checkpoints/model_DC.keras')

# Generate validation data (Step 6)
val_sims = model.sample(200)

# Get posteriors (Step 7)
post_draws = approximator.sample(conditions=val_sims, num_samples=1000)

# Check recovery (Step 8)
bf.diagnostics.plots.recovery(
    estimates=post_draws,
    targets=val_sims,
    variable_names=['theta', 'b0', 'k', ...]
)
```

### Application (Steps 9-13)

```python
# Using utils_real_data.py

from utils_real_data import (
    load_and_validate_data,
    process_all_subjects
)

# Load data (Step 9)
df = load_and_validate_data('pedestrian_data.csv')

# Process all subjects (Steps 10-13)
results = process_all_subjects(
    df=df,
    approximator=approximator,
    adapter=adapter,
    num_samples=2000
)

# Results saved automatically!
```

## Key Files and Their Roles

```
DDM_DC_Pedestrain.py
├── prior_DC()           → Step 1: Sample parameters
├── ddm_DC_alphaToCpp()  → Step 3: Simulate data
├── meta()               → Step 2: Select TTA
└── adopt()              → Step 4: Transform data

train.py
└── train_amortizer()    → Step 5: Train network

evaluation_conditional.ipynb
├── Load model           → Setup
├── model.sample()       → Step 6: Validation data
├── approximator.sample() → Step 7: Get posteriors
└── recovery plot        → Step 8: Check recovery

utils_real_data.py
├── load_and_validate_data()    → Step 9: Load real data
├── prepare_subject_data()      → Step 9: Format data
├── infer_subject_parameters()  → Steps 10-11: Infer
└── process_all_subjects()      → Steps 12-13: Batch process
```

## The Magic: condition_variables

The critical line that makes conditional inference work:

```python
# In adopt() function:
.rename("tta_condition", "condition_variables")
```

This tells BayesFlow:
- "tta_condition" is CONTEXT (known during inference)
- NOT something to infer (like theta, b0, etc.)

During training:
- Network sees: (data, TTA=2.5) → parameters
- Network sees: (data, TTA=3.0) → parameters
- Network sees: (data, TTA=3.5) → parameters
- Network sees: (data, TTA=4.0) → parameters

Network learns: "Same parameters can produce different data depending on TTA"

During inference:
- You provide: (real_data, TTA=2.5)
- Network thinks: "What parameters would produce this data at TTA=2.5?"
- Returns: posterior distribution over parameters

## Common Mistakes vs Correct Approach

### ❌ WRONG: Simulate all TTAs at once

```python
# Don't do this!
def simulator_wrong(..., tta_conditions=[2.5, 3.0, 3.5, 4.0]):
    for tta in tta_conditions:
        # simulate for each TTA
    return {'tta_2.5': data1, 'tta_3.0': data2, ...}
```

Problem: Network can't learn condition-specific effects

### ✓ CORRECT: Simulate one TTA per call

```python
# Do this!
def ddm_DC_alphaToCpp(..., tta_condition):
    # tta_condition is a SINGLE value (e.g., 3.0)
    # simulate 60 trials with this TTA
    return {'x': data}  # Shape: (60, 2)
```

Benefit: Network learns p(θ | data, TTA)

## Summary

**The key insight**: Each simulation during training represents ONE experimental condition. The network learns how conditions affect data. During inference, you provide the known condition, and the network uses it to interpret your data correctly.

This allows you to estimate subject-level parameters that generalize across all TTA conditions!
