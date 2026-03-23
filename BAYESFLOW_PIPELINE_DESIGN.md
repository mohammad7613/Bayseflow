# BayesFlow Conditional Inference Pipeline for DDM Parameter Recovery
## Pedestrian Crossing Behavioral Data Analysis

**Status:** Design & Implementation Guide  
**Model:** DDM with Collapsing Boundaries (Percepton-based)  
**BayesFlow Version:** 2.0.8  
**Date:** February 2026

---

## 1. Executive Summary

Your simulation code defines a **drift-diffusion model (DDM) with collapsing decision boundaries** conditioned on time-to-arrival (TTA). The key insight is:

- **8 parameters are SHARED across all trials and TTA conditions** (not TTA-dependent)
- **Parameter-per-trial variation** exists only for `alpha_trial` (drift rate within trials)
- **TTA is a CONDITIONING VARIABLE**, not a parameter to estimate
- **Response variables** are [choicet, cpp] pairs per trial

This maps perfectly to **BayesFlow conditional inference** where we learn $p(\boldsymbol{\theta} | \mathbf{D}, \text{TTA})$.

---

## 2. Key Design Decisions

### 2.1 TTA as Conditioning Variable (NOT as inference variable)

**ANSWER TO YOUR MAIN QUESTION:**

In BayesFlow:
- ✅ **Include TTA in the inference pipeline as a conditioning variable** 
- ✅ **Use `meta()` function to randomly select ONE TTA per simulation**
- ✅ **Pass TTA through the adapter as `condition_variables`**
- ✅ **The posterior network learns** $p(\boldsymbol{\theta} | \text{Data}, \text{TTA})$

**Why this approach:**
1. More data-efficient: same network learns behavior under all TTA levels
2. Realistic: humans have consistent strategy across TTA conditions (matches your model assumption)
3. Enables posterior predictions for novel TTA values via conditional inference
4. Aligns with BayesFlow 2.0.8 `condition_variables` design pattern

### 2.2 Data Structure

**Behavioral Data Format:**
```
For each participant × TTA condition × trial:
├── Response Time (RT): seconds relative to TTA = 0
├── CPP measurement: neural signal amplitude
└── Meta: TTA condition (2.5, 3.0, 3.5, 4.0 seconds)
```

**Simulator Output Shape:**
- `x`: `(number_of_trials, 2)` → trials × [RT, CPP]
- `tta_condition`: scalar → {2.5, 3.0, 3.5, 4.0}

### 2.3 Parameter Semantics

| Parameter | Meaning | Range | Type |
|---|---|---|---|
| `theta` | Decision criterion (perception threshold) | 0.1-3.0 | Independent |
| `b0` | Initial accumulator boundary | 0.5-2.0 | Independent |
| `k` | Boundary collapse rate | 0.1-3.0 | Independent |
| `mu_ndt` | Mean non-decision time | 0.2-0.6 | Independent |
| `sigma_ndt` | SD of non-decision time | 0.06-0.1 | Independent |
| `mu_alpha` | Mean drift rate | 0.1-1.0 | Independent |
| `sigma_alpha` | Trial-to-trial drift variability | 0.0-0.3 | Independent |
| `sigma_cpp` | CPP measurement noise | 0.0-0.3 | Independent |

**Key:** All parameters are FIXED across trials/conditions. Trial variability in drift (`sigma_alpha`) is modeled stochastically.

---

## 3. BayesFlow 2.0.8 Conditional Workflow

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 CONDITIONAL INFERENCE SETUP                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
            ┌──────────────────────────────┐
            │   Prior (θ) Distribution     │
            │  - 8 parameters uniformly    │
            │    distributed               │
            └──────────────────────────────┘
                              │
                              ▼
            ┌──────────────────────────────┐
            │   Meta-Function (meta)       │
            │  - Sample TTA ∈ {2.5,3.0,    │
            │    3.5, 4.0}                 │
            │  - Set num_trials = 60       │
            └──────────────────────────────┘
                              │
                              ▼
            ┌──────────────────────────────┐
            │   Simulator (Physics)        │
            │  - Evidence accumulation     │
            │  - Collapsing boundary       │
            │  - Generate [RT, CPP] pairs  │
            └──────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
              Simulated Data      Meta Data
              x: (60, 2)          tta_condition
                                      │
                                      ▼
            ┌──────────────────────────────┐
            │   Adapter (Preprocessing)    │
            │  - Broadcast trial count     │
            │  - Standardize x             │
            │  - Rename to:                │
            │    • summary_variables (x)   │
            │    • inference_variables (θ) │
            │    • condition_variables (TTA)
            └──────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
    Summary         Inference      Condition
    Network         Network        Variables
    SetTransformer CouplingFlow
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │NN(Data)→ │  │NN(s,c)→  │  │ TTA      │
    │ s (42d)  │  │ θ (8d)   │  │ (4 vals) │
    └──────────┘  └──────────┘  └──────────┘
        │              │              │
        └──────────────┼──────────────┘
                       ▼
            ┌──────────────────────────┐
            │  CouplingFlow(condition) │
            │  p(θ|data,TTA)          │
            └──────────────────────────┘
                       │
                       ▼
            ┌──────────────────────────┐
            │  Training Loop           │
            │  - Minimize KL divergence│
            │  - Online sampling       │
            │  - N_sim × epochs steps  │
            └──────────────────────────┘
```

### 3.2 Implementation Structure

Your existing `DDM_DC_Pedestrain.py` already implements this correctly. Let's verify and extend:

**Current Implementation Review:**
✅ `prior_DC()` - Defines 8-parameter prior  
✅ `ddm_DC_alphaToCpp()` - Physics simulator for single TTA  
✅ `meta()` - Randomly samples TTA ∈ CONDITIONS  
✅ `adopt()` - Adapter with `condition_variables`  
✅ Conditional simulator setup  

**Status:** Core structure is sound. Needs only minor tweaks for production.

---

## 4. Parameter Recovery Pipeline (Phase 1: Synthetic Data)

### 4.1 Workflow Phases

```
Phase 1: PARAMETER RECOVERY (Synthetic Data)
├── Generate synthetic data from true parameters
├── Train posterior network on synthetic data
├── Evaluate parameter recovery accuracy
├── Compute posteriors for ground-truth θ
└── Validate: p(θ_true | synthetic_data, TTA) ≈ δ(θ_true)

Phase 2: REAL DATA INFERENCE (Behavioral Data)
├── Load and preprocess real behavioral data
├── Format as summary_variables with condition_variables
├── Apply trained posterior network
├── Extract posterior samples for each subject
├── Analyze parameter estimates across subjects
└── Compare with classical/frequentist DDM fitting

Phase 3: CONDITIONAL PREDICTIONS (Model Validation)
├── Posterior predictive distribution: p(y_new | y_old, TTA_new)
├── Compare posterior predictions vs. real behavior on held-out TTA
└── Calibration analysis
```

### 4.2 Parameter Recovery Test

**Objective:** Verify that the posterior network can recover true parameters from synthetic data.

**Test Design:**
1. Sample N=1000 true parameter sets from prior
2. Generate synthetic data for each (M=60 trials per TTA condition × 4 TTAs = 240 trials total? Or per TTA separately?)
3. Train posterior network on generated data
4. Test recovery: sample from posterior given generated data
5. Compute recovery error: $|\hat{\theta} - \theta_{\text{true}}|$

**Key Question to Resolve:** Per-TTA or Multi-TTA organization?

**Two Options:**

**OPTION A: Each Simulation = Single TTA (RECOMMENDED)**
- Meta returns ONE random TTA per simulation
- Simulator generates 60 trials for THAT TTA only
- Output: (60, 2) array + scalar TTA
- Network learns: p(θ | 60 trials, TTA)
- Training data diversity: Network sees same θ across different TTAs (natural)

**OPTION B: Each Simulation = All TTAs**
- Meta returns all TTA conditions
- Simulator generates 60 trials × 4 TTAs = 240 trials total
- Output: (240, 2) array + array of TTA per trial
- Network learns: p(θ | 240 trials)

**Recommendation: OPTION A** (already implemented in your code)
- Matches real experiment structure (separate TTA blocks)
- More flexible for real data (variable trial counts)
- Better conditioning: network explicitly learns TTA dependence
- Aligns with Bayesflow 2.0.8 patterns

---

## 5. Implementation Roadmap

### Phase 1: Parameter Recovery Validation

**Step 1.1: Create dedicated parameter recovery script**
```python
# File: parameter_recovery.py
"""
Synthetic data parameter recovery test for DDM_DC model.
Validates that posterior network can recover ground-truth parameters.
"""

import numpy as np
import bayesflow as bf
import torch
import matplotlib.pyplot as plt
from DDM_DC_Pedestrain import prior_DC, ddm_DC_alphaToCpp, meta, adopt, CONDITIONS

# 1. Generate synthetic dataset for recovery test
# 2. Train posterior network from scratch
# 3. Test recovery on synthetic data
# 4. Produce recovery plots and statistics
```

**Step 1.2: Design recovery experiments**
- Recovery accuracy: Coverage analysis (credible intervals contain truth ~95%)
- Bias analysis: E[θ_posterior] ≈ θ_true
- Efficiency: Posterior width vs. number of trials

**Step 1.3: Implement recovery test harness**
```python
def test_parameter_recovery(
    model_simulator,
    adapter,
    n_test_samples=100,
    num_trials_per_condition=60,
    num_conditions=4
):
    """
    Test parameter recovery with ground-truth parameters.
    
    For each ground-truth parameter set:
    1. Generate synthetic data under all TTA conditions
    2. Pass to trained posterior network
    3. Compare posterior mean to true parameters
    """
    pass
```

### Phase 2: Real Data Integration

**Step 2.1: Data format specification**
```python
# Real data structure:
# Format: Pandas DataFrame or dict with hierarchical indices
# participant_id, tta_condition, trial_id → [RT, CPP]

# Expected shape after aggregation per (participant, TTA):
# (n_participants × 4_TTAs, 60_trials, 2_variables)

# Loading function:
def load_behavioral_data(csv_path):
    """
    Load and format real behavioral data.
    Returns: dict with keys for each participant
    """
    pass
```

**Step 2.2: Real data inference adapter**
```python
def adapt_real_data(behavioral_data_dict, participant_id):
    """
    Transform real behavioral data to BayesFlow format.
    
    Input: 
    - dict with TTA conditions as keys
    - values: (num_trials, 2) arrays
    
    Output:
    - dict with summary_variables, condition_variables
    """
    pass
```

**Step 2.3: Batch inference**
```python
def infer_population_parameters(
    workflow,
    behavioral_data_by_participant,
    n_posterior_samples=5000
):
    """
    Infer parameters for all participants.
    Returns: dict with posterior samples per participant.
    """
    pass
```

### Phase 3: Diagnostics & Validation

**Step 3.1: Posterior predictive checks**
```python
def posterior_predictive_check(
    workflow,
    behavioral_data,
    n_samples=1000
):
    """
    Generate predictions from posterior and compare to observations.
    """
    pass
```

**Step 3.2: Conditional calibration**
```python
def calibration_across_conditions(
    workflow,
    behavioral_data,
    held_out_tta
):
    """
    Train on 3 TTA conditions, predict behavior on 4th.
    Validate conditional generalization.
    """
    pass
```

---

## 6. Code Modifications Required

### 6.1 Verify/Extend DDM_DC_Pedestrain.py

**Current Status:** ✅ Looks complete. Minor cleanup needed.

**Required Changes:**
1. Add docstring explaining conditional inference design
2. Add comments for real data users
3. Create separate `prior_DC_for_real_data()` if bounds differ
4. Add validation checks

### 6.2 Create New: `parameter_recovery_test.py`

See template in Section 5.1

### 6.3 Create New: `inference_workflow.py`

Will contain training and inference for both synthetic and real data.

### 6.4 Create New: `real_data_adapter.py`

Will handle loading and formatting behavioral CSV files.

---

## 7. Data Format Specification

### 7.1 Real Behavioral Data Input

Expected CSV structure (from "1.pdf" article):

```
participant_id, tta_condition, trial_id, response_time, cpp_measurement
1,              2.5,          1,        0.45,         -3.2
1,              2.5,          2,        0.38,         -2.8
...
1,              3.0,          1,        0.52,         -3.5
...
```

### 7.2 Summary Statistics

For aggregation:
```python
{
    'participant_1': {
        'tta_2.5': (60, 2),  # 60 trials, 2 variables
        'tta_3.0': (60, 2),
        'tta_3.5': (60, 2),
        'tta_4.0': (60, 2),
    },
    'participant_2': {...},
    ...
}
```

### 7.3 Adapter Output

After BayesFlow adapter processing:

```python
{
    'summary_variables': shape (n_trials, 2) standardized
    'inference_variables': shape (8,) → [θ, b0, k, μ_ndt, σ_ndt, μ_α, σ_α, σ_cpp]
    'condition_variables': shape (1,) → TTA value
}
```

---

## 8. Network Architecture Recommendations

### 8.1 Summary Network (Data Compression)

**Current:** `SetTransformer(summary_dim=10)`

**Why SetTransformer for behavioral data:**
- Trials are exchangeable (order doesn't matter)
- Natural handling of variable trial counts
- Learns permutation-invariant statistics
- Ideal for per-trial measurements

**Alternative for larger datasets:**
```python
# Option 1: Deeper SetTransformer
summary_network = bf.networks.SetTransformer(
    summary_dim=20,  # Output dimension
    num_blocks=3,    # Number of transformer blocks
)

# Option 2: Combined architecture
summary_network = bf.networks.SetTransformerCoupling(
    summary_net=bf.networks.SetTransformer(summary_dim=15),
    ...
)
```

### 8.2 Inference Network (Posterior)

**Current:** `CouplingFlow()`

**For conditional inference in BayesFlow 2.0.8:**
```python
# Standard: CouplingFlow automatically handles conditioning
inference_network = bf.networks.CouplingFlow(
    num_dimensions=8,      # 8 parameters
    conditional_shape=(1,), # TTA is scalar condition
)

# Alternative: More expressive
inference_network = bf.networks.CouplingFlow(
    num_dimensions=8,
    conditional_shape=(1,),
    num_coupling_layers=6,  # Increase for higher accuracy
    num_dense_layers=4,
)
```

**Why CouplingFlow:**
- Normalizing flow for flexible posterior
- Autoregressive design enables exact likelihood
- Conditionable: $p(\theta|z,c)$ where $z$ = summary, $c$ = TTA

### 8.3 Workflow Assembly

```python
# Current implementation is correct:
workflow = bf.BasicWorkflow(
    simulator=model[0],           # ddm_DC_alphaToCpp simulator
    adapter=model[1],             # Data preprocessor
    inference_network=bf.networks.CouplingFlow(),
    summary_network=bf.networks.SetTransformer(summary_dim=10),
)
```

---

## 9. Training Strategy for Parameter Recovery

### 9.1 Phase 1: Synthetic Data Training

**Configuration:**
```python
# Training hyperparameters
n_simulations = 10000    # Total simulations for training
epochs = 10              # Full epochs through simulator
batch_size = 32          # Batch size for gradient updates
num_batches_per_epoch = 625  # n_simulations / epochs / batch_size

# Optimizer
optimizer = torch.optim.AdamW(
    workflow.approximator.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)

# Training
history = workflow.fit_online(
    epochs=epochs,
    batch_size=batch_size,
    num_batches_per_epoch=num_batches_per_epoch,
    max_queue_size=100,
)
```

**Key Metrics to Monitor:**
- Loss convergence: Should decrease smoothly
- Validation accuracy: Check on held-out synthetic test set
- Parameter recovery: MSE of recovered vs. true parameters

### 9.2 Phase 2: Real Data Fine-tuning (Future)

If needed, can fine-tune on real data via transfer learning:
```python
# Load pre-trained network from synthetic training
workflow.approximator.load_state_dict(
    torch.load('trained_model/ddm_dc_synthetic.pt')
)

# Fine-tune on small real data subset with different learning rate
optimizer = torch.optim.Adam(..., lr=1e-4)  # Lower lr for fine-tuning
# ... continue training with real data batches ...
```

---

## 10. Expected Outcomes & Diagnostics

### 10.1 Parameter Recovery Test Results

**Expected Performance:**

| Metric | Target | Method |
|--------|--------|--------|
| Coverage (95% CI) | 94-96% | Count true params in posterior CI |
| Bias | < 5% of prior range | E[θ̂] - θ_true |
| RMSE | < 0.1 (units) | √(E[(θ̂-θ)²]) |
| Efficiency | HPD width | Compare to Laplace approximation |

### 10.2 Diagnostic Plots

For each parameter:
1. **Recovery scatter:** θ_true vs. θ_posterior_mean (w/ 95% CI band)
2. **Q-Q plot:** Posterior vs. empirical CDFs
3. **Residual plot:** (θ̂ - θ_true) vs. σ_posterior

By TTA:
4. **Conditional recovery:** Separate plots for each TTA value
5. **TTA effect:** Parameter estimates as function of TTA

### 10.3 Inference Statistics

```python
# Report these for each parameter:
statistics = {
    'parameter_name': {
        'mean': float,          # mean of posterior
        'std': float,           # std dev of posterior
        'credible_interval': (lower, upper),  # 95% HPD
        'bias': float,          # E[θ̂] - θ_true
        'rmse': float,          # √E[(θ̂-θ)²]
        'true_value': float,    # ground truth (if synthetic)
    }
}
```

---

## 11. Comparison with Classical DDM Approaches

**Your Model vs. Zgonnikov et al. 2-stage DDM:**

| Feature | Your Model | Zgonnikov |
|---------|-----------|-----------|
| Decision process | Collapsing boundary | Sequential sampling |
| Conditioning | TTA (time pressure) | Explicit time model |
| Measurement | RT + CPP (neural) | Only RT + choice |
| Parameters shared | All (independent TTAs) | All (independent TTAs) |
| Neural integration | Yes (via CPP) | No |
| Boundary form | Sigmoid collapse: $b(t) = \frac{b_0}{1+e^{-k(t-t^*)}}$ | Fixed or linear |

**BayesFlow Advantages:**
1. Automatic ABC without likelihood
2. Handles neural data (CPP) naturally
3. Conditional inference: p(θ|data,TTA) is explicit
4. Amortized inference: single network for all subjects
5. Uncertainty quantification built-in

---

## 12. Implementation Timeline

**Week 1:**
- [ ] Finalize prior distributions (match real data range if needed)
- [ ] Create parameter recovery test script
- [ ] Run 100-trial synthetic recovery experiment

**Week 2:**
- [ ] Implement full training pipeline for synthetic data
- [ ] Train model with 10k+ simulations
- [ ] Generate recovery diagnostics

**Week 3:**
- [ ] Load real behavioral data
- [ ] Implement real data adapter
- [ ] Run inference on first 5 participants (test)

**Week 4:**
- [ ] Scale to full dataset
- [ ] Posterior predictive checks
- [ ] Prepare results paper/report

---

## 13. Bayesflow 2.0.8 Specific Notes

### 13.1 Conditional Inference API

```python
# BayesFlow 2.0.8 supports conditioning via:
adapter.rename('x', 'summary_variables')
adapter.rename('tta_condition', 'condition_variables')

# Workflow handles this automatically:
workflow = bf.BasicWorkflow(
    ...,
    inference_network=bf.networks.CouplingFlow(),
)
# CouplingFlow automatically detects condition_variables
```

### 13.2 Data Format Requirements

```python
# Required keys in adapter output:
{
    'summary_variables': ...,     # Observed data (standardized)
    'inference_variables': ...,   # Parameters to infer
    'condition_variables': ...,   # Conditioning context (TTA)
}

# BayesFlow 2.0.8 backend automatically:
# 1. Extracts condition_variables
# 2. Passes to inference_network as additional input
# 3. Learns p(θ | z, c) where z=summary, c=condition
```

### 13.3 Version Compatibility

✅ BayesFlow 2.0.8 supports:
- `bf.networks.SetTransformer` for data compression
- `bf.networks.CouplingFlow` for conditional posteriors
- `bf.BasicWorkflow` with conditioning
- Online training via `fit_online()`
- PyTorch backend with custom optimizers

---

## 14. Deliverables Checklist

**Phase 1: Design & Validation**
- [x] Architecture design document (THIS FILE)
- [ ] Parameter recovery test suite
- [ ] Synthetic data generation examples
- [ ] Network training harness
- [ ] Diagnostics pipeline

**Phase 2: Implementation**
- [ ] Extended DDM_DC_Pedestrain.py (with comments)
- [ ] parameter_recovery_test.py
- [ ] inference_workflow.py
- [ ] Training scripts with checkpointing

**Phase 3: Real Data Integration**
- [ ] real_data_adapter.py
- [ ] Data loading utilities
- [ ] Population inference scripts
- [ ] Results visualization

**Phase 4: Documentation**
- [ ] Usage guide for parameter recovery
- [ ] Usage guide for real data inference
- [ ] Example Jupyter notebook
- [ ] API reference

---

## References

1. **Your Perception Model:** Article "Improving models of pedestrian crossing behavior using neural signatures of decision-making" (1.pdf)
   - Key insight: DDM + CPP measurement
   - Data structure: (participant × TTA × trial) responses
   - Parameter structure: 8 fixed parameters across conditions

2. **Classical DDM Fitting:** Zgonnikov et al. "Pedestrians' road-crossing decisions: Comparing different drift-diffusion models"
   - 8-parameter Zgonnikov model
   - Multi-condition fitting strategy
   - Pipeline approach BayesFlow can replicate & extend

3. **BayesFlow Documentation:** https://docs.bayesflow.org/ (v2.0.8)
   - Conditional inference: https://docs.bayesflow.org/conditional
   - SetTransformer: https://docs.bayesflow.org/networks#set-based-summary
   - CouplingFlow: https://docs.bayesflow.org/networks#coupling-flows

---

## FAQ

**Q: Should I include TTA as an inference variable or conditioning variable?**  
A: Conditioning variable. Your model assumes parameters are invariant to TTA; only the decision process adapts via collision boundary. TTA should condition the posterior, not be inferred.

**Q: How many trials do I need per TTA condition?**  
A: Your model: 60 trials/condition (BayesFlow can work with variable lengths). Power analysis needed for real data; typically 40-100 trials per condition.

**Q: Can I train on all 4 TTAs simultaneously?**  
A: Yes! That's the design: meta() randomly selects one TTA per simulation, so the network sees all conditions during training. Each iteration samples a fresh simulation with random TTA.

**Q: What's the difference between `summary_variables` and `inference_variables`?**  
A: summary_variables = observed data (compressed). inference_variables = parameters we want to infer. The posterior network learns to map data → parameters.

**Q: How do I handle missing data or variable trial counts?**  
A: Use `.as_set()` in adapter. Treats trials as unordered set, enabling variable-length data. This is already in your code.

**Q: Can I encode prior knowledge about parameter correlations?**  
A: Yes! Define a more structured prior in `prior_DC()` or use a hierarchical model. Currently uniform; can be Gaussian with covariance matrix.

---

## Next Steps

1. **Review this design** against your understanding of the behavioral data
2. **Confirm prior ranges** match expected parameter values from real experiment
3. **Run parameter recovery test** to validate architecture
4. **Collect real data** into specified CSV format
5. **Begin systematic inference** on population data

---

**Questions? See the FAQ section or contact maintainer.**
