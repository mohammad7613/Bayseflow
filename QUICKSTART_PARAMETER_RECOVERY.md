# BayesFlow Parameter Recovery: Quick Start Guide

This guide walks you through implementing the parameter recovery pipeline for your DDM_DC pedestrian crossing model.

## File Structure

```
train_joint_models/
├── BAYESFLOW_PIPELINE_DESIGN.md        ← Full architecture & theory
├── parameter_recovery_test.py           ← Phase 1: Synthetic data validation
├── inference_workflow.py                ← Phase 2: Real data inference (to be created)
├── real_data_adapter.py                 ← Phase 3: Data loading (to be created)
│
├── DDM_DC_Pedestrain.py                 ← Your model (with conditional inference)
├── train.py                             ← Training utilities
├── main.py                              ← Entry point
│
├── 1.pdf                                ← Behavioral data article
├── Pedestrians' road-crossing...pdf     ← Classical DDM reference
│
├── results/
│   └── parameter_recovery/              ← Phase 1 outputs
│       ├── training_loss.png
│       ├── recovery_diagnostics.png
│       └── recovery_diagnostics.csv
│
└── trained_model/
    └── parameter_recovery_checkpoints/  ← Saved networks
        └── posterior_network.pt
```

---

## Quick Start: 3 Steps

### Step 1: Validate Parameter Recovery (5-10 minutes)

Test that your posterior network can recover parameters from synthetic data:

```bash
# Run with default settings (50 test params, 10 epochs, 10k simulations)
python parameter_recovery_test.py

# Or customize:
python parameter_recovery_test.py --n_test_params 20 --epochs 5 --n_sim 5000
```

**What happens:**
1. Generates 50 ground-truth parameter sets from the prior
2. Creates synthetic data for each param set × TTA combination
3. Trains a posterior network on 10k synthetic simulations
4. Tests if network can recover the 50 true parameter sets
5. Produces plots and statistics

**Expected output:**
- `results/parameter_recovery/recovery_diagnostics.png` - Visual validation
- `results/parameter_recovery/recovery_diagnostics.csv` - Detailed statistics
- `results/parameter_recovery/training_loss.png` - Loss convergence
- Coverage ~94-96% (true params in 95% credible intervals)
- Mean bias < 5% of prior range

**Success criteria:**
✓ All 8 parameters have > 90% coverage  
✓ No systematic bias  
✓ Loss converges smoothly  
✓ Recovery scatter plot shows diagonal pattern

---

### Step 2: Understand the Architecture

Read [BAYESFLOW_PIPELINE_DESIGN.md](BAYESFLOW_PIPELINE_DESIGN.md) sections:

- **Section 2: Key Design Decisions** → Why TTA is conditioning variable
- **Section 3: BayesFlow 2.0.8 Conditional Workflow** → Network architecture
- **Section 4: Parameter Recovery Pipeline** → Phases 1-3
- **Section 14: FAQ** → Your specific questions addressed

**Key takeaway:** Your model has 8 parameters that are SHARED across all trials and TTA conditions. TTA conditions should be conditioning variables (not inferred), and the posterior network learns $p(\theta | data, TTA)$.

---

### Step 3: Prepare Real Data Integration (20-30 minutes)

Once parameter recovery validates, you're ready for real data:

**3a. Format behavioral data:**

```csv
participant_id, tta_condition, trial_id, response_time, cpp_measurement
1,              2.5,          1,        0.45,         -3.2
1,              2.5,          2,        0.38,         -2.8
...
1,              3.0,          1,        0.52,         -3.5
...
N,              4.0,          60,       0.41,         -2.9
```

**3b. Load and infer** (pseudocode, implementation in Phase 2):

```python
from parameter_recovery_test import workflow  # Load trained network

# Load behavioral data
behavioral_data = load_csv('behavioral_data.csv')

# Infer parameters for each participant
for participant_id in behavioral_data.unique_participants:
    # Extract data for this participant across all TTAs
    participant_data = behavioral_data[participant_data.participant_id == participant_id]
    
    # Infer posterior
    posterior_samples = workflow.sample_posterior(
        data=participant_data,
        n_samples=5000,
    )
    
    # Analyze results
    posterior_mean = posterior_samples.mean(axis=0)
    credible_intervals = np.percentile(posterior_samples, [2.5, 97.5], axis=0)
    
    print(f"Participant {participant_id} posterior means:", posterior_mean)
```

---

## Key Concepts: TTA Conditioning

### Why NOT Include TTA as an Inference Parameter?

Your model architecture:
```
θ, b0, k, ... (8 fixed parameters shared across trials)
    ↓
For each trial:
  α_trial ~ N(μ_α, σ_α)  # Trial-specific drift
  decision_boundary(t) = b0 / (1 + exp(-k(TTA - t - b0/2)))
    ↓
Generates [RT, CPP] response
```

**Decision:** TTA affects the decision process but NOT the parameters. So:
- ✅ TTA is a **conditioning variable** (goes into condition_variables)
- ✅ Parameters are **shared** across TTA levels
- ✅ Network learns $p(\theta | data, TTA)$ for each TTA
- ✅ Same 8 parameters explain behavior across all TTA conditions

### How Conditional Inference Works

**During training:**
```
For each simulation:
  1. Sample TTA ~ {2.5, 3.0, 3.5, 4.0}
  2. Sample θ ~ Prior
  3. Generate data D(θ, TTA)
  4. Adapter transforms: D → summary_variables, θ → inference_variables, TTA → condition_variables
  5. Network learns: p(θ | summary, condition=TTA)
```

**During inference on real data:**
```
Real data for participant + TTA condition
    ↓
Format as summary_variables + condition_variables
    ↓
Feed to trained network
    ↓
Get posterior p(θ | real_data, TTA)
    ↓
Sample posterior or compute credible intervals
```

---

## Common Questions

### Q: Should I use all TTA data together or separately per TTA?

**A:** Use all TTA data together (recommended approach):
- Aggregate all trials across TTA conditions for each participant
- Let the network learn the conditional dependence naturally
- More data-efficient and robust

Alternative: Infer separately per TTA, then average results. Less efficient.

### Q: How many trials do I need?

**A:** Your model: 60 trials/condition is typical.
- Minimum: ~30 trials for stable inference
- Ideal: 60-100 trials per condition
- More trials → narrower credible intervals

### Q: Can I test on held-out TTA conditions?

**A:** Yes! Use cross-validation:
- Train on TTA ∈ {2.5, 3.0, 3.5}
- Test on held-out TTA = 4.0
- Validates conditional generalization

### Q: What if real data doesn't recover well?

**Possible issues:**
1. Model misspecification (e.g., parameters don't span real data range)
   - Expand prior ranges in prior_DC()
   - Review article 1.pdf for real parameter ranges

2. Insufficient network capacity
   - Increase summary_dim or num_coupling_layers
   - See Section 8 of BAYESFLOW_PIPELINE_DESIGN.md

3. Data quality issues
   - Check for outliers, mistakes in data entry
   - Validate against experiment protocol in 1.pdf

4. Real parameters show condition-dependence
   - If some parameters vary by TTA, need separate inference per TTA
   - Or use hierarchical model

---

## Performance Expectations

### Parameter Recovery (Phase 1):
- **Runtime:** 5-10 min on GPU (10k simulations, 10 epochs)
- **Memory:** ~4GB GPU
- **Accuracy:** > 90% coverage on all parameters

### Real Data Inference (Phase 2):
- **Per-participant inference:** ~1 second (on GPU)
- **Population inference (50 participants):** ~1 minute
- **Uncertainty quantification:** Free (built into posterior)

---

## Next Steps After Parameter Recovery

1. **Review [BAYESFLOW_PIPELINE_DESIGN.md](BAYESFLOW_PIPELINE_DESIGN.md) Sections 5-7**
   - Understand real data format
   - Network architecture tweaks
   - Data loading utilities

2. **Create `real_data_adapter.py`** (provided template in Section 5.2 of design doc)
   - Load CSV behavioral data
   - Format for BayesFlow network
   - Handle variable trial counts

3. **Create `inference_workflow.py`** (provided template in Section 5 of design doc)
   - Inference pipeline for population data
   - Posterior predictive checks
   - Visualization and reporting

4. **Validate on real data**
   - Run inference on first 5 participants as test
   - Check posterior credible intervals for reasonableness
   - Validate against classical DDM fits (if available)

---

## Troubleshooting

### Issue: Parameter recovery has low coverage (<85%)

**Solutions:**
1. Increase training simulations: `--n_sim 20000`
2. Increase epochs: `--epochs 20`
3. Increase summary network capacity: Edit `build_summary_network()` to increase num_blocks
4. Check simulator output ranges match prior ranges
5. Validate synthetic data generates sensible RTs and CPPs

### Issue: Training loss doesn't converge

**Solutions:**
1. Reduce learning rate in train_posterior_network() (lr=5e-4)
2. Increase batch size for stable gradients (batch_size=64)
3. Check data shapes in adapter output
4. Verify condition_variables format (should be float32, shape (batch, 1))

### Issue: Memory error during training

**Solutions:**
1. Reduce batch size: `--batch_size 16`
2. Reduce num_batches_per_epoch (fewer sims per epoch)
3. Run on CPU (slower but lower memory)
4. Reduce network capacity (fewer layers/dims)

---

## References & Further Reading

- **Design Document:** [BAYESFLOW_PIPELINE_DESIGN.md](BAYESFLOW_PIPELINE_DESIGN.md) (20 sections, full rationale)
- **Your Model:** [DDM_DC_Pedestrain.py](DDM_DC_Pedestrain.py)
- **BayesFlow Docs:** https://docs.bayesflow.org/ (v2.0.8)
- **Article 1:** "Improving models of pedestrian crossing behavior using neural signatures of decision-making" (1.pdf)
- **Article 2:** "Pedestrians' road-crossing decisions: Comparing different drift-diffusion models" (Pedestrians' road-crossing decisions.pdf)

---

## File Checklist

After completing Phase 1, you should have:

- ✅ `parameter_recovery_test.py` - Created (this session)
- ✅ `BAYESFLOW_PIPELINE_DESIGN.md` - Created (this session)
- ✅ `DDM_DC_Pedestrain.py` - Already exists, already correct
- ⏳ `real_data_adapter.py` - To be created (Phase 2)
- ⏳ `inference_workflow.py` - To be created (Phase 2)

Results to expect:
- 📊 `results/parameter_recovery/recovery_diagnostics.png`
- 📊 `results/parameter_recovery/training_loss.png`
- 📄 `results/parameter_recovery/recovery_diagnostics.csv`
- 💾 `trained_model/parameter_recovery_checkpoints/posterior_network.pt`

---

## Questions?

Refer to [BAYESFLOW_PIPELINE_DESIGN.md](BAYESFLOW_PIPELINE_DESIGN.md) **Section 14: FAQ** for answers to:
- TTA conditioning vs. inference
- Trial count requirements
- Handling missing data
- Incorporating prior knowledge
- Transfer learning for real data

---

**Status:** Ready to run Phase 1 parameter recovery test!
