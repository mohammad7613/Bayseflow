# 🎯 ONE-PAGE SUMMARY: TTA in BayesFlow

## Your Question
> **"How should I involve TTA conditions in Bayesflow?"**

---

## Quick Answer

```
DON'T:  Infer TTA as the 9th parameter
        Train 4 separate networks per TTA
        Mix TTA with the 8 cognitive parameters

DO:     Use TTA as a CONDITIONING VARIABLE
        Train ONE network learning p(θ | data, TTA)
        Keep 8 parameters shared across all TTAs
```

---

## Why This Works

### Your Model Structure
```
FIXED 8 PARAMETERS (shared across trials & TTAs):
├── θ, b0, k (decision dynamics)
├── μ_ndt, σ_ndt (timing)
├── μ_α, σ_α (drift rate)
└── σ_cpp (measurement noise)

↓

FOR EACH TRIAL:
├── α_trial ~ N(μ_α, σ_α)          [Trial-level variation]
├── Decision boundary adapts to TTA  [Exogenous condition]
└── Generate RT & CPP

↓

SAME 8 PARAMETERS EXPLAIN DATA ACROSS ALL TTAS
```

### Why Conditioning is Better

| Approach | Networks | Data/Network | Extrapolation | Efficiency |
|----------|----------|---------------|---------------|-----------|
| **Separate** (Wrong) | 4 | 25% | ❌ No | ❌ Poor |
| **Conditional** (Right) | 1 | 100% | ✅ Yes | ✅ Best |

---

## Implementation (You Already Have This!)

```python
# Your meta() function - samples ONE TTA per simulation ✓
def meta():
    tta = RNG.choice([2.5, 3.0, 3.5, 4.0])  # ONE random TTA
    return {"tta_condition": tta, "number_of_trials": 60}

# Your simulator - generates data for that ONE TTA ✓
def ddm_DC_alphaToCpp(..., tta_condition, ...):
    # Use the single TTA value
    # Return (60, 2) array of [RT, CPP] pairs

# Your adapter - renames TTA to condition_variables ✓
def adopt(p):
    adapter = (
        bf.Adapter()
        .standardize("x")
        .concatenate(list(p.keys()), into="inference_variables")  # 8 params
        .rename("x", "summary_variables")                          # Data
        .rename("tta_condition", "condition_variables")            # TTA ← KEY!
    )
    return adapter
```

---

## What the Network Learns

```
                    TRAINING
                       ↓
    ┌─────────────────────────────────┐
    │  Generate 10,000 simulations:   │
    │  - Random θ_i from prior        │
    │  - Random TTA_i \in {2.5,...}   │
    │  - Generate Data D_i from model │
    │  - Format: (D, TTA) → posterior  │
    │                                 │
    │  Network sees varied:            │
    │  ✓ All parameter combinations   │
    │  ✓ All 4 TTA values            │
    │  ✓ How posteriors shift with TTA│
    └─────────────────────────────────┘
                       ↓
            LEARNS p(θ | data, TTA)
                       ↓
    ┌──────────────────────────────────┐
    │  Can then infer on REAL DATA:    │
    │  - Given behavioral observations │
    │  - AND the TTA condition         │
    │  - Produces posterior over θ     │
    │                                  │
    │  Bonus:                          │
    │  ✓ Same θ across diff TTAs       │
    │  ✓ Posteriors adapt to TTA       │
    │  ✓ Single network for all        │
    └──────────────────────────────────┘
```

---

## Visual: Data Flow

```
TRAINING LOOP (10,000 iterations):

Iteration 1:
  θ ~ Prior
  TTA ~ {2.5, 3.0, 3.5, 4.0}  ← Random pick
  D ~ Simulator(θ, TTA)
       ↓
  X = {summary_vars: standardize(D), 
       condition_vars: [TTA],
       inference_vars: θ}
       ↓
  Network sees: [summary, condition] → learn posterior

Iteration 2:
  (Different θ, possibly different TTA)
  Repeat...

Iteration 10,000:
  Network has seen all combinations
  Learned: p(θ | D, TTA) ✓
```

---

## Validation: What to Expect

### After Training (parameter recovery phase):

```
For 50 ground-truth parameter sets:

Generate data → Run through network → Check recovery

    θ_true ──────────┤
                     │   ← Are they close?
    θ_posterior_mean ├──→ YES for all 8 params? ✓
                     │
                     └─→ NO for some param? ❌ Debug

Expected: 95% of θ_true fall in posterior credible intervals
```

---

## Key Differences: Same Data, Different TTA

```
SCENARIO: Human sees same stimulus pattern, but under different TTA

Same observed data D
   ├─→ Network asked: p(θ | D, TTA=2.5) → Posterior A
   └─→ Network asked: p(θ | D, TTA=4.0) → Posterior B

Posterior A ≠ Posterior B  ← This is CORRECT!

Why? Because under time pressure (short TTA), 
different decision thresholds produce same data pattern.

But across real trials, network learns:
"Same cognitive θ values, just conditioned on timing"
```

---

## Files You Now Have

| File | Purpose |
|------|---------|
| `TTA_CONDITIONING_GUIDE.md` | **Read this first** (20 min) - Explains every detail |
| `parameter_recovery_test.py` | **Run this** (10 min) - Validates architecture |
| `BAYESFLOW_PIPELINE_DESIGN.md` | Full reference (45 min) - Everything in detail |
| `QUICKSTART_PARAMETER_RECOVERY.md` | Getting started (10 min) - Next steps |

---

## RUN THIS NOW

```bash
python parameter_recovery_test.py

# Expected output (~/5 min):
# ✓ Training loss converges
# ✓ Recovery diagnostics plot: recovery_diagnostics.png
# ✓ Coverage ~95% on all parameters
# ✓ No systematic bias
```

If output looks good → Architecture validated ✅

---

## One-Sentence Summary

> **Your 8 cognitive parameters are fixed across TTAs; only the decision boundary adapts to time pressure. BayesFlow learns this via one conditional posterior network: p(θ|data,TTA).**

---

## Next Question?

- **"How does BayesFlow handle the conditioning?"** → TTA_CONDITIONING_GUIDE.md
- **"What's the full architecture?"** → BAYESFLOW_PIPELINE_DESIGN.md
- **"How do I load real data?"** → QUICKSTART_PARAMETER_RECOVERY.md (Section 3b)
- **"What if I have a different question?"** → DELIVERY_SUMMARY.md

Status: **Ready to validate and proceed to real data** ✅
