# 🔍 ARCHITECTURE AUDIT: Current vs. Specification

**Date:** February 13, 2026  
**Model:** DDM_DC Pedestrian Crossing  
**BayesFlow Version:** 2.0.8  

---

## Executive Summary

✅ **MOSTLY COMPLIANT** but with one **ARCHITECTURAL DIFFERENCE** in TTA handling.

| Component | Requirement | Current | Status |
|-----------|-----------|---------|--------|
| **Summary Network Type** | DeepSet/SetTransformer (permutation-invariant) | SetTransformer ✓ | **✅ PASS** |
| **Inference Network Type** | Conditional INN (cINN) | CouplingFlow ✓ | **✅ PASS** |
| **Trial Exchangeability** | Trials treated as set (order-independent) | `.as_set("x")` ✓ | **✅ PASS** |
| **TTA Input Strategy** | Trial-wise concatenation: `z_i = [RT_i, CPP_i, TTA]` | Global condition variable | **⚠️ DIFFERENT APPROACH** |
| **Data Shape to Summary** | (N_trials, 3) | (N_trials, 2) | **⚠️ MISSING TTA** |
| **Network Conditioning** | Via concatenated features | Via `condition_variables` | **⚠️ DIFFERENT APPROACH** |

---

## Detailed Audit

### ✅ Part 1: Summary Network - COMPLIANT

**Specification Requirement:**
```
Summary Network architecture (DeepSet):
1. Featurization (φ): Independent mapping of each trial vector
2. Pooling (Σ): Symmetric aggregation (sum/mean) across trials  
3. Regression (ρ): Final summary embedding for inference network
Requirements:
- Permutation invariant (treats trials as set, not sequence)
- Handles variable-length data
```

**Current Implementation:**
```python
summary_network = bf.networks.SetTransformer(summary_dim=10)
```

**Analysis:**
- ✅ SetTransformer is a permutation-invariant architecture (exactly the specification)
- ✅ Handles variable-length sets of trials naturally
- ✅ Uses attention-based aggregation (type of pooling operation)
- ✅ Outputs fixed-size embedding to inference network

**Verdict:** ✅ **FULLY COMPLIANT**

---

### ✅ Part 2: Inference Network - COMPLIANT

**Specification Requirement:**
```
Inference Network (cINN):
- Conditional Invertible Neural Network
- Maps parameter space ↔ base (Gaussian) distribution
- Conditioned on summary embedding
- Enables sampling via inverse transform
```

**Current Implementation:**
```python
inference_network = bf.networks.CouplingFlow()
```

**Analysis:**
- ✅ CouplingFlow is a well-known cINN architecture
- ✅ Implements affine coupling layers with learned transformations
- ✅ Maintains invertibility (can sample via inverse)
- ✅ BayesFlow 2.0.8 automatically conditions on summary embedding

**Verdict:** ✅ **FULLY COMPLIANT**

---

### ⚠️ Part 3: TTA Handling - ARCHITECTURAL DIFFERENCE

**Specification Requirement:**

```
Trial-Wise Concatenation Strategy
────────────────────────────────

Raw Observable per trial:   x_i = [RT_i, CPP_i]
Local context per trial:    c_i = [TTA_i]  (if this trial happened under TTA=3.0)
Augmented input per trial:  z_i = [RT_i, CPP_i, TTA_i]

Summary Network sees:
- Batch of N trials: {z_1, z_2, ..., z_N}
- Shape: (N, 3) where all 3 variables come in together
- Featurization learns conditional features on each trial
- Pooling aggregates across set of trials

Rationale:
"By seeing the triplet (RT, CPP, TTA) together for every trial, 
the network can learn conditional features such as 
'An RT of 1.0s is extremely fast for TTA=4.0s (implying high drift/low bound), 
but average for TTA=2.5s (implying collapsed bound).'"
```

**Current Implementation:**

```python
def adopt(p):
    adapter = (
        bf.Adapter()
        .broadcast("number_of_trials", to="x")
        .as_set("x")                                    # x = (60, 2) = [RT, CPP]
        .standardize("x", mean=0.0, std=1.0)
        .sqrt("number_of_trials")
        .convert_dtype("float64", "float32")
        .concatenate(list(p.keys()), into="inference_variables")
        .rename("x", "summary_variables")              # (60, 2) array
        .rename("tta_condition", "condition_variables")  # scalar or (1,) value
    )
    return adapter
```

**Data Flow in Current Implementation:**

```
Simulator output:
├── x: (60, 2) array = [(RT_1, CPP_1), ..., (RT_60, CPP_60)]
├── tta_condition: scalar = 3.0
└── Parameters: θ, b0, k, ...

After adapter:
├── summary_variables: (60, 2) float32 = [RT, CPP] pairs
├── condition_variables: scalar or (1,) = TTA value  ← SEPARATE
└── inference_variables: (8,) float32 = parameters to infer

Network behavior:
├── Summary Network INPUT: (60, 2) data ONLY
│   └── Does NOT see TTA in the trial data
├── Summary Network OUTPUT: embedding h
└── Inference Network: 
    ├── Conditions model on: h + condition_variables (TTA)
    └── Outputs p(θ | h, TTA) merger
```

**Analysis of Difference:**

| Aspect | Specification | Current | Impact |
|--------|--------------|---------|--------|
| TTA integration | **TRIAL-LEVEL**: Concatenated to each row | **GLOBAL**: Separate condition_variables | ✓ Functional but different |
| Input shape to Summary | (N, 3) with [RT, CPP, TTA] | (N, 2) with [RT, CPP] | ⚠️ Lost trial-level context |
| Feature learning | Learn p(RT, CPP \| TTA) jointly | Learn p(RT, CPP) then condition on TTA | ⚠️ Weaker feature extraction |
| Conditional dependency | Encoded in each trial vector | Encoded in inference network | ⚠️ Different flow |
| Permutation invariance | Still maintained (trials are set) | Still maintained (trials are set) | ✓ Preserved |

**Verdict:** ⚠️ **ARCHITECTURAL DIFFERENCE - NOT WRONG, BUT SUBOPTIMAL**

---

## Detailed Comparison

### Specification Approach: Trial-Wise Concatenation

```
HOW IT WORKS:
─────────────

Step 1: For each trial, create augmented vector
        Trial i with TTA=3.0: z_i = [RT_i, CPP_i, 3.0]
        Trial j with TTA=2.5: z_j = [RT_j, CPP_j, 2.5]
        (In mixed design, trials have different TTAs!)

Step 2: Summary Network featurization φ
        For each z_i independently:
        φ([RT_i, CPP_i, TTA_i]) → feature_i ∈ ℝ^128
        
        The network learns to extract features like:
        - "Fast RT + high TTA = high drift/threshold"
        - "Slow RT + low TTA = collapsed boundary"
        - These are CONDITIONAL features

Step 3: Pooling Σ (permutation-invariant)
        aggregate(feature_1, ..., feature_N) → h ∈ ℝ^d
        (sum or mean, order doesn't matter)

Step 4: Inference Network ρ
        Takes h (already has TTA info embedded)
        Outputs p(θ | h)

ADVANTAGE:
The Summary Network learns the CONDITIONAL STATISTICS directly.
It sees how RT and CPP vary WITH TTA in the same feature vector.
```

### Current Approach: Global Conditioning

```
HOW IT WORKS:
─────────────

Step 1: Summary Network sees pure data (no TTA)
        φ([RT_i, CPP_i]) → feature_i ∈ ℝ^128 (for all trials)
        Order doesn't matter (still permutation-invariant)

Step 2: Pooling Σ
        aggregate(feature_1, ..., feature_N) → h ∈ ℝ^d

Step 3: Inference Network THEN conditions on TTA
        p(θ | h, TTA_global)
        
        The network translates:
        "Given this summary h and GIVEN that TTA=3.0,
         what is the posterior on θ?"

TRADEOFF:
- Simpler implementation (TTA is global, just one value per batch)
- Summary network doesn't learn conditional features
- Inference network must learn TTA-parameter mappings
- Still works (conditioning is powerful), but indirect
```

---

## Technical Impact Analysis

### Scenario: Can the Network Learn "Temporal Velocity Compression"?

**The Problem:**
"RT of 0.8s is FAST under TTA=4.0 (→ high drift, low threshold)  
But RT of 0.8s is NORMAL under TTA=2.5 (→ boundary collapsed)"

Let's model how each approach learns this:

#### Specification Approach (Trial-Wise)

```
Summary Network featurization for Trial A (TTA=4.0, RT=0.8s):
Input: [0.8, CPP_value, 4.0]
  ↓
Hidden layers learn interactions: "RT=0.8 AND TTA=4.0 means FAST"
  ↓
feature_A = [high_drift_indicator, low_threshold_indicator, ...]

Summary Network featurization for Trial B (TTA=2.5, RT=0.8s):
Input: [0.8, CPP_value, 2.5]
  ↓
Hidden layers learn: "RT=0.8 AND TTA=2.5 means NORMAL"
  ↓
feature_B = [medium_drift_indicator, high_threshold_indicator, ...]

Result: Same RT value → DIFFERENT features learned
        because TTA was right there in the input!
        
Feature extraction directly embeds the conditional logic.
```

#### Current Approach (Global Conditioning)

```
Summary Network featurization (TTA invisible):
Trial A (TTA=4.0, RT=0.8s):
Input: [0.8, CPP_value]  ← NO TTA
  ↓
Hidden layers see only "RT=0.8 appears in this batch"
  ↓
feature_A depends on aggregation over ALL trials

Trial B (TTA=2.5, RT=0.8s):
Input: [0.8, CPP_value]  ← NO TTA
  ↓
Same input → similar features
  ↓
feature_B ≈ feature_A

Pooling step: aggregate all features (order-irrelevant)
  ↓
h = summary embedding (contains no explicit TTA info)

Inference Network then conditions:
p(θ | h, TTA_global)
  ↓
Network must LATER learn that same h paired with TTA=4.0
versus h paired with TTA=2.5 should give different posteriors

Result: TTA-conditional decoding happens AFTER pooling
        Less direct encoding of conditional structure
```

**Verdict on This Scenario:**
- ✅ Specification approach learns the conditional structure **during featurization**
- ✅ Current approach learns conditional structure **during inference network**
- ⚠️ Current approach is less direct but still potentially viable
- ⚠️ May require larger/more expressive inference network to compensate

---

## Identifiability & Parameter Recovery Implications

### Specification Approach (Trial-Wise): STRONGER SIGNAL

```
Example: Can we distinguish θ from σ_alpha?

θ = decision threshold (affects absolute RT scale)
σ_alpha = drift variability (affects RT spread)

Trial at TTA=4.0 with slow mean RT:
  With TTA visible: Network sees "slow RT for long TTA → low drift/threshold"
  → Strong signal for θ estimation
  
Trial at TTA=2.5 with same slow RT:
  With TTA visible: Network sees "slow RT for short TTA → collapsed boundary!"
  → Completely different inference about θ
  → These conditional patterns help disentangle parameters

Summary: Trial-level TTA view creates RICHER CONDITIONAL PATTERNS
         that help separate parameters with different TTA dependencies.
```

### Current Approach (Global): POTENTIALLY WEAKER SIGNAL

```
Without trial-level TTA cues:
  
  Summary network outputs h that averages over different TTAs
  Inference network must learn:
  "When TTA=4.0: these patterns in h → high θ"
  "When TTA=2.5: same patterns in h → lower θ"
  
  But h was computed WITHOUT seeing TTA,
  so it may lose discriminative power for TTA-dependent parameters.
  
Especially hard: Parameters like k (collapse rate)
  k directly parametrizes the boundary-TTA interaction
  Without seeing TTA in the summary stage,
  the inference network must reverse-engineer this entirely
```

---

## Recommendations

### Option 1: Keep Current Implementation (Faster to validate)
**If you want:** Quick parameter recovery validation
- ✅ Current code likely still works (conditioning is powerful)
- ✅ Can run parameter_recovery_test.py immediately
- ⚠️ May need longer training or larger networks
- ⚠️ Parameter recovery coverage might be slightly lower (e.g., 85% instead of 95%)
- ❌ Not following the recommended pipeline

**Action:** 
1. Run `parameter_recovery_test.py` as-is
2. Check coverage metrics (target >90%)
3. If <90% coverage on some parameters (especially k), consider Option 2

---

### Option 2: Implement Specification (Trial-Wise Concatenation)
**If you want:** Stronger identifiability & optimal architecture
- ✅ Follows your specification exactly
- ✅ Better parameter recovery (especially for k, θ)
- ✅ More direct conditional feature learning
- ❌ Requires rewriting the simulator & adapter

**Implementation Changes:**

1. **Modify simulator to track TTA per trial:**
```python
def ddm_DC_alphaToCpp(..., tta_condition):
    ...
    x_all = []
    tta_all = []  # NEW: track TTA for each trial
    
    for _ in range(number_of_trials):
        tta0 = tta_condition + np.random.uniform(0, 0.1)
        ...
        x_all.append([choicert, cpp])
        tta_all.append(tta0)  # NEW: record actual TTA with jitter
    
    return dict(
        x=np.stack(x_all),           # (60, 2)
        tta_per_trial=np.stack(tta_all)  # (60, 1) NEW!
    )
```

2. **Modify adapter to concatenate TTA to each trial:**
```python
def adopt(p):
    adapter = (
        bf.Adapter()
        .broadcast("number_of_trials", to="x")
        .broadcast("number_of_trials", to="tta_per_trial")  # NEW
        # Concatenate x and tta_per_trial BEFORE as_set
        .concatenate(["x", "tta_per_trial"], along_axis=1, into="x_augmented") # NEW
        .as_set("x_augmented")  # NOW operating on (60, 3) = [RT, CPP, TTA]
        .standardize("x_augmented", mean=0.0, std=1.0)
        # ... rest ...
        .rename("x_augmented", "summary_variables")  # Now (60, 3)
        # NO MORE condition_variables!
    )
```

3. **Remove global conditioning from inference network:**
```python
# Instead of:
inference_network = bf.networks.CouplingFlow()  # with condition_variables

# Use:
inference_network = bf.networks.CouplingFlow()  # no conditioning needed
# (Conditioning info is now in summary_variables)
```

**Estimated Effort:** 2-4 hours

---

## Summary Table: What to Check

Run parameter_recovery_test.py and examine:

```
CRITICAL METRICS for Current Implementation:
────────────────────────────────────────────

Parameter              | Expected | Current | Pass/Fail
─────────────────────────────────────────────────────
θ (threshold)         |   >90%   |    ?    |   ?
b0 (boundary)         |   >90%   |    ?    |   ?
k (collapse - HARD!)  |   >85%   |    ?    |   ? ← Most sensitive to TTA visibility
μ_ndt                 |   >90%   |    ?    |   ?
σ_ndt                 |   >90%   |    ?    |   ?
μ_α                   |   >90%   |    ?    |   ?
σ_α                   |   >90%   |    ?    |   ?
σ_cpp                 |   >90%   |    ?    |   ?

If k coverage < 85%:
  → Current approach is struggling
  → Consider Option 2 (trial-wise concatenation)

If all > 90%:
  → Current approach is sufficient
  → Global conditioning is powerful enough
```

---

## Theoretical Justification

**Why Trial-Wise is Specified:**

From your specification:
> "By seeing the triplet (RT, CPP, TTA) together for every trial, the network can learn conditional features such as "An RT of 1.0s is extremely fast for TTA=4.0s (implying high drift/low bound), but average for TTA=2.5s (implying collapsed bound)." This conditional feature extraction is precisely what allows the recovery of the kinematics-dependent parameters."

This is theoretically sound:
1. ✅ Collapsing boundaries are **kinematically coupled** to TTA
2. ✅ Parameters k, θ have **strong TTA dependencies** in decision dynamics
3. ✅ Trial-level TTA view creates **richer feature space** for Summary Network
4. ✅ Maintains **permutation invariance** (still a set of trials)
5. ✅ **Best practices** in causal representation learning recommend including context in observations

**Why Current Might Still Work:**

1. ✅ Conditioning in inference network is still powerful (proven in many applications)
2. ✅ CouplingFlow has enough capacity to learn complex conditional mappings
3. ✅ SetTransformer summary can learn aggregate statistics sufficient for inference
4. ⚠️ But less direct encoding of TTA-parameter relationships

---

## Verdict & Recommendation

| Implementation | Compliant? | Should use? | Estimated Recovery |
|---|---|---|---|
| **Current (Global Conditioning)** | ⚠️ Partially | ✅ Yes, for now (quick test) | 85-92% (depends on k) |
| **Specification (Trial-Wise)** | ✅ Fully | ✅ Yes, if recovery poor | 92-96% (optimal) |

**Recommended Action Plan:**

1. **NOW:** Run `parameter_recovery_test.py` with current code
2. **Review:** Check coverage on parameter k (most sensitive to TTA visibility)
3. **IF k coverage < 85%:** Implement Option 2 (trial-wise concatenation)
4. **IF all > 90%:** Current approach is acceptable; proceed to real data

---

## Next Steps

Would you like me to:

**A)** Create a modified version implementing trial-wise concatenation?  
**B)** Add detailed metrics to parameter_recovery_test.py to highlight k recovery?  
**C)** Run a quick theoretical analysis of parameter identifiability?  

Or proceed with validating current implementation first?
