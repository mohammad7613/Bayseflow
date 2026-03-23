# 🔄 Implementation Comparison: Global vs. Trial-Wise TTA

## Visual Architecture Comparison

### Current Implementation (Global Conditioning)

```
SIMULATOR OUTPUT:
┌─────────────────────────────────────────────────┐
│  x: (60, 2)                    [RT_1, CPP_1]   │
│                                [RT_2, CPP_2]   │
│                                    ...         │
│                                [RT_60, CPP_60] │
│                                               │
│  tta_condition: 3.0             ← Single TTA   │
│  (scalar, one value per batch)               │
└─────────────────────────────────────────────────┘
                    ↓
                (ADAPTER)
                    ↓
┌─────────────────────────────────────────────────┐
│  summary_variables: (60, 2)                     │
│    Data stream (No TTA info)                   │
│                                               │
│  condition_variables: 3.0                      │
│    Separate global conditioning               │
│                                               │
│  inference_variables: (8,)                     │
│    [θ, b0, k, μ_ndt, σ_ndt, μ_α, σ_α, σ_cpp] │
└─────────────────────────────────────────────────┘
           ↓                    ↓
    ┌─────────────────┐  ┌──────────────┐
    │  Summary Net    │  │ Condition    │
    │  (SetTransformer)   │ Value (TTA)  │
    │  INPUT: (60,2)  │  │              │
    │  OUTPUT: h ∈ℝ^d │  │              │
    └─────────────────┘  └──────────────┘
           │                    │
           └────────┬───────────┘
                    ↓
            ┌────────────────────┐
            │ Inference Network   │
            │ (CouplingFlow)      │
            │ INPUT: h, TTA       │
            │ OUTPUT: p(θ|h,TTA) │
            └────────────────────┘
                    ↓
            Sample from posterior
```

### Specification (Trial-Wise Concatenation)

```
SIMULATOR OUTPUT:
┌─────────────────────────────────────────────────┐
│  x: (60, 2)                    [RT_1, CPP_1]   │
│                                [RT_2, CPP_2]   │
│                                    ...         │
│                                [RT_60, CPP_60] │
│                                               │
│  tta_per_trial: (60, 1)         [3.05]         │
│    (one TTA value per trial)    [2.98]         │
│  (with jitter from simulator)       ...        │
│                                [3.07]         │
└─────────────────────────────────────────────────┘
                    ↓
                (ADAPTER)
                    ↓
        (CONCATENATE x + tta_per_trial)
                    ↓
┌─────────────────────────────────────────────────┐
│  summary_variables: (60, 3)                     │
│    TRIAL-WISE AUGMENTED DATA                   │
│    [RT_1, CPP_1, TTA_1]                        │
│    [RT_2, CPP_2, TTA_2]                        │
│         ...                                    │
│    [RT_60, CPP_60, TTA_60]                     │
│                                               │
│  inference_variables: (8,)                     │
│    [θ, b0, k, μ_ndt, σ_ndt, μ_α, σ_α, σ_cpp] │
└─────────────────────────────────────────────────┘
                    ↓
         ┌──────────────────────┐
         │  Summary Net         │
         │ (SetTransformer)     │
         │ INPUT: (60, 3)       │
         │   [RT, CPP, TTA]     │
         │   per trial          │
         │ OUTPUT: h ∈ℝ^d       │
         │ (already has TTA     │
         │  embedded in h)      │
         └──────────────────────┘
                    ↓
        ┌────────────────────────┐
        │ Inference Network       │
        │ (CouplingFlow)         │
        │ INPUT: h (only!)       │
        │ OUTPUT: p(θ|h)         │
        │ (TTA info already in h) │
        └────────────────────────┘
                    ↓
            Sample from posterior
```

---

## Key Differences Table

| Component | Global (Current) | Trial-Wise (Spec) |
|-----------|--------|------------|
| **Simulator Output** | `x:(60,2) + tta_condition:scalar` | `x:(60,2) + tta_per_trial:(60,1)` |
| **Preprocessing** | Standardize x only | Concatenate x + TTA → (60,3) |
| **Summary Input** | (60, 2) [RT, CPP] | (60, 3) [RT, CPP, TTA] |
| **Summary Output** | h (no TTA context) | h (TTA context embedded) |
| **Inference Input** | h + TTA (separate) | h (only, TTA inside) |
| **Feature Extraction** | Agnostic to TTA | TTA-aware interactions |
| **Permutation Invariance** | ✓ Yes (trials are set) | ✓ Yes (augmented trials are set) |
| **Flexibility** | Condition on ANY value at inference | Fixed to TTA values seen in training |
| **Complexity** | Simpler (fewer code changes) | Richer (conditional features) |

---

## Feature Learning Comparison

### Example: Learning "RT Interpretation Depends on TTA"

**Specification Approach (Trial-Wise):**
```python
# Input to featurization layer φ
Trial i: [RT=1.2s, CPP=−3.5, TTA=4.0s]
Trial j: [RT=1.2s, CPP=−3.2, TTA=2.5s]

# Featurization (learned by hidden layers)
Trial i features: "RT=1.2 with TTA=4.0 → HIGH drift, LOW threshold" 
                  → feature_i indicates "FAST decision"
                  
Trial j features: "RT=1.2 with TTA=2.5 → NORMAL drift, COLLAPSED boundary"
                  → feature_j indicates "AVERAGE decision"

# Despite same RT value, different feature representations!
# Network learned conditional interpretation
```

**Global Conditioning Approach (Current):**
```python
# Input to featurization layer φ
Trial i: [RT=1.2s, CPP=−3.5]
Trial j: [RT=1.2s, CPP=−3.2]

# Featurization (learned by hidden layers)
Trial i features: φ([1.2, -3.5]) → some_feature_vector
Trial j features: φ([1.2, -3.2]) → similar_feature_vector

# Same RT → similar features (no TTA context in featurization)
# Pooling: aggregate over all trials (order irrelevant)
# h = summary embedding (TTA-blind aggregate)

# Later, in inference network:
# "Given h and TTA=4.0, p(θ|h,4.0)"
# vs
# "Given h and TTA=2.5, p(θ|h,2.5)"
# Network must LEARN that same h → different θ depending on TTA
```

---

## Flow Comparison: How Parameters Get Inferred

### Specification (Trial-Wise)

```
Raw Behavior                Learning                    Inference
───────────────────────────────────────────────────────────────────

RT=0.8s under TTA=4.0 ──→ Summary Network:          At Inference:
  ├─ Input: [0.8, CPP, 4.0]   "This is FAST        p(θ | behavior, TTA=4.0)
  └─ Feature: "high_drift"     for this TTA"      └─ Network knows
     "low_threshold"                                 RT=0.8 is fast
                                                     → infers high drift
                                                     → infers low θ

RT=0.8s under TTA=2.5 ──→ Summary Network:          At Inference:
  ├─ Input: [0.8, CPP, 2.5]   "This is NORMAL    p(θ | behavior, TTA=2.5)
  └─ Feature: "normal_drift"   for this TTA"     └─ Network knows
     "high_threshold"                               RT=0.8 is normal
                                                    → infers normal drift
                                                    → infers high θ

KEY: Learning happens in Summary Network
     Each trial's TTA is visible during feature extraction
     → Conditional patterns learned DIRECTLY in hidden layers
     → Inference network receives TTA-aware summaries
```

### Current (Global Conditioning)

```
Raw Behavior                Learning                    Inference
───────────────────────────────────────────────────────────────────

RT=0.8s under TTA=4.0 ──→ Summary Network:          At Inference:
  ├─ Input: [0.8, CPP]        "I see RT=0.8"    p(θ | summary, TTA=4.0)
  └─ Feature: "medium"        (pooled over
     (no TTA context)          all TTAs)    └─ Inference Network MUST learn:
                                              "When TTA=4.0, this summary
RT=0.8s under TTA=2.5 ──→ Summary Network:     → high drift"
  ├─ Input: [0.8, CPP]        "I see RT=0.8"   "When TTA=2.5, same summary
  └─ Feature: "medium"        (same as above)  → normal drift"
     (no TTA context)                          → Must reverse-engineer
                                                coupling in inference net

KEY: Learning happens in Inference Network
     Summary provides TTA-blind aggregate
     → Inference network must learn conditional structure
     → More work for inference network, less information
     → Less direct encoding of kinematics
```

---

## The "Collapse Rate" Challenge: Why Trial-Wise Helps

Parameter **k** (boundary collapse rate) is particularly sensitive to TTA visibility:

### What k Does

```
Boundary at time t under TTA condition:
b(t) = b₀ / (1 + exp(-k(TTA - t - b₀/2)))

k controls HOW FAST the boundary collapses toward TTA
- High k: Boundary nearly intact until late (lots of time to accumulate)
- Low k: Boundary starts collapsing early (limited accumulation window)
```

### Trial-Wise Approach (Sees TTA Per Trial)

```
Summary Network featurization sees:
- Trial A: [RT=0.5s, CPP=−2.8, TTA=4.0s]
  "This person had 4.0s, but decided in 0.5s"
  → Network: "Boundary must have collapsed EARLY"
  → Feature: "high k" (fast collapse)

- Trial B: [RT=0.7s, CPP=−3.1, TTA=2.5s]
  "This person had 2.5s, decided in 0.7s (almost all available time)"
  → Network: "Used up most available time"
  → Feature: "low k" (slow collapse)"

Result: Same person, different RT values, network extracts
conditional interpretation of k by seeing TTA with each trial.
```

### Global Conditioning Approach (TTA Separate)

```
Summary Network sees:
- Trial A: [RT=0.5s, CPP=−2.8]  (no TTA!)
  Feature: "relatively_short"

- Trial B: [RT=0.7s, CPP=−3.1]  (no TTA!)
  Feature: "relatively_long"

Pooling: combine all features (order-irrelevant)
→ h = pooled summary (no explicit k information)

Inference Network LATER sees:
- "Given h and TTA=4.0, what's k?"
  Must learn: "Oh, if TTA is large and summary shows short RTs,
              then k must be high"
              
- "Given h and TTA=2.5, what's k?"
  Must learn: "If TTA is small and summary shows normal RTs,
              then k must be low"

Result: Inference network must learn the mathematical relationship
        between TTA, RTs, and k values AFTER seeing pooled summary.
        More difficult, less direct signal.
```

---

## When Each Approach Is Preferred

### Use **Global Conditioning** When:
- ✅ You want flexibility (inference on novel TTA values at test time)
- ✅ You want simplicity (easier code, fewer changes)
- ✅ You have lots of training data (network can learn indirect mappings)
- ✅ Parameter recovery test shows >90% coverage (current approach works)

### Use **Trial-Wise Concatenation** When:
- ✅ You want optimal information flow (TTA manifest in every trial)
- ✅ You have limited training data (every data point matters)
- ✅ Parameters are strongly coupled to TTA (like k, θ)
- ✅ You want cleaner theoretical architecture
- ✅ Parameter recovery shows <90% coverage on k or θ (current approach struggling)

---

## Summary: Should You Switch?

| Scenario | Recommendation |
|----------|---|
| "I want quick validation" | Stay with current, run parameter_recovery_test.py, check k_coverage |
| "k_coverage < 85%" | Switch to trial-wise (2-4 hrs work) |
| "All parameters ≥ 90%" | Current approach is fine, proceed |
| "I'm publishing this" | Use trial-wise (more principled) |
| "I'm just exploring" | Current is fine for now |

---

## Implementation Effort Comparison

### Current (Global Conditioning)
*Already implemented*
- Code: ✅ Done
- Lines changed: 0
- Commands: `python parameter_recovery_test.py`
- Time: ~10 minutes

### Trial-Wise Concatenation
- Code: Need modifications
- Lines changed: ~30-40
- Modifications: Simulator, adapter, feature dimension adjustments
- Time: ~2-4 hours

---

## Next Steps

**Recommend this sequence:**

1. ✅ Run parameter_recovery_test.py with CURRENT code
   - Time: 5-10 minutes
   - Goal: Get baseline metrics

2. 📊 Review results, especially k coverage
   - If k ≥ 85%: Current approach works
   - If k < 85%: Need trial-wise

3. 🔄 If needed, implement trial-wise concatenation
   - Use template in corrected_implementation.py (next)
   - Rerun parameter recovery with new code
   
4. 🚀 Proceed to real data with validated approach

---

Would you like me to create the corrected trial-wise implementation ready to use?
