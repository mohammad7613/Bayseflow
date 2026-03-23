# 📝 Code Comparison: Original vs Trial-Wise Adapters

## Side-by-Side: Line-by-Line Comparison

```python
╔════════════════════════════════════════╦════════════════════════════════════════╗
║     Original (DDM_DC_Pedestrain.py)    ║  Trial-Wise (DDM_DC_Pedestrain_TrialWise.py)
║                                        ║
║ def adopt(p):                          ║ def adopt_TrialWise(p):
║     adapter = (                        ║     adapter = (
║         bf.Adapter()                   ║         bf.Adapter()
╠════════════════════════════════════════╬════════════════════════════════════════╣
║ Line 1: Broadcast to x                 ║ Line 1: Broadcast to x
║ .broadcast(                            ║ .broadcast(
║     "number_of_trials",                ║     "number_of_trials",
║     to="x"                             ║     to="x"
║ )                                      ║ )
║                                        ║ • SAME
╠════════════════════════════════════════╬════════════════════════════════════════╣
║ (Nothing here)                         ║ Line 2: Broadcast to tta_per_trial ⭐
║                                        ║ .broadcast(
║                                        ║     "number_of_trials",
║                                        ║     to="tta_per_trial"
║                                        ║ ) ← NEW! Prepares for concatenation
╠════════════════════════════════════════╬════════════════════════════════════════╣
║ (Nothing here)                         ║ Line 3: Concatenate arrays ⭐⭐⭐ KEY!
║                                        ║ .concatenate(
║                                        ║     ["x", "tta_per_trial"],
║                                        ║     along_axis=1,      ← Horizontal stacking
║                                        ║     into="x_augmented"
║                                        ║ )
║                                        ║ • Input: (60,2) + (60,1)
║                                        ║ • Output: (60,3) [RT, CPP, TTA]
║                                        ║ • THIS is the specification requirement!
╠════════════════════════════════════════╬════════════════════════════════════════╣
║ Line 2: as_set("x")                    ║ Line 4: as_set("x_augmented")
║ .as_set("x")                           ║ .as_set("x_augmented")
║ • Treats (60,2) as exchangeable        ║ • Treats (60,3) as exchangeable
║ • [RT, CPP] per trial                  ║ • [RT, CPP, TTA] per trial
╠════════════════════════════════════════╬════════════════════════════════════════╣
║ Line 3: Standardize x                  ║ Line 5: Standardize x_augmented
║ .standardize(                          ║ .standardize(
║     "x",                               ║     "x_augmented",
║     mean=0.0, std=1.0                  ║     mean=0.0, std=1.0
║ )                                      ║ )
║ • Normalizes [RT, CPP]                 ║ • Normalizes [RT, CPP, TTA] together!
║ • TTA normalization: SKIP              ║ • TTA now gets treated as feature
╠════════════════════════════════════════╬════════════════════════════════════════╣
║ Line 4: sqrt                           ║ Line 6: sqrt
║ .sqrt("number_of_trials")              ║ .sqrt("number_of_trials")
║ • SAME                                 ║ • SAME
╠════════════════════════════════════════╬════════════════════════════════════════╣
║ Line 5: convert_dtype                  ║ Line 7: convert_dtype
║ .convert_dtype(                        ║ .convert_dtype(
║     "float64", "float32"               ║     "float64", "float32"
║ )                                      ║ )
║ • SAME                                 ║ • SAME
╠════════════════════════════════════════╬════════════════════════════════════════╣
║ Line 6: concatenate parameters         ║ Line 8: concatenate parameters
║ .concatenate(                          ║ .concatenate(
║     list(p.keys()),                    ║     list(p.keys()),
║     into="inference_variables"         ║     into="inference_variables"
║ )                                      ║ )
║ • Parameters: [θ, b0, k, ...]          ║ • SAME
║ • SAME                                 ║
╠════════════════════════════════════════╬════════════════════════════════════════╣
║ Line 7: rename x                       ║ Line 9: rename x_augmented
║ .rename("x",                           ║ .rename("x_augmented",
║         "summary_variables")           ║         "summary_variables")
║ • summary = (60, 2)                    ║ • summary = (60, 3) ⭐
║ • Contains: [RT, CPP]                  ║ • Contains: [RT, CPP, TTA]
╠════════════════════════════════════════╬════════════════════════════════════════╣
║ Line 8: rename tta                     ║ (Nothing here!)
║ .rename(                               ║
║     "tta_condition",                   ║ • NO separate condition_variables!
║     "condition_variables"              ║ • TTA is inside summary_variables
║ )                                      ║ • Networks see it as part of data
║ • condition = scalar 3.0                ║
║ • TTA kept SEPARATE ⭐                  ║
╚════════════════════════════════════════╩════════════════════════════════════════╝
```

---

## Network Input Format: The Critical Difference

### Original Adapter Output

```python
# After .adopt(prior_DC()):
{
    'summary_variables': np.array([
        [0.45, -2.3],   # Trial 1: [RT_norm, CPP_norm]
        [0.52, -2.8],   # Trial 2
        ...
        [0.41, -3.1],   # Trial 60
    ]),  # Shape: (60, 2)
    
    'condition_variables': np.array(3.0),  # ← TTA SEPARATE!
    
    'inference_variables': np.array([
        0.45,   # θ
        1.2,    # b0
        0.82,   # k
        0.35,   # μ_ndt
        0.08,   # σ_ndt
        0.55,   # μ_α
        0.12,   # σ_α
        0.15    # σ_cpp
    ]),  # Shape: (8,)
    
    'number_of_trials': 7.75
}
```

### Trial-Wise Adapter Output

```python
# After .adopt_TrialWise(prior_DC()):
{
    'summary_variables': np.array([
        [0.45, -2.3, 0.12],   # Trial 1: [RT_norm, CPP_norm, TTA_norm]
        [0.52, -2.8, 0.08],   # Trial 2
        ...
        [0.41, -3.1, 0.15],   # Trial 60
    ]),  # Shape: (60, 3)  ← NOW 3D with TTA!
    
    # NO 'condition_variables'!
    
    'inference_variables': np.array([
        0.45, 1.2, 0.82, 0.35, 0.08, 0.55, 0.12, 0.15
    ]),  # Shape: (8,)
    
    'number_of_trials': 7.75
}
```

---

## How Networks Use These Inputs

### Original Pipeline

```
SetTransformer Summary Network:
  Input: summary_variables (60, 2)
         Each row = [RT_norm, CPP_norm]
  
  What it learns:
    "These are reaction times and CPP values"
    "They have some pattern or distribution"
  
  What it CANNOT learn:
    ❌ Which TTA produced these times
    ❌ How TTA affects RT-CPP relationship
    ❌ Conditional feature extraction
  
  Output: h shape (hidden_dim,)
          h = summary of 60 trials (but TTA-blind)

         ↓ ↓ ↓

CouplingFlow Inference Network:
  Input: h (TTA-blind) + condition_variables (TTA=3.0)
  
  Task: Learn p(θ | h, TTA)
  
  Architecture problem:
    ⚠️  Must learn TTA's effect AFTER pooling
    ⚠️  TTA information arrives too late
    ⚠️  Summary network can't use TTA to guide feature extraction


Flow Diagram:
┌──────────────────────────┐
│ (RT, CPP) per trial      │
└────────────┬─────────────┘
             │
      summary_network  ← TTA is NOT here!
             │
         h (60→1)  ← Lost TTA context
             │
      ─────────────────────────
      Add TTA here: 3.0  ← Too late, h is fixed
      ─────────────────────────
             │
      inference_network
             │
         p(θ | h, TTA)  ← Posterior
```

### Trial-Wise Pipeline

```
SetTransformer Summary Network:
  Input: summary_variables (60, 3)
         Each row = [RT_norm, CPP_norm, TTA_norm]
  
  What it learns:
    ✅ "These RTs and CPPs occurred under different TTAs"
    ✅ "When TTA=3.0, RTs tend to be longer"
    ✅ "When TTA=4.0, CPP values change"
    ✅ Conditional feature extraction!
  
  Output: h shape (hidden_dim,)
          h = conditional summary (TTA-AWARE)

         ↓ ↓ ↓

CouplingFlow Inference Network:
  Input: h (already TTA-aware)
  
  Task: Learn p(θ | h)
  
  Advantage:
    ✅ Summary network already captured conditional info
    ✅ Inference network gets rich context
    ✅ More efficient learning


Flow Diagram:
┌──────────────────────────┐
│ (RT, CPP, TTA) per trial │
└────────────┬─────────────┘
             │
      summary_network  ← TTA is HERE!
             │  (learns conditional features)
         h (60→1)  ← TTA context preserved!
             │
           (no TTA addition)
             │
      inference_network
             │
         p(θ | h)  ← Rich posterior
```

---

## Why The 8 Operations Are Ordered This Way

### Critical Sequencing Rules

```
RULE 1: Broadcast must come FIRST
─────────────────────────────────
✅ .broadcast("number_of_trials", to="x")
   .broadcast("number_of_trials", to="tta_per_trial")
   .concatenate(...)  ← Now x and tta are both "broadcast"

❌ Can't do:
   .concatenate(["x", "tta_per_trial"])  ← Before broadcast!
   This would fail because broadcast alignment not done.


RULE 2: Concatenate before as_set
─────────────────────────────────
✅ .concatenate(["x", "tta"], into="x_aug")
   .as_set("x_aug")  ← Mark the COMBINED array as set

❌ Can't do:
   .as_set("x")
   .as_set("tta")
   .concatenate(["x", "tta"])  ← as_set must be on result!


RULE 3: Standardization applies to what you're standardizing
──────────────────────────────────────────────────────────────
If you want TTA standardized → standardize x_augmented (WITH TTA)
If you want TTA separate → standardize only x (no TTA)


RULE 4: Rename at the END
─────────────────────────
✅ (process array)
   .rename("internal_name", "network_name")

❌ Can't do:
   .rename("x", "summary_variables")
   .concatenate(["summary_variables", "tta"])  ← Uses renamed key!


RULE 5: Parameters concatenation (inference_variables) is SEPARATE
───────────────────────────────────────────────────────────────────
.concatenate(["x", "tta"], ...)  ← Data concatenation
.concatenate(list(p.keys()), ...)  ← Parameter concatenation
                                     Both happen but DIFFERENT arrays
```

---

## What the `.concatenate()` Operation Actually Does

### Syntax

```python
.concatenate(
    ["array1", "array2"],      # Which arrays to combine
    along_axis=1,              # Axis direction: 0=vertical, 1=horizontal
    into="combined_result"     # Output name
)
```

### Visualization

```
along_axis=0 (stack vertically):
───────────────────────────────
array1: (3, 2)          array2: (2, 2)          Result: (5, 2)
[1, 2]                  [7, 8]                  [1, 2]
[3, 4]                  [9, 10]     →           [3, 4]
[5, 6]                                          [5, 6]
                                                [7, 8]
                                                [9, 10]


along_axis=1 (stack horizontally):
───────────────────────────────────
array1: (60, 2)         array2: (60, 1)         Result: (60, 3)
[RT1, CPP1]             [TTA1]                  [RT1, CPP1, TTA1]
[RT2, CPP2]             [TTA2]          →       [RT2, CPP2, TTA2]
[RT3, CPP3]             [TTA3]                  [RT3, CPP3, TTA3]
...                     ...                     ...
[RT60, CPP60]           [TTA60]                 [RT60, CPP60, TTA60]


In our code:
.concatenate(
    ["x", "tta_per_trial"],
    along_axis=1,              ← Horizontal!
    into="x_augmented"         ← Creates (60, 3)
)
```

---

## Decision Tree: Which Adapter to Use?

```
Do you want TTA visible to Summary Network?
│
├─ YES (Specification says YES)
│  ├─ Condition: Include TTA in its own trials
│  ├─ Use: adopt_TrialWise()
│  ├─ Format: summary_variables (60, 3) [RT, CPP, TTA]
│  ├─ Pros: ✅ Conditional features ✅ Better k recovery
│  └─ Cons: ⚠️  More complex

└─ NO (Current implementation)
   ├─ Condition: Keep TTA separate
   ├─ Use: adopt()
   ├─ Format: summary_variables (60, 2) [RT, CPP]
   │          condition_variables (scalar) TTA
   ├─ Pros: ✅ Simpler ✅ Flexible
   └─ Cons: ⚠️  Indirect conditioning ⚠️  Worse k recovery


Alternative (if you want both benefits):
│
└─ Keep separate but pass to Summary
   ├─ Use: adopt_TrialWise_Alternative()
   ├─ Format: summary_variables (60, 2) [RT, CPP]
   │          context_variables (60, 1) [TTA per trial]
   ├─ Pros: ✅ TTA visible to summary ✅ More flexible
   └─ Cons: ⚠️  Even more complex
```

---

## Summary: The One Critical Operation

The **ONE critical difference** between the adapters is this line:

```python
# ORIGINAL: Nothing here
#           x and tta_condition stay separate

# TRIAL-WISE: THIS LINE
.concatenate(
    ["x", "tta_per_trial"],
    along_axis=1,
    into="x_augmented"
)
```

**This single operation**:
- Changes data from (60, 2) → (60, 3)
- Embeds TTA within trial data
- Enables Summary Network to learn conditional features
- Improves parameter recovery (especially k)

Everything else just follows from this concatenation!

