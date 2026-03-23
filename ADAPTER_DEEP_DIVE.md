# 📊 Deep Dive: Adapter (adopt) Operations Explained

## What Is An Adapter?

An adapter in BayesFlow is a **data transformation pipeline** that takes raw simulator output and converts it into the exact format required by the neural networks.

```
Raw Simulator Output                Adapter Pipeline               Network-Ready Format
─────────────────────               ─────────────────               ──────────────────
{                                   .broadcast()                   {
  'x': (60, 2),                     .as_set()                        'summary_variables': ...,
  'tta_condition': 3.0,             .standardize()                   'inference_variables': ...,
  'theta': 0.45,                    .concatenate()                   'condition_variables': ...
  'b0': 1.2,                        .rename()                      }
  ...
}                                   
                    ↓               ↓                           ↓
                    Messy           Organized                   Clean
                    Raw             Transform                   Final
```

---

## Original Adapter (Current Implementation)

**File:** `DDM_DC_Pedestrain.py`

```python
def adopt(p):
    """
    Original: GLOBAL CONDITIONING approach
    TTA is kept separate from trial data
    """
    adapter = (
        bf.Adapter()
        .broadcast("number_of_trials", to="x")
        .as_set("x")
        .standardize("x", mean=0.0, std=1.0)
        .sqrt("number_of_trials")
        .convert_dtype("float64", "float32")
        .concatenate(list(p.keys()), into="inference_variables")
        .rename("x", "summary_variables")
        .rename("tta_condition", "condition_variables")
    )
    return adapter
```

### Step-by-Step Breakdown (Original)

```
STEP 1: .broadcast("number_of_trials", to="x")
────────────────────────────────────────────────
Input dict:
{
  'x': (60, 2) array,
  'number_of_trials': 60  ← This value
  'tta_condition': 3.0,
  ...
}

What it does:
- Takes the value 60 from 'number_of_trials'
- Makes it available to following operations
- Doesn't change the data, just marks it as accessible

Output:
Same as input, but 'number_of_trials' is now "broadcast" (ready to use)


STEP 2: .as_set("x")
──────────────────────
Input:
{
  'x': (60, 2) array
}

What it does:
- Marks x as an UNORDERED SET of trials
- Tells networks: "These 60 trials have no special order"
- Enables flexibility: network can handle variable # of trials

Output:
{
  'x': (60, 2) still, but now flagged as "set"
}


STEP 3: .standardize("x", mean=0.0, std=1.0)
──────────────────────────────────────────────
Input:
{
  'x': (60, 2) with raw values
       Maybe RT ranges [0.2, 3.0]
       Maybe CPP ranges [-5.0, 0.5]
}

What it does:
- Normalizes each column to have mean=0, std=1
- Converts raw values to z-scores
- Important for neural network training (stable gradients)

Output:
{
  'x': (60, 2) standardized
       RT: mean≈0, std≈1
       CPP: mean≈0, std≈1
}


STEP 4: .sqrt("number_of_trials")
──────────────────────────────────
Input:
{
  'number_of_trials': 60
}

What it does:
- Computes √60 ≈ 7.75
- Creates a feature engineering value
- Provides scale information to networks

Output:
{
  'number_of_trials': 7.75 (engineered feature)
}


STEP 5: .convert_dtype("float64", "float32")
──────────────────────────────────────────────
Input:
All arrays in float64 (Python default)

What it does:
- Converts to float32 (PyTorch expects this)
- Reduces memory usage
- Matches GPU expectations

Output:
{
  'x': (60, 2) float32,
  'number_of_trials': 7.75 float32,
  'theta': 0.45 float32,
  ...
}


STEP 6: .concatenate(list(p.keys()), into="inference_variables")
──────────────────────────────────────────────────────────────────
Input:
{
  'theta': 0.45,
  'b0': 1.2,
  'k': 0.82,
  'mu_ndt': 0.35,
  'sigma_ndt': 0.08,
  'mu_alpah': 0.55,
  'sigma_alpha': 0.12,
  'sigma_cpp': 0.15,
  ... (other keys)
}

What it does:
- Takes all 8 parameters listed in p.keys()
- Stacks them into ONE vector
- Creates inference_variables = [θ, b0, k, μ_ndt, σ_ndt, μ_α, σ_α, σ_cpp]

Output:
{
  'inference_variables': (8,) vector = [0.45, 1.2, 0.82, ...]
}


STEP 7: .rename("x", "summary_variables")
───────────────────────────────────────────
Input:
{
  'x': (60, 2) standardized
}

What it does:
- Changes key name from 'x' to 'summary_variables'
- Makes it clear this is what networks learn from

Output:
{
  'summary_variables': (60, 2)
}


STEP 8: .rename("tta_condition", "condition_variables")
─────────────────────────────────────────────────────────
Input:
{
  'tta_condition': 3.0
}

What it does:
- Renames to 'condition_variables'
- Tells inference network: "This is what I condition on"

Output:
{
  'condition_variables': 3.0 (scalar)
}


FINAL OUTPUT:
─────────────
{
  'summary_variables': (60, 2) float32,      ← Trial data [RT, CPP]
  'inference_variables': (8,) float32,       ← Parameters
  'condition_variables': 3.0 float32,        ← TTA (SEPARATE)
  'number_of_trials': 7.75 float32
}
```

---

## Trial-Wise Adapter (Primary Version)

**File:** `DDM_DC_Pedestrain_TrialWise.py`

```python
def adopt_TrialWise(p):
    """
    TRIAL-WISE CONCATENATION approach
    TTA is concatenated WITH trial data
    """
    adapter = (
        bf.Adapter()
        .broadcast("number_of_trials", to="x")
        .broadcast("number_of_trials", to="tta_per_trial")  # NEW!
        .concatenate(
            ["x", "tta_per_trial"], 
            along_axis=1,
            into="x_augmented"  # NEW!
        )
        .as_set("x_augmented")  # CHANGED: Now operates on (60, 3)
        .standardize("x_augmented", mean=0.0, std=1.0)  # CHANGED
        .sqrt("number_of_trials")
        .convert_dtype("float64", "float32")
        .concatenate(list(p.keys()), into="inference_variables")
        .rename("x_augmented", "summary_variables")  # CHANGED
        # NO .rename("tta_condition", "condition_variables")
    )
    return adapter
```

### Step-by-Step Breakdown (Trial-Wise Primary)

```
STEP 1-2: .broadcast() for both x and tta_per_trial
──────────────────────────────────────────────────────
Input:
{
  'x': (60, 2) = [RT_i, CPP_i],
  'tta_per_trial': (60, 1) = [TTA_i],  ← NEW: per-trial TTA values
  'number_of_trials': 60,
  ...
}

What it does:
- Makes number_of_trials available to BOTH x and tta_per_trial
- Ensures they're aligned (both length 60)

Output:
Same, with both x and tta_per_trial broadcast-ready


STEP 3: .concatenate(["x", "tta_per_trial"], along_axis=1, into="x_augmented")
───────────────────────────────────────────────────────────────────────────────
THIS IS THE KEY DIFFERENCE! ← Specification requirement

Input:
{
  'x': (60, 2) with shape [RT, CPP],
       [[0.45, -2.3],
        [0.52, -2.8],
        ...
        [0.41, -3.1]]
  
  'tta_per_trial': (60, 1) with shape [TTA],
       [[3.05],
        [2.98],
        ...
        [3.07]]
}

What it does:
- Takes two (60, N) arrays
- Stacks them HORIZONTALLY (along_axis=1)
- Creates NEW form: [RT, CPP, TTA] per trial

Visualization:
     x                    tta_per_trial              x_augmented
┌──────────────┐        ┌──────────┐              ┌──────────────────┐
│RT_1  CPP_1   │        │TTA_1     │              │RT_1 CPP_1 TTA_1  │
│RT_2  CPP_2   │   +    │TTA_2     │    ═════→    │RT_2 CPP_2 TTA_2  │
│...   ...     │        │...       │              │...  ...   ...    │
│RT_60 CPP_60  │        │TTA_60    │              │RT_60 CPP_60 TTA_60│
└──────────────┘        └──────────┘              └──────────────────┘
   (60, 2)                (60, 1)                      (60, 3)  ← AUGMENTED!

Output:
{
  'x_augmented': (60, 3) with [RT, CPP, TTA] per row
}


STEP 4: .as_set("x_augmented")
───────────────────────────────
Input:
{
  'x_augmented': (60, 3)  ← Now we have 3 features per trial
}

What it does:
- Marks x_augmented as unordered set
- Still preserves exchangeability (order doesn't matter)
- But now each "item in set" is a 3D vector [RT, CPP, TTA]

Output:
{
  'x_augmented': (60, 3) flagged as set
}


STEP 5: .standardize("x_augmented", mean=0.0, std=1.0)
──────────────────────────────────────────────────────
Input:
{
  'x_augmented': (60, 3) raw values
       RT: ranges [0.2, 3.0]
       CPP: ranges [-5.0, 0.5]
       TTA: ranges [2.5, 4.0]  ← Now included!
}

What it does:
- Normalizes ALL THREE variables together
- Each column (RT, CPP, TTA) becomes standardized independently
- Important: TTA now gets normalized like any feature

Output:
{
  'x_augmented': (60, 3) standardized
       RT_norm: mean≈0, std≈1
       CPP_norm: mean≈0, std≈1
       TTA_norm: mean≈0, std≈1  ← Treated as feature
}


STEPS 6-7: Feature engineering & type conversion
─────────────────────────────────────────────────
Same as original:
- .sqrt("number_of_trials")
- .convert_dtype("float64", "float32")


STEP 8: .concatenate(..., into="inference_variables")
──────────────────────────────────────────────────────
Same as original:
- Stack all 8 parameters into one vector


STEP 9: .rename("x_augmented", "summary_variables")
─────────────────────────────────────────────────────
Input:
{
  'x_augmented': (60, 3)  ← Contains [RT, CPP, TTA]
}

What it does:
- Renames to 'summary_variables'
- NOW summary_variables HAS TTA INSIDE IT!

Output:
{
  'summary_variables': (60, 3)  ← [RT_norm, CPP_norm, TTA_norm]
}


FINAL OUTPUT (Trial-Wise):
──────────────────────────
{
  'summary_variables': (60, 3) float32,      ← [RT, CPP, TTA] PER TRIAL
  'inference_variables': (8,) float32,       ← Parameters
  # NO 'condition_variables' ← TTA is inside summary!
  'number_of_trials': 7.75 float32
}
```

---

## Side-by-Side Comparison: All 8 Operations

```
┌──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
│ Operation                    │ Original (Current)           │ Trial-Wise (Spec)            │
├──────────────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ 1. broadcast number_of_trials│ .broadcast(...to="x")        │ Same                         │
│    to x                      │                              │                              │
├──────────────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ 2. broadcast to tta_per_trial│ (doesn't exist)              │ NEW!                         │
│                              │                              │ .broadcast(...to=            │
│                              │                              │   "tta_per_trial")           │
├──────────────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ 3. Concatenate               │ (no concatenation)           │ NEW! CRITICAL!              │
│    [x, tta_per_trial]        │ x and tta stay separate      │ .concatenate(["x",          │
│                              │                              │   "tta_per_trial"],         │
│                              │                              │   along_axis=1,             │
│                              │                              │   into="x_augmented")       │
├──────────────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ 4. as_set()                  │ .as_set("x")                 │ .as_set("x_augmented")       │
│    (which array?)            │ Operates on (60, 2)          │ Operates on (60, 3)         │
├──────────────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ 5. standardize               │ .standardize("x", ...)       │ .standardize("x_augmented", │
│    (standardize what?)       │ Only standardizes [RT, CPP]  │ Standardizes [RT,CPP,TTA]   │
├──────────────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ 6. sqrt                      │ Same                         │ Same                         │
├──────────────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ 7. convert_dtype             │ Same                         │ Same                         │
├──────────────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ 8. concatenate (parameters)  │ Same                         │ Same                         │
├──────────────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ 9. rename x                  │ .rename("x",                 │ .rename("x_augmented",       │
│                              │   "summary_variables")       │   "summary_variables")       │
│                              │ summary = (60, 2)            │ summary = (60, 3)           │
├──────────────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ 10. rename tta               │ .rename("tta_condition",     │ (no separate rename)        │
│                              │   "condition_variables")     │ TTA already in summary      │
│                              │ condition = scalar           │                              │
└──────────────────────────────┴──────────────────────────────┴──────────────────────────────┘
```

---

## Final Output Format Comparison

### Original (Current)

```python
network_input = {
    'summary_variables': (60, 2),   # [RT, CPP] - TTA-BLIND
    'condition_variables': 3.0,      # TTA - SEPARATE
    'inference_variables': (8,)      # Parameters
}

# What networks see:
Summary Network:
  INPUT: (60, 2) data [RT, CPP]
  └─ Doesn't know this came from TTA=3.0
  OUTPUT: h (TTA-blind embedding)

Inference Network:
  INPUT: h (TTA-blind) + 3.0 (TTA)
  └─ Must learn: "Given h and TTA=3.0, what's the posterior?"
  OUTPUT: p(θ | h, TTA=3.0)
```

### Trial-Wise (Specification)

```python
network_input = {
    'summary_variables': (60, 3),   # [RT, CPP, TTA] - TTA-AWARE!
    'inference_variables': (8,)      # Parameters
    # No condition_variables!
}

# What networks see:
Summary Network:
  INPUT: (60, 3) data [RT, CPP, TTA]
  └─ Sees TTA for each individual trial
  OUTPUT: h (TTA-AWARE embedding)

Inference Network:
  INPUT: h (already TTA-aware from summary)
  └─ Just learns: p(θ | h)
  OUTPUT: p(θ | h)
```

---

## Alternative Version (Also in TrialWise File)

```python
def adopt_TrialWise_Alternative(p):
    """
    ALTERNATIVE: Separate but both go to summary
    """
    adapter = (
        bf.Adapter()
        .broadcast("number_of_trials", to="x")
        .broadcast("number_of_trials", to="tta_per_trial")
        
        .standardize("x", mean=0.0, std=1.0)
        .as_set("x")
        .as_set("tta_per_trial")  # BOTH are sets!
        
        .sqrt("number_of_trials")
        .convert_dtype("float64", "float32")
        
        .concatenate(list(p.keys()), into="inference_variables")
        
        .rename("x", "summary_variables")
        .rename("tta_per_trial", "context_variables")  # Different key!
    )
    return adapter
```

### How Alternative Differs

```
ALTERNATIVE final output:
{
  'summary_variables': (60, 2) [RT, CPP],
  'context_variables': (60, 1) [TTA_per_trial],  ← Different!
  'inference_variables': (8,)
}

Instead of concatenating into one array,
this keeps TTA separate BUT still passes it to networks
as a second input (context_variables, not a single condition)

When to use:
- If you want TTA as optional/flexible
- If you might want TTA-conditioning on/off
- More flexible but more complex
```

---

## Why Trial-Wise Primary Is Better

### Original: Information Flow

```
Raw data (includes TTA)
       ↓
Simulator returns x + tta_condition (separate)
       ↓
Adapter takes tta_condition OUT of trial data
       ↓
Summary Network: sees only [RT, CPP], TTA hidden
       ↓
Featurization: "Learn RT-CPP patterns" (TTA-blind)
       ↓
Pooling: Combine into h (still TTA-blind)
       ↓
Inference Network: "Here's h and TTA=3.0, infer θ"
       ↓
Learning happens: Map (h, TTA) → θ (INDIRECT)
```

### Trial-Wise: Information Flow

```
Raw data (includes TTA)
       ↓
Simulator returns x + tta_per_trial
       ↓
Adapter CONCATENATES them: [RT, CPP, TTA] ← KEPT TOGETHER
       ↓
Summary Network: sees [RT, CPP, TTA]
       ↓
Featurization: "Learn conditional RT-CPP-TTA patterns"
       ↓
Pooling: Combine into h (TTA context preserved)
       ↓
Inference Network: "Here's h" (already TTA-aware)
       ↓
Learning happens: directly in summary via conditional features (DIRECT)
```

---

## Summary Table: Operations Explained

| Operation | Purpose | Input | Output |
|-----------|---------|-------|--------|
| `broadcast()` | Make value available to pipeline | scalar value | value (tagged) |
| `concatenate()` | Combine arrays horizontally | [array1, array2] | combined_array |
| `as_set()` | Mark as unordered (exchangeable) | array | array (flagged as set) |
| `standardize()` | Normalize to mean=0, std=1 | raw data | normalized data |
| `sqrt()` | Feature engineering | scalar | sqrt(scalar) |
| `convert_dtype()` | Type conversion | float64 | float32 |
| `rename()` | Change key name | {old_key: value} | {new_key: value} |

---

This explanation should clarify exactly what the adapter does and why trial-wise concatenation is the critical difference!

