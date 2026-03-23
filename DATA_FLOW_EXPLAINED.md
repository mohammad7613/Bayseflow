# 🔄 Data Flow: How meta() + simulator() + adopt() Connect

## Quick Answer

**YES** — The output of `meta()` and `simulator()` are **automatically merged by BayesFlow** before being passed to `adopt()`.

Here's the exact sequence:

```
1. BayesFlow calls prior() → generates parameters
   └─ Output: {'theta': 0.45, 'b0': 1.2, 'k': 0.82, ...}

2. BayesFlow calls meta() → generates context
   └─ Output: {'number_of_trials': 60, 'tta_condition': 3.0}

3. BayesFlow MERGES steps 1 & 2
   └─ Output: {
        'theta': 0.45, 'b0': 1.2, 'k': 0.82, ...,  ← from prior
        'number_of_trials': 60, 'tta_condition': 3.0   ← from meta
      }

4. BayesFlow calls simulator(merged_dict) → generates observations
   └─ Input: merged dict with BOTH parameters and context
   └─ Output: {'x': (60, 2) array, [optionally: 'tta_per_trial']}

5. BayesFlow MERGES steps 3 & 4
   └─ Output: {
        'theta': 0.45, 'b0': 1.2, 'k': 0.82, ...,     ← parameters
        'number_of_trials': 60, 'tta_condition': 3.0,  ← context
        'x': (60, 2) array,                            ← observations
        [optional: 'tta_per_trial': (60, 1) array]
      }

6. adopt() transforms this merged dict → network input
   └─ Input: merged dict from step 5
   └─ Output: {
        'summary_variables': (60, 2) or (60, 3),
        'condition_variables': 3.0,  [or omitted]
        'inference_variables': (8,),
        ...
      }

7. Networks receive final transformed dict
   └─ Ready for training!
```

---

## How BayesFlow's `make_simulator()` Works

```python
# Original code:
model_DC = bf.simulators.make_simulator(
    [prior_DC, ddm_DC_alphaToCpp],  ← Functions list!
    meta_fn=meta                      ← Separate context function
)

all_models = {'model_DC': [model_DC, adopt(prior_DC())]}
```

### The Key: `make_simulator([func1, func2, func3, ...])`

```
Functions in the list are called IN ORDER:
─────────────────────────────────────────

list = [prior_DC, ddm_DC_alphaToCpp]
                ↓
         1️⃣ prior_DC()  called first
         2️⃣ result merged
         3️⃣ meta() called (separate)
         4️⃣ result merged
         5️⃣ ddm_DC_alphaToCpp() called with MERGED results
         6️⃣ final result returned


In detail:

Step 1: Call prior_DC()
────────────────────
BayesFlow: prior_DC()
Result: {
  'theta': 0.45,
  'b0': 1.2,
  'k': 0.82,
  'mu_ndt': 0.35,
  'sigma_ndt': 0.08,
  'mu_alpah': 0.55,
  'sigma_alpha': 0.12,
  'sigma_cpp': 0.15,
}


Step 2: Call meta_fn=meta()
──────────────────────────
BayesFlow: meta()
Result: {
  'number_of_trials': 60,
  'tta_condition': 3.0,
}


Step 3: MERGE prior + meta
─────────────────────────
BayesFlow automatically combines (like dict.update()):
𝐌𝐞𝐫𝐠𝐞𝐝 = {
  'theta': 0.45,           ← from prior
  'b0': 1.2,               ← from prior
  'k': 0.82,               ← from prior
  'mu_ndt': 0.35,          ← from prior
  'sigma_ndt': 0.08,       ← from prior
  'mu_alpah': 0.55,        ← from prior
  'sigma_alpha': 0.12,     ← from prior
  'sigma_cpp': 0.15,       ← from prior
  'number_of_trials': 60,  ← from meta
  'tta_condition': 3.0,    ← from meta
}


Step 4: Call simulator(merged)
──────────────────────────────
BayesFlow: ddm_DC_alphaToCpp(**merged)

def ddm_DC_alphaToCpp(
    theta=0.45,              ← unpacked from merged
    b0=1.2,                  ← unpacked from merged
    k=0.82,                  ← unpacked from merged
    mu_ndt=0.35,             ← unpacked from merged
    sigma_ndt=0.08,          ← unpacked from merged
    mu_alpah=0.55,           ← unpacked from merged
    sigma_alpha=0.12,        ← unpacked from merged
    sigma_cpp=0.15,          ← unpacked from merged
    number_of_trials=60,     ← unpacked from meta
    tta_condition=3.0,       ← unpacked from meta
    dt=0.005
):
    # Generates observations using these values
    return dict(x=x)  # or dict(x=x, tta_per_trial=...)


Step 5: MERGE prior + meta + simulator
────────────────────────────────────────
BayesFlow combines all outputs:
𝐅𝐢𝐧𝐚𝐥 = {
  'theta': 0.45,                        ← from prior
  'b0': 1.2,                            ← from prior
  'k': 0.82,                            ← from prior
  'mu_ndt': 0.35,                       ← from prior
  'sigma_ndt': 0.08,                    ← from prior
  'mu_alpah': 0.55,                     ← from prior
  'sigma_alpha': 0.12,                  ← from prior
  'sigma_cpp': 0.15,                    ← from prior
  'number_of_trials': 60,               ← from meta
  'tta_condition': 3.0,                 ← from meta
  'x': np.array([...]),                 ← from simulator (60, 2)
  ['tta_per_trial': np.array([...])]    ← from simulator (60, 1) if added
}


Step 6: Pass to adopt()
──────────────────────
BayesFlow: adopt(prior_DC())(final_dict)

Adapter has access to ALL keys:
✅ Can access 'x'
✅ Can access 'tta_condition'
✅ Can access 'number_of_trials'
✅ Can access all parameter keys (theta, b0, k, ...)
✅ Can access 'tta_per_trial' (if simulator returns it)


Step 7: Output to networks
──────────────────────────
Adapter returns:
𝐍𝐞𝐭𝐰𝐨𝐫𝐤_𝐈𝐧𝐩𝐮𝐭 = {
  'summary_variables': ...,        ← network sees this
  'condition_variables': ...,      ← network sees this
  'inference_variables': ...,      ← network sees this
  'number_of_trials': ...          ← network sees this
}
```

---

## Visual: The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BayesFlow make_simulator() Pipeline              │
└─────────────────────────────────────────────────────────────────────┘

                            ╔════════════════════════════════════════╗
                            ║   SETUP PHASE (Happens Once)           ║
                            ║                                        ║
                            ║ model = make_simulator(                ║
                            ║   [prior_DC,                           ║
                            ║    ddm_DC_alphaToCpp],                 ║
                            ║   meta_fn=meta                         ║
                            ║ )                                      ║
                            ║                                        ║
                            ║ all_models = {'model_DC':              ║
                            ║   [model_DC,                           ║
                            ║    adopt(prior_DC())]                  ║
                            ║ }                                      ║
                            ╚════════════════════════════════════════╝

                            ╔════════════════════════════════════════╗
                            ║   RUNTIME PHASE (Repeated During Training)
                            ║                                        ║
                            ║ For each training iteration:           ║
                            ╚════════════════════════════════════════╝

     1️⃣  PRIOR GENERATION
     ┌────────────────────────────────────────────────────────────┐
     │ prior_DC()                                                 │
     │                                                            │
     │ Returns:                                                   │
     │ {'theta': 0.45,  'b0': 1.2,  'k': 0.82,  ...}              │
     └────────────────┬───────────────────────────────────────────┘
                      │ (8 parameters sampled from prior)
                      ↓
     2️⃣  META GENERATION (Context selection)
     ┌────────────────────────────────────────────────────────────┐
     │ meta()                                                     │
     │                                                            │
     │ Returns:                                                   │
     │ {'number_of_trials': 60, 'tta_condition': 3.0}             │
     └────────────────┬───────────────────────────────────────────┘
                      │ (Context for this simulation)
                      ↓
     3️⃣  MERGE PRIOR + META
     ┌────────────────────────────────────────────────────────────┐
     │ {'theta': 0.45, 'b0': 1.2, ..., 'number_of_trials': 60,   │
     │  'tta_condition': 3.0}                                     │
     │                                                            │
     │ ← Dictionary.update() style merge                          │
     └────────────────┬───────────────────────────────────────────┘
                      │ (Now we have BOTH parameters and context)
                      ↓
     4️⃣  SIMULATOR EXECUTION
     ┌────────────────────────────────────────────────────────────┐
     │ ddm_DC_alphaToCpp(**merged)                                │
     │                                                            │
     │ • Receives all parameters from prior                       │
     │ • Receives context from meta (number_of_trials, tta_cond)  │
     │ • Generates observations: x = (60, 2)                      │
     │ • Returns dict(x=x) or dict(x=x, tta_per_trial=...)        │
     │                                                            │
     │ Returns:                                                   │
     │ {'x': array(60,2)}  or                                     │
     │ {'x': array(60,2), 'tta_per_trial': array(60,1)}           │
     └────────────────┬───────────────────────────────────────────┘
                      │ (Observations generated)
                      ↓
     5️⃣  MERGE ALL (prior + meta + simulator)
     ┌────────────────────────────────────────────────────────────┐
     │ {                                                          │
     │  'theta': 0.45,              ← from prior                  │
     │  'b0': 1.2,                  ← from prior                  │
     │  'k': 0.82,                  ← from prior                  │
     │  'mu_ndt': 0.35,             ← from prior                  │
     │  'sigma_ndt': 0.08,          ← from prior                  │
     │  'mu_alpah': 0.55,           ← from prior                  │
     │  'sigma_alpha': 0.12,        ← from prior                  │
     │  'sigma_cpp': 0.15,          ← from prior                  │
     │  'number_of_trials': 60,     ← from meta                   │
     │  'tta_condition': 3.0,       ← from meta                   │
     │  'x': array(60,2),           ← from simulator              │
     │  'tta_per_trial': array(60,1) ← from simulator (optional)  │
     │ }                                                          │
     └────────────────┬───────────────────────────────────────────┘
                      │ (COMPLETE dictionary - ready for adapter)
                      ↓
     6️⃣  ADAPTER TRANSFORMATION
     ┌────────────────────────────────────────────────────────────┐
     │ adopt(prior_DC())(complete_dict)                           │
     │                                                            │
     │ Operations on complete_dict:                               │
     │ • .broadcast("number_of_trials", to="x")                   │
     │ • .as_set("x")                                             │
     │ • .standardize("x", ...)                                   │
     │ • .concatenate(list(p.keys()), into=...)                   │
     │ • .rename(...) for output keys                             │
     │                                                            │
     │ Returns:                                                   │
     │ {'summary_variables': (60,2),                              │
     │  'condition_variables': 3.0,                               │
     │  'inference_variables': (8,),                              │
     │  'number_of_trials': 7.75}                                 │
     └────────────────┬───────────────────────────────────────────┘
                      │ (Network-ready format)
                      ↓
     7️⃣  NETWORKS RECEIVE INPUT
     ┌────────────────────────────────────────────────────────────┐
     │ summary_network ← summary_variables (60, 2)                │
     │                ← ??? condition_variables (3.0)             │
     │                                                            │
     │ inference_network ← pooled_h                               │
     │                  ← condition_variables (3.0)               │
     │                  (or not, depending on architecture)       │
     │                                                            │
     │ Output: p(θ | summary, condition)                          │
     └────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step: What Each Function "Knows"

### `meta()` — What does it know?

```python
def meta():
    # ❌ Does NOT know parameters (hasn't called prior yet)
    # ❌ Does NOT know what the simulator will do
    # ✅ Job: ONLY select context (TTA, trial count, etc.)
    
    tta_flag = RNG.choice(CONDITIONS)  # Pick a TTA
    return {
        "number_of_trials": 60,
        "tta_condition": tta_flag,
    }
```

**Answer:** `meta()` is **context generator**, doesn't need to know parameters.

---

### `ddm_DC_alphaToCpp()` — What does it know?

```python
def ddm_DC_alphaToCpp(
    theta,           # ✅ Knows: from prior
    b0,              # ✅ Knows: from prior
    k,               # ✅ Knows: from prior
    mu_ndt,          # ✅ Knows: from prior
    sigma_ndt,       # ✅ Knows: from prior
    mu_alpah,        # ✅ Knows: from prior
    sigma_alpha,     # ✅ Knows: from prior
    sigma_cpp,       # ✅ Knows: from prior
    number_of_trials,# ✅ Knows: from meta
    tta_condition,   # ✅ Knows: from meta
    dt=0.005         # ✅ Knows: default value
):
    # Generates observations using ALL of these!
    return dict(x=x)
```

**Answer:** `ddm_DC_alphaToCpp()` sees **everything**: parameters (from prior) + context (from meta).

---

### `adopt()` — What does it know?

```python
def adopt(p):  # p = prior_DC()
    # At adoption time: ONLY has parameter names from p
    # But when called on data, processes complete dict
    
    adapter = (
        bf.Adapter()
        # Can use: 'x' (simulator output)
        # Can use: 'number_of_trials' (from meta)
        # Can use: 'tta_condition' (from meta)
        # Can use: parameter keys (from prior)
        # Can use: 'tta_per_trial' (if simulator returns it)
        
        .concatenate(list(p.keys()), ...)
        # This uses parameter names passed to adopt()
    )
    return adapter
```

**Answer:** `adopt()` is given `prior_DC()` once, then processes the complete merged dictionary during runtime.

---

## The Key Insight: Merging Happens AUTOMATICALLY

```
What the user sees:
═══════════════════════

model = bf.simulators.make_simulator(
    [prior_DC, ddm_DC_alphaToCpp],
    meta_fn=meta
)
```

What BayesFlow does internally:
─────────────────────────────────

```python
def make_simulator(functions, meta_fn):
    def simulator_fn():
        # Step 1: Call prior
        params = functions[0]()  # prior_DC()
        
        # Step 2: Call meta
        context = meta_fn()  # meta()
        
        # Step 3: Merge
        merged = {**params, **context}
        
        # Step 4: Call remaining functions with merged dict
        for func in functions[1:]:
            result = func(**merged)
            # Step 5: Merge again
            merged = {**merged, **result}
        
        # Step 6: Return to adopt()
        return merged
    
    return simulator_fn
```

---

## Comparison: Original vs Trial-Wise Data Flow

### Original (DDM_DC_Pedestrain.py)

```
Simulator returns:
─────────────────
dict(x=x)   ← Only observations, NO tta_per_trial


Merged dict becomes:
────────────────────
{
  'theta': 0.45,
  'b0': 1.2,
  'k': 0.82,
  ...
  'number_of_trials': 60,
  'tta_condition': 3.0,     ← TTA as SCALAR
  'x': array(60,2),         ← Data without TTA
}


Adapter operations:
───────────────────
.rename("x", "summary_variables")           # (60, 2) [RT, CPP]
.rename("tta_condition", "condition_variables")  # 3.0 (scalar)

Network input:
──────────────
{
  'summary_variables': (60, 2),      ← TTA-blind data
  'condition_variables': 3.0,        ← TTA separate
  'inference_variables': (8,)
}
```

### Trial-Wise (DDM_DC_Pedestrain_TrialWise.py)

```
Simulator returns:
──────────────────
dict(x=x, tta_per_trial=tta_per_trial)  ← INCLUDES tta_per_trial!


Merged dict becomes:
────────────────────
{
  'theta': 0.45,
  'b0': 1.2,
  'k': 0.82,
  ...
  'number_of_trials': 60,
  'tta_condition': 3.0,              ← TTA as scalar (used in simulator)
  'x': array(60,2),                  ← Data without TTA
  'tta_per_trial': array(60,1),      ← NEW! TTA per trial
}


Adapter operations:
───────────────────
.concatenate(["x", "tta_per_trial"], along_axis=1, into="x_augmented")
.as_set("x_augmented")
.rename("x_augmented", "summary_variables")
# NO rename for tta_condition!

Network input:
──────────────
{
  'summary_variables': (60, 3),      ← TTA-AWARE data [RT, CPP, TTA]
  # No 'condition_variables'!
  'inference_variables': (8,)
}
```

---

## Summary: The Data Path

```
prior() ──┐
          ├──merge──┐
meta()  ──┤         ├──unpack→ simulator() ──merge──┐
          │                                        ├──→ adopt() → networks
          └─────────────────────────────────────────────┘

At each step, ALL previous outputs are merged into the next step.
```

