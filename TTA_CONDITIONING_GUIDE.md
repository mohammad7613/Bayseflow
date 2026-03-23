# TTA Conditioning in BayesFlow: Detailed Explanation

## Your Core Challenge

> "My challenge is how I should involve TTA conditions in Bayesflow. You should notice that the parameters we want to estimate are independent from TTA and set in each trial independently."

This document explains exactly how to handle this in BayesFlow 2.0.8.

---

## The Critical Insight: Understanding Your Model Structure

### What Your Model Does

Your `ddm_DC_alphaToCpp()` simulator implements:

```
INPUTS:
  - 8 fixed parameters: θ, b0, k, μ_ndt, σ_ndt, μ_α, σ_α, σ_cpp
  - 1 TTA condition value (scalar): {2.5, 3.0, 3.5, 4.0}
  - number_of_trials: how many trials to simulate

PROCESS (per trial):
  1. Sample trial-specific drift: α_trial ~ N(μ_α, σ_α)   // Uses σ_α parameter
  2. Add jitter to TTA: TTA₀ = TTA + uniform(0, 0.1)
  3. Accumulate evidence until boundary or TTA reached
  4. Add non-decision time: RT ~ N(μ_ndt, σ_ndt)
  5. Generate CPP: cpp ~ N(α_trial, σ_cpp)
  6. Output: [RT, CPP] pair

OUTPUTS (per trial):
  - Response Time (RT): time when accumulator hits boundary
  - CPP: neural signal measurement

KEY: The 8 parameters stay the SAME across all trials and TTA conditions.
     Only α_trial varies per trial (trial-to-trial noise).
     Only the decision boundary formula adapts to TTA.
```

### What BayesFlow Needs to Learn

You want to infer the posterior distribution:

$$p(\boldsymbol{\theta} | \mathbf{D}, \text{TTA})$$

Where:
- $\boldsymbol{\theta} = [\theta, b_0, k, \mu_{ndt}, \sigma_{ndt}, \mu_\alpha, \sigma_\alpha, \sigma_{cpp}]$ (8 parameters)
- $\mathbf{D} = [(RT_1, CPP_1), ..., (RT_{60}, CPP_{60})]$ (behavioral data for one condition)
- $\text{TTA} \in \{2.5, 3.0, 3.5, 4.0\}$ (conditioning variable)

**Key:** The posterior DEPENDS on TTA. For example:
- Same data might indicate $\theta=1.5$ if TTA=2.5
- But $\theta=1.2$ if TTA=4.0 (more time → lower threshold)

---

## How BayesFlow Handles Conditioning in 2.0.8

### Step 1: Define Components (Already Done in Your Code)

```python
# Prior over the 8 parameters (TTA-independent)
def prior_DC():
    return {
        'theta': RNG.uniform(0.1, 3.0),
        'b0': RNG.uniform(0.5, 2.0),
        ...
    }

# Simulator: takes 8 params + TTA → generates data
def ddm_DC_alphaToCpp(θ, b₀, ..., number_of_trials, tta_condition, dt=0.005):
    # Generates (number_of_trials, 2) array of [RT, CPP] pairs
    # for ONE specific TTA value
    pass

# Meta-function: Randomly samples ONE TTA per simulation
def meta():
    tta_flag = RNG.choice(CONDITIONS)  # Pick ONE TTA
    return {
        "number_of_trials": 60,
        "tta_condition": tta_flag,      # This TTA value
    }
```

### Step 2: The Adapter Magic (Where Conditioning Happens)

```python
def adopt(p):
    adapter = (
        bf.Adapter()
        # ... preprocessing steps ...
        .standardize("x", mean=0.0, std=1.0)        # Normalize observed data
        .convert_dtype("float64", "float32")
        
        # KEY STEP 1: Define parameters to infer
        .concatenate(list(p.keys()), into="inference_variables")
        
        # KEY STEP 2: Define observed data
        .rename("x", "summary_variables")
        
        # KEY STEP 3: Define conditioning variable
        .rename("tta_condition", "condition_variables")  ← TTA goes here!
    )
    return adapter
```

**What this does:**
```
Input dict from simulator:
{
    'x': (60, 2) array,        # Generated trials
    'tta_condition': 3.0,      # TTA value used
    'number_of_trials': 60,
    'theta': 1.234,
    'b0': 1.567,
    ... (other parameters)
}
        ↓
        ↓ (adapter transforms)
        ↓
Output dict to network:
{
    'summary_variables': (60, 2) float32 array,    # The data
    'condition_variables': (1,) float32 array,     # TTA = [3.0]
    'inference_variables': (8,) float32 array,     # The 8 parameters
}
```

### Step 3: Network Receives All Three Types

The BayesFlow network sees:

```python
# Inside the network during training:

summary_variables = (N_batch, 60, 2) float32      # Observed behavioral data
condition_variables = (N_batch, 1) float32        # TTA values for each batch
inference_variables = (N_batch, 8) float32        # Ground truth parameters

# Network learns to map:
# (summary_variables, condition_variables) → posterior over inference_variables

# Internally:
z = summary_network(summary_variables)            # Compress 60 trials to summary
p_theta = inference_network(z, condition_variables) # Posterior with TTA conditioning
```

### Step 4: Network Learns Conditional Distribution

The CouplingFlow network in BayesFlow 2.0.8 automatically:

1. **Accepts condition_variables** as an additional input
2. **Learns the conditional dependency** on TTA
3. **Outputs a conditional posterior** $p(\boldsymbol{\theta} | z, \text{TTA})$

```python
# After training, you can use it like:

# For TTA = 2.5:
posterior_distribution_tta_2_5 = workflow.approximator(
    data=summary_vars,
    condition=torch.tensor([[2.5]]),  # Condition on TTA=2.5
)
samples_tta_2_5 = posterior_distribution_tta_2_5.sample((5000,))

# For TTA = 4.0:
posterior_distribution_tta_4_0 = workflow.approximator(
    data=summary_vars,
    condition=torch.tensor([[4.0]]),  # Condition on TTA=4.0
)
samples_tta_4_0 = posterior_distribution_tta_4_0.sample((5000,))

# p(θ | data, TTA=2.5) ≠ p(θ | data, TTA=4.0)
# because decision boundaries different, but SAME 8 parameters explain both!
```

---

## Why This Approach? (Design Rationale)

### Option A: Separate Inference Per TTA ❌ (Not recommended)

```
For each TTA condition:
  - Train separate network p(θ | data)_TTA_2.5
  - Train separate network p(θ | data)_TTA_3.0
  - Train separate network p(θ | data)_TTA_3.5
  - Train separate network p(θ | data)_TTA_4.0

Disadvantages:
- ❌ 4× more networks to train
- ❌ Less data per network (wasteful)
- ❌ Can't extrapolate to novel TTA values
- ❌ Doesn't encode your model assumption: params don't depend on TTA
```

### Option B: Conditioning Inference ✅ (Your design)

```
Train ONE network: p(θ | data, TTA)

Advantages:
- ✅ Single network for all conditions
- ✅ More data per network (more efficient)
- ✅ Can predict for novel TTA values
- ✅ Explicitly encodes: "8 params fixed, TTA adapts decision boundary"
- ✅ Matches your model structure perfectly
- ✅ Better generalization
```

**Example:** Human 1 has only 40 trials for TTA=2.5 but 60 for TTA=3.0:
- Option A: Network 2.5 has fewer data → less accurate
- Option B: Network learns from all 100 trials, just conditions on TTA → uses all data

---

## Concrete Example: Your Workflow in Action

### Phase 1: Training (What parameter_recovery_test.py does)

```python
# Generate 10,000 simulations for training

For i in 1 to 10,000:
    1. Sample θ ~ prior_DC()              # Random 8 parameters
    2. Sample TTA ~ {2.5, 3.0, 3.5, 4.0}  # Random TTA
    3. Generate data: D ~ ddm_DC_alphaToCpp(θ, TTA=TTA)  # 60 trials
    4. Adapter transforms:
       summary_variables = standardize(D.x)
       condition_variables = [TTA]
       inference_variables = [θ₁, θ₂, ..., θ₈]
    5. Network trains: minimize KL(q(θ|D,TTA) || p(θ|D,TTA))
```

**Result after training:**
- Network has seen  10k × 4 = 40k condition-data pairs
- Learned that different TTAs → different posteriors (but same 8 params)
- Can now infer on real behavioral data

### Phase 2: Inference on Real Data (What happens next)

```python
# Load real behavioral data for Human #42

Real_data_TTA_2_5 = [60 observations with TTA=2.5]
Real_data_TTA_3_0 = [45 observations with TTA=3.0]
Real_data_TTA_3_5 = [58 observations with TTA=3.5]
Real_data_TTA_4_0 = [62 observations with TTA=4.0]

# Option 1: Pool all conditions
all_data = concatenate all observations
condition_labels = [2.5, 2.5, ..., 2.5, 3.0, 3.0, ..., 4.0, 4.0, ...]
posterior_across_all = workflow(all_data, condition_labels)
# Single posterior incorporating all conditions

# Option 2: Infer separately per condition
posterior_2_5 = workflow(Real_data_TTA_2_5, [2.5]*60)
posterior_3_0 = workflow(Real_data_TTA_3_0, [3.0]*45)
posterior_3_5 = workflow(Real_data_TTA_3_5, [3.5]*58)
posterior_4_0 = workflow(Real_data_TTA_4_0, [4.0]*62)
# Compare posteriors: should be similar (if model is right)
```

---

## Implementation Checklist: Are You Ready?

### Your Simulator ✅
```python
def ddm_DC_alphaToCpp(..., tta_condition, ...):
    """Takes SINGLE TTA value, generates data"""
    # ✅ Takes tta_condition as input
    # ✅ Returns (num_trials, 2) array + dict
    pass
```

### Your Meta Function ✅
```python
def meta():
    """Returns ONE random TTA per call"""
    tta_flag = RNG.choice(CONDITIONS)  # ✅ ONE TTA, not all 4
    return {
        "number_of_trials": 60,
        "tta_condition": tta_flag,     # ✅ As meta-variable
    }
```

### Your Adapter ✅
```python
def adopt(p):
    """Renames tta_condition to condition_variables"""
    adapter = (
        # ... other steps ...
        .rename("tta_condition", "condition_variables")  # ✅ CRITICAL
    )
    return adapter
```

### Your Network ✅
```python
inference_network = bf.networks.CouplingFlow(
    num_dimensions=8,
    conditional_shape=(1,),  # ✅ TTA is 1-D scalar
)
# ✅ BayesFlow 2.0.8 automatically handles conditioning
```

---

## Expected Behavior: Validation Checks

After training, your conditional network should show:

```python
# Load trained workflow
workflow = load_checkpoint(...)

# Sample data point
participant_data_TTA_2_5 = (60, 2) array
participant_data_TTA_4_0 = (60, 2) array

# Inference 1: Same data, different TTA
posterior_2_5 = workflow(participant_data_TTA_2_5, TTA=2.5)
posterior_4_0 = workflow(participant_data_TTA_2_5, TTA=4.0)  # Same data!

# OBSERVATION: Posteriors should differ!
# - TTA=2.5 (quick decision): might infer higher θ (earlier stopping)
# - TTA=4.0 (more time): might infer lower θ (later stopping)
# Because the same behavioral pattern means different decisions under time pressure

# Inference 2: Different data from same person across TTAs
posterior_across_all = workflow(all_data_from_person, varying_TTAs)

# OBSERVATION: Should find SAME 8 parameters across conditions
# Because person has consistent cognitive strategy
```

---

## Common Misconceptions to Avoid

### ❌ Misconception 1: "Should I include TTA as the 9th inference parameter?"

NO. Your model structure assumes TTA is NOT a parameter:
- TTA is exogenous (determined by experimenter/environment)
- TTA only affects the decision boundary formula
- The 8 cognitive parameters are fixed across TTAs

Your code is correct: TTA is conditioning variable, not inference parameter.

### ❌ Misconception 2: "Should I train separate networks for each TTA?"

NO. Conditional inference is more efficient:
- One network learns from all TTA conditions
- Network learns the conditional structure naturally
- Less computation, better generalization

### ❌ Misconception 3: "How do I handle different # trials per TTA?"

Use `.as_set()` in adapter (already in your code):
```python
.as_set("x")  # Treats trials as exchangeable, enables variable length
```
Network automatically handles 45 or 60 trials—doesn't matter.

### ❌ Misconception 4: "Don't I need to normalize TTA like the data?"

NO. Leave TTA raw (usually scaled to 0-10 or similar):
```python
# Standardize data
.standardize("x", mean=0.0, std=1.0)  # Yes

# Don't standardize TTA
.rename("tta_condition", "condition_variables")  # Use as-is
```
Network embedding layer handles scaling.

---

## Debugging: What Could Go Wrong?

### Symptom 1: "Posterior doesn't change when I change TTA"

**Problem:** Network didn't learn conditioning.  
**Solution:**
1. Check that condition_variables shape is (batch, 1): `print(batch['condition_variables'].shape)`
2. Verify all 4 TTAs appear in training: `print(train_batch['condition_variables'].unique())`
3. Increase num_coupling_layers (more capacity for conditional mapping)

### Symptom 2: "Parameter recovery is bad (< 80% coverage)"

**Problem:** Network didn't learn sufficient diversity.  
**Solution:**
1. Increase n_simulations_train to 20000+
2. Increase epochs to 15-20
3. Verify prior ranges cover realistic parameter space
4. Check synthetic data is actually variable (shouldn't get same output every time)

### Symptom 3: "Real data inference gives crazy posteriors"

**Problem:** Model mismatch or data quality issue.  
**Solution:**
1. Run parameter recovery test first (validates network is working)
2. Check real behavioral data format matches assumptions
3. Verify RTs and CPPs are in plausible ranges (compare to paper)
4. Start with first participant as manual test

---

## Mathematical Formulation (Optional Reading)

The posterior you're learning is:

$$p(\boldsymbol{\theta} | \mathbf{D}, c) = \frac{p(\mathbf{D} | \boldsymbol{\theta}, c) p(\boldsymbol{\theta})}{p(\mathbf{D} | c)}$$

Where:
- $\mathbf{D}$: observed behavioral data (60 trials, [RT, CPP] each)
- $c$: conditioning variable (TTA ∈ {2.5, 3.0, 3.5, 4.0})
- $\boldsymbol{\theta}$: 8 cognitive parameters
- $p(\boldsymbol{\theta})$: prior (uniform in your case)

BayesFlow learns this using:
1. **Simulator:** provides $p(\mathbf{D} | \boldsymbol{\theta}, c)$ via sampling
2. **Summary network:** compresses $\mathbf{D}$ to $z = \psi(\mathbf{D})$
3. **Inference network:** learns $q(\boldsymbol{\theta} | z, c) \approx p(\boldsymbol{\theta} | \mathbf{D}, c)$

Training minimizes: $\text{KL}(q || p) = \mathbb{E}_p[\log q - \log p]$

---

## Summary: Your Workflow Choice

### ✅ What You Decided (Correct!)

```
Inference task: p(θ | data, TTA)
    ↓
Design: Condition on TTA
    ↓
Implementation: condition_variables in adapter
    ↓
Network: CouplingFlow with conditional_shape=(1,)
    ↓
Result: 8 parameters + TTA conditioning = accurate inference
```

### ✅ Why This Is Right

1. **Model alignment:** Your DDM assumes parameters don't depend on TTA
2. **Data efficiency:** Uses all trials for single network
3. **Flexibility:** Can infer on variable trial counts per condition
4. **Generalization:** Can predict on novel TTA values
5. **Interpretability:** Clear separation: fixed params vs. TTA-dependent boundaries

---

## Next Steps

1. **Run parameter_recovery_test.py** to validate this architecture
2. **Read full design document** for real data integration
3. **Check FAQ** in BAYESFLOW_PIPELINE_DESIGN.md for more details
4. **Proceed to Phase 2** once recovery validation passes

---

## Questions?

This document answers: **"How should I involve TTA conditions in Bayesflow?"**

Answer: **Use as conditioning variable via condition_variables in the adapter.**

For more context:
- See BAYESFLOW_PIPELINE_DESIGN.md Sections 2-3
- See QUICKSTART_PARAMETER_RECOVERY.md for practical steps
- See BAYESFLOW_PIPELINE_DESIGN.md Section 14 FAQ
