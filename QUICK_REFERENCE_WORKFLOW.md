# Quick Reference: Trial-Wise DDM-DC Workflow

## ✅ What Was Fixed & Created

### Fixed Issue
```
❌ ERROR: Adapter.concatenate() got an unexpected keyword argument 'along_axis'
✅ FIXED: Removed unsupported parameter from DDM_DC_Pedestrain_TrialWise.py line 178
```

### Created Files
1. **workflow_trialwise.py** - Complete workflow backend with 3 phases
2. **main_workflow.py** - New entry point with flexible phase selection
3. **WORKFLOW_GUIDE_TRIALWISE.md** - Comprehensive documentation
4. **ENTRY_POINTS_EXPLAINED.md** - Explains main.py vs main_workflow.py

---

## 🚀 Quick Start (Choose One)

### Option A: Quick Training (Original)
```bash
python main.py
```
✅ Starts training immediately
✅ No arguments needed
✅ Must continue running to completion

### Option B: Structured Workflow (Recommended)
```bash
# Phase 1: Train (1-2 hours)
python main_workflow.py train

# Phase 2: Evaluate recovery (10-20 min)
python main_workflow.py recovery

# Phase 3: Ready for real data
python main_workflow.py inference

# Or all at once:
python main_workflow.py all
```

---

## 📊 Three Workflow Phases

### PHASE 1: TRAINING
```python
python main_workflow.py train

What happens:
- Simulates DDM-DC trials from your prior
- Trains neural networks to map RT→parameter posterior
- Automatically resumes from checkpoint if interrupted
- Saves to: trained_model1/checkpoints/{model_name}.keras

Configuration:
- n_sim: 1000 simulations per epoch
- epochs: 100 total epochs
- batch_size: 32
- resume_epochs: 50 (additional when resuming)
```

### PHASE 2: RECOVERY ANALYSIS
```python
python main_workflow.py recovery

What happens:
- Generates 5000 test simulations with TRUE parameters
- Runs trained model on test simulations
- Evaluates: Does posterior recover true parameters?
- Saves to: results/{model_name}_recovery.npz

Expected output:
- posterior_samples: (5000, 8) - 8 inferred parameters
- Good recovery: Posterior mean ≈ True value
```

### PHASE 3: REAL DATA INFERENCE
```python
from workflow_trialwise import infer_subject_parameters, infer_batch_subjects

# Single subject
result = infer_subject_parameters({
    'cpp': [0.5, 0.8, 0.3, ...],  # Neural signal per trial
    'reaction_times': [0.45, 0.52, ...]  # Behavior per trial
}, model_name='model_DC_TrialWise')

# Multiple subjects
results_df = infer_batch_subjects(subject_list, model_name='model_DC_TrialWise')

Output:
- 8 parameters: [v, a, t0, z, ndt, dc, sv, st]
- Point estimates: mean, median, std for each parameter
- CSV file: batch_inference_{timestamp}.csv
```

---

## 🔧 Configuration

All settings in `workflow_trialwise.py`:

```python
CONFIG = {
    "device": "cuda" or "cpu",  # Auto-detected
    "training": {
        "n_sim": 1000,          # ← Increase for better recovery
        "epochs": 100,          # ← Increase for longer training
        "batch_size": 32,       # ← Decrease if GPU OOM
        "resume_epochs": 50,
    },
    "recovery": {
        "n_test_sims": 5000,
        "n_posterior_samples": 5000
    },
    "paths": {
        "checkpoints": "trained_model1/checkpoints",
        "results": "results",
        "logs": "logs"
    }
}
```

---

## 📁 Key Files

```
train_joint_models/
├── main.py                          ← Original (quick training)
├── main_workflow.py                 ← New (multi-phase)
├── workflow_trialwise.py            ← Backend (core implementation)
├── DDM_DC_Pedestrain_TrialWise.py  ← Models (FIXED ✅)
├── train.py                         ← Training utilities
├── utils_real_data.py               ← Real data loading
│
├── WORKFLOW_GUIDE_TRIALWISE.md      ← Full documentation
├── ENTRY_POINTS_EXPLAINED.md        ← main.py vs main_workflow.py
├── QUICKSTART.md                    ← Original quickstart
│
├── trained_model1/checkpoints/      ← Saved models
│   └── model_DC_TrialWise.keras
│
├── results/                         ← Analysis outputs
│   ├── model_DC_TrialWise_recovery.npz
│   ├── model_DC_TrialWise_recovery.json
│   └── batch_inference_*.csv
│
└── logs/
    └── workflow.log                 ← All events with timestamps
```

---

## 📝 Real Data Format

Your subject data must have:

```python
subject_data = {
    'subject_id': 'S001',                          # Optional
    'cpp': np.array([0.45, 0.52, 0.38, ...]),     # n_trials
    'reaction_times': np.array([0.45, 0.52, ...]) # n_trials (seconds)
}
```

**Important**:
- CPP and RT must have same length (one value per trial)
- RT should be in seconds (typical: 0.3-1.0s)
- CPP should be normalized/scaled appropriately for your study

---

## 💡 Common Tasks

### Train from scratch
```bash
python main.py
# or
python main_workflow.py train
```

### Resume interrupted training
```bash
# Same command - automatically detects checkpoint
python main_workflow.py train
```

### Evaluate recovery after training
```bash
python main_workflow.py recovery
# Check results in: results/model_DC_TrialWise_recovery.json
```

### Infer on real subject
```python
from workflow_trialwise import infer_subject_parameters

# Load your data
cpp = load_cpp_data('S001')      # Your loading function
rt = load_reaction_times('S001')  # Your loading function

# Estimate parameters
result = infer_subject_parameters({
    'cpp': cpp,
    'reaction_times': rt
}, model_name='model_DC_TrialWise')

# Access results
estimates = result['posterior_mean']          # [v, a, t0, z, ndt, dc, sv, st]
uncertainty = result['posterior_std']
samples = result['posterior_samples']  # Full posterior for Bayesian analysis
```

### Batch process multiple subjects
```python
from workflow_trialwise import infer_batch_subjects

subjects = [
    {'subject_id': 'S001', 'cpp': ..., 'reaction_times': ...},
    {'subject_id': 'S002', 'cpp': ..., 'reaction_times': ...},
    # ... more subjects
]

results = infer_batch_subjects(subjects, model_name='model_DC_TrialWise')
results.to_csv('estimates.csv')
```

### View training progress
```bash
# Monitor logs in real-time
tail -f logs/workflow.log
```

---

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: bayesflow` | `source bayesflow_env/bin/activate.fish` |
| `CUDA out of memory` | Reduce `batch_size`, `n_sim`, or use CPU |
| `Checkpoint not found` | Training hasn't completed yet |
| Poor recovery | Increase `epochs` or `n_sim` |
| Slow training | Try `batch_size=16` or reduce `epochs` |

---

## 📚 Documentation

- **Full guide**: [WORKFLOW_GUIDE_TRIALWISE.md](WORKFLOW_GUIDE_TRIALWISE.md)
- **Entry points**: [ENTRY_POINTS_EXPLAINED.md](ENTRY_POINTS_EXPLAINED.md)
- **This file**: [QUICK_REFERENCE_WORKFLOW.md](QUICK_REFERENCE_WORKFLOW.md)

---

## 🎯 Next Steps

1. ✅ Fix applied - code ready to run
2. 📖 Read [WORKFLOW_GUIDE_TRIALWISE.md](WORKFLOW_GUIDE_TRIALWISE.md) for details
3. 🚀 Run training: `python main_workflow.py train`
4. 📊 Run recovery: `python main_workflow.py recovery` (after training)
5. 🧠 Run inference: Use `infer_batch_subjects()` on your real data

---

**Created**: 2026-02-13
**Status**: ✅ Ready to use
