# Two Entry Points: main.py vs main_workflow.py

## Summary

You now have **two separate entry points** for your DDM-DC Trial-Wise workflow:

### 1. **main.py** (Original - Unchanged)
- **Purpose**: Direct training execution
- **Behavior**: Immediately starts training all models
- **Usage**: `python main.py`
- **Output**: Begins training with current parameters

### 2. **main_workflow.py** (New - Comprehensive)
- **Purpose**: Flexible workflow with multiple phases
- **Behavior**: Allows selection of which phase to run
- **Usage**: `python main_workflow.py [phase]`
- **Phases**: `train`, `recovery`, `inference`, `all`

---

## Comparison Table

| Feature | main.py | main_workflow.py |
|---------|---------|-----------------|
| Quick Start | ✅ Yes | ❌ No (requires phase argument) |
| Training | ✅ Automatic | ✅ Manual (`train` phase) |
| Recovery Analysis | ❌ Not included | ✅ Available (`recovery` phase) |
| Real Data Inference | ❌ Not included | ✅ Ready (`inference` phase) |
| Configuration UI | ❌ None | ✅ Status display |
| Resume Support | ✅ Automatic | ✅ Automatic |
| Logging | ✅ Basic | ✅ Comprehensive |

---

## When to Use Each

### Use **main.py** if:
- You just want to train your models
- You prefer minimal setup
- You're familiar with the code
- You want to run training in background

### Use **main_workflow.py** if:
- You want to run multiple phases sequentially
- You need parameter recovery analysis
- You want to prepare for real data inference
- You want comprehensive logging and status tracking
- You want flexibility to run individual phases

---

## Usage Examples

### **main.py** - Training Only
```bash
# Start training immediately
python main.py

# (Automatically resumes from checkpoint if exists)
```

### **main_workflow.py** - Flexible Workflow
```bash
# Phase 1: Train (1-2 hours depending on config)
python main_workflow.py train

# Phase 2: Recovery Analysis (10-20 minutes)
python main_workflow.py recovery

# Phase 3: Prepare for Real Data Inference
python main_workflow.py inference

# Or run everything at once
python main_workflow.py all
```

---

## File Structure

```
train_joint_models/
├── main.py                    ← Original (unchanged)
├── main_workflow.py           ← New (multi-phase)
├── workflow_trialwise.py      ← Backend implementation
├── DDM_DC_Pedestrain_TrialWise.py  ← Fixed (concatenate)
├── train.py                   ← Training utilities
├── utils_real_data.py         ← Real data utilities
├── WORKFLOW_GUIDE_TRIALWISE.md ← Comprehensive guide
└── trained_model1/
    └── checkpoints/           ← Saved models
```

---

## Configuration

Both entry points use the same configuration from `workflow_trialwise.py`:

```python
CONFIG = {
    "training": {
        "n_sim": 1000,          # Sims per epoch
        "epochs": 100,          # Total epochs
        "batch_size": 32,
        "resume_epochs": 50,    # Resume additional epochs
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

To modify: Edit the CONFIG dictionary in `workflow_trialwise.py`

---

## Output Organization

```
trained_model1/checkpoints/
├── model_DC_TrialWise.keras     ← Trained model checkpoint

results/
├── model_DC_TrialWise_recovery.npz    ← Posterior samples
├── model_DC_TrialWise_recovery.json   ← Metadata
└── batch_inference_*.csv              ← Real data results

logs/
└── workflow.log                  ← Comprehensive log
```

---

## Key Features Added

✅ **Fixed**: Removed `along_axis` parameter from `Adapter.concatenate()` 

✅ **Training**: Checkpoint saving/resuming with PyTorch optimizer state

✅ **Recovery Analysis**: Parameter recovery evaluation on test simulations

✅ **Real Data Inference**: Single subject and batch inference pipeline

✅ **Logging**: Timestamped logs to `logs/workflow.log`

✅ **GPU Support**: Automatic CUDA detection

---

## Next Steps

### For Quick Training:
```bash
python main.py
```

### For Complete Workflow:
```bash
# Terminal 1: Train (hours)
python main_workflow.py train

# Terminal 2 (after training done): Evaluate
python main_workflow.py recovery

# Terminal 3: Prepare for inference
python main_workflow.py inference
```

### For Real Data:
```python
from main_workflow import infer_batch_subjects

# Load your subject data
subject_list = [...]  # List of dicts with 'cpp', 'reaction_times'

# Run inference
results_df = infer_batch_subjects(subject_list, model_name='model_DC_TrialWise')

# Save results
results_df.to_csv('estimated_parameters.csv')
```

---

## Troubleshooting

**Q: Which should I use?**
- **Training only** → `main.py`
- **Full workflow** → `main_workflow.py`

**Q: Can I switch between them?**
- Yes! They both use the same checkpoints and configuration
- Progress is preserved between runs

**Q: How do I modify training settings?**
- Edit `CONFIG` in `workflow_trialwise.py`
- Changes apply to both entry points

**Q: What if training is interrupted?**
- Checkpoint is automatically saved
- Run again (any entry point) to resume from last epoch

---

## References

- **Original code**: `original_main.py` (deleted, now `main.py`)
- **New workflow**: `main_workflow.py` + `workflow_trialwise.py`
- **Guide**: [WORKFLOW_GUIDE_TRIALWISE.md](WORKFLOW_GUIDE_TRIALWISE.md)
- **Backend**: [workflow_trialwise.py](workflow_trialwise.py)

---

**Last Updated**: 2026-02-13
