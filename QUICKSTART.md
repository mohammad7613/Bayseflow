# Quick Reference: Environment Setup

## 🚀 One-Command Setup (Automated)

```bash
bash setup_environment.sh
```

This script will:
- ✓ Check Python version
- ✓ Create virtual environment
- ✓ Install all dependencies
- ✓ Detect GPU/CPU
- ✓ Run verification tests

---

## 📋 Manual Setup (3 Steps)

### Step 1: Create Environment
```bash
python3 -m venv bayesflow_env
source bayesflow_env/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Verify
```bash
python test_setup.py
```

✓ Done!

---

## 💻 Daily Usage

### Start Working
```bash
cd "/media/mohammad/New Volume/DoctoralSharif/Articles/Matin/train_joint_models"
source bayesflow_env/bin/activate
```

### Train Model
```bash
python main.py
```

### Run Evaluation
```bash
jupyter notebook evaluation_conditional.ipynb
```

### When Done
```bash
deactivate
```

---

## 🔧 Troubleshooting

### "No module named 'bayesflow'"
```bash
source bayesflow_env/bin/activate
pip install bayesflow
```

### "No module named 'torch'"
```bash
pip install torch torchvision
```

### NumPy version issues
```bash
pip install "numpy<2.0"
```

### GPU not working
```bash
# Check CUDA
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📦 Required Packages

```
✓ Python 3.8-3.12
✓ bayesflow >= 1.0.0
✓ torch >= 2.0.0
✓ keras >= 3.0.0
✓ tensorflow >= 2.16.0 (2.16+ for Python 3.12)
✓ numpy < 2.0.0
✓ pandas >= 2.0.0
✓ matplotlib >= 3.7.0
✓ scipy >= 1.10.0
✓ jupyter (for notebooks)
```

---

## ⚡ Quick Commands

```bash
# Setup (once)
bash setup_environment.sh

# Activate (every session)
source bayesflow_env/bin/activate

# Test
python test_setup.py

# Train
python main.py

# Deactivate
deactivate
```

---

## 📚 Full Documentation

- **Detailed setup**: [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)
- **Usage guide**: [CONDITIONAL_INFERENCE_GUIDE.md](CONDITIONAL_INFERENCE_GUIDE.md)
- **Workflow**: [WORKFLOW_VISUAL.md](WORKFLOW_VISUAL.md)

---

## ⏱️ Expected Time

- **Setup**: 5-10 minutes
- **Training (CPU)**: 2-4 hours
- **Training (GPU)**: 20-40 minutes
- **Evaluation**: 10-15 minutes

---

## ✅ Verification

After setup, you should see:
```
============================================================
ALL TESTS PASSED ✓
============================================================
```

If not, check [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for troubleshooting.
