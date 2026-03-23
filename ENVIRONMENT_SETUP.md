# Python Environment Setup for BayesFlow Conditional Inference

## Quick Start (Recommended)

```bash
# Navigate to your project directory
cd "/media/mohammad/New Volume/DoctoralSharif/Articles/Matin/train_joint_models"

# Create virtual environment
python3 -m venv bayesflow_env

# Activate environment
source bayesflow_env/bin/activate  # Linux/Mac
# OR
# bayesflow_env\Scripts\activate  # Windows

# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python test_setup.py
```

If all tests pass, you're ready to go! 🎉

---

## Detailed Setup Instructions

### Step 1: Check Python Version

BayesFlow requires **Python 3.8 or higher** (3.9-3.12 recommended).

**Note**: Python 3.12 requires TensorFlow 2.16+ (automatically handled by requirements.txt)

```bash
python3 --version
```

If you need to install Python:
- **Ubuntu/Debian**: `sudo apt-get install python3.10 python3.10-venv`
- **Fedora/RHEL**: `sudo dnf install python3.10`
- **macOS**: `brew install python@3.10`
- **Windows**: Download from [python.org](https://www.python.org/downloads/)

### Step 2: Create Virtual Environment

```bash
# Navigate to project directory
cd "/media/mohammad/New Volume/DoctoralSharif/Articles/Matin/train_joint_models"

# Create virtual environment
python3 -m venv bayesflow_env

# Activate it
source bayesflow_env/bin/activate
```

You should see `(bayesflow_env)` in your terminal prompt.

### Step 3: Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 4: Install PyTorch

BayesFlow uses PyTorch as backend. Install the appropriate version for your system:

**For CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**For GPU (CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For GPU (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Check GPU availability:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 5: Install BayesFlow

```bash
pip install bayesflow
```

This installs BayesFlow and its core dependencies.

### Step 6: Install Additional Dependencies

```bash
pip install keras tensorflow numpy pandas matplotlib scipy jupyter notebook ipykernel
```

### Step 7: Verify Installation

```bash
python test_setup.py
```

Expected output:
```
============================================================
Testing Conditional Inference Setup
============================================================

[1/6] Testing imports...
✓ All imports successful

[2/6] Checking TTA conditions...
✓ CONDITIONS match experiment design

[3/6] Testing simulator...
✓ Simulator produces correct output format

[4/6] Testing meta function...
✓ Meta function works correctly

[5/6] Testing model.sample()...
✓ Model sampling works correctly

[6/6] Testing adapter...
✓ Adapter correctly handles condition variables

============================================================
ALL TESTS PASSED ✓
============================================================
```

---

## Using requirements.txt (Easiest Method)

I've created a `requirements.txt` file. Install everything at once:

```bash
# Activate your virtual environment first
source bayesflow_env/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

---

## Package Versions (Tested and Compatible)

```
bayesflow>=1.0.0
torch>=2.0.0
keras>=3.0.0
tensorflow>=2.16.0  # 2.16+ for Python 3.12, 2.15+ for Python 3.8-3.11
numpy>=1.23.0,<2.0.0
pandas>=2.0.0
matplotlib>=3.7.0
scipy>=1.10.0
jupyter>=1.0.0
notebook>=7.0.0
ipykernel>=6.25.0
```

---

## Environment Activation

**Every time you work on this project**, activate the environment:

```bash
# Navigate to project
cd "/media/mohammad/New Volume/DoctoralSharif/Articles/Matin/train_joint_models"

# Activate environment
source bayesflow_env/bin/activate
```

To deactivate when done:
```bash
deactivate
```

---

## Setting Up Jupyter Notebook (Optional)

If you want to use the evaluation notebook:

```bash
# With environment activated
pip install jupyter notebook ipykernel

# Register kernel
python -m ipykernel install --user --name=bayesflow_env --display-name "Python (BayesFlow)"

# Start Jupyter
jupyter notebook
```

In Jupyter, select the "Python (BayesFlow)" kernel for your notebooks.

---

## Troubleshooting

### Issue: "No module named 'bayesflow'"

**Solution:**
```bash
# Make sure environment is activated
source bayesflow_env/bin/activate

# Install BayesFlow
pip install bayesflow
```

### Issue: "ImportError: No module named 'torch'"

**Solution:**
```bash
pip install torch torchvision
```

### Issue: "KERAS_BACKEND not set" or Keras errors

**Solution:**
```bash
# Install both Keras and TensorFlow
pip install keras>=3.0.0 tensorflow>=2.15.0

# Set backend explicitly (in your Python code or shell)
export KERAS_BACKEND=torch  # Linux/Mac
# OR
set KERAS_BACKEND=torch  # Windows
```

### Issue: BayesFlow version conflicts

**Solution:**
```bash
# Uninstall and reinstall with specific version
pip uninstall bayesflow
pip install bayesflow==1.0.0  # or latest version
```

### Issue: NumPy version conflicts

BayesFlow may have issues with NumPy 2.0+. Use NumPy 1.x:

**Solution:**
```bash
pip install "numpy<2.0"
```

### Issue: TensorFlow version conflicts

**For Python 3.12+**: Requires TensorFlow 2.16 or higher
**For Python 3.8-3.11**: Can use TensorFlow 2.15 or higher

**Solution:**
```bash
# Python 3.12
pip install tensorflow>=2.16.0

# Python 3.8-3.11
pip install tensorflow>=2.15.0
```

The requirements.txt file is already configured correctly for all Python versions.

### Issue: GPU not detected

**Solution:**
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Testing Your Installation

### Quick Test

```bash
python -c "import bayesflow as bf; print(f'BayesFlow version: {bf.__version__}')"
```

### Full Test

```bash
python test_setup.py
```

### Manual Test

```bash
python
```

Then in Python:
```python
>>> import bayesflow as bf
>>> import torch
>>> import numpy as np
>>> import keras
>>> 
>>> # Test basic imports
>>> print("✓ All imports successful")
>>> 
>>> # Load your model
>>> from DDM_DC_Pedestrain import all_models
>>> print("✓ Model loaded successfully")
>>> 
>>> # Test sampling
>>> model, adapter = all_models['model_DC']
>>> sample = model.sample(1)
>>> print(f"✓ Sampling works: {sample['x'].shape}")
```

---

## Alternative: Using Conda (Advanced)

If you prefer Conda:

```bash
# Create conda environment
conda create -n bayesflow_env python=3.10

# Activate
conda activate bayesflow_env

# Install PyTorch with conda
conda install pytorch torchvision -c pytorch

# Install other packages
pip install bayesflow keras tensorflow pandas matplotlib scipy jupyter
```

---

## Directory Structure After Setup

```
train_joint_models/
├── bayesflow_env/           # Virtual environment (created)
│   ├── bin/
│   ├── lib/
│   └── ...
├── DDM_DC_Pedestrain.py
├── train.py
├── requirements.txt         # Dependencies (created)
├── test_setup.py
├── evaluation_conditional.ipynb
├── utils_real_data.py
└── ... (other files)
```

---

## System Requirements

### Minimum Requirements
- **CPU**: 2+ cores
- **RAM**: 8 GB
- **Storage**: 5 GB for environment + data
- **OS**: Linux, macOS, or Windows

### Recommended for Training
- **CPU**: 8+ cores OR **GPU**: NVIDIA with CUDA support
- **RAM**: 16 GB+
- **Storage**: 20 GB+ (for checkpoints and results)

---

## Expected Training Performance

### CPU (8 cores, 16GB RAM)
- **50 epochs** × 10,000 sims/epoch: ~2-4 hours

### GPU (NVIDIA RTX 3080, CUDA 11.8)
- **50 epochs** × 10,000 sims/epoch: ~20-40 minutes

### GPU (NVIDIA A100)
- **50 epochs** × 10,000 sims/epoch: ~10-15 minutes

---

## Post-Installation Workflow

Once environment is set up:

1. **Activate environment**:
   ```bash
   source bayesflow_env/bin/activate
   ```

2. **Test setup**:
   ```bash
   python test_setup.py
   ```

3. **Train model**:
   ```bash
   python main.py
   ```

4. **Evaluate** (Jupyter):
   ```bash
   jupyter notebook evaluation_conditional.ipynb
   ```

5. **Apply to real data**:
   ```python
   from utils_real_data import process_all_subjects
   # ... your code
   ```

---

## Getting Help

If you encounter issues:

1. Check this guide's Troubleshooting section
2. Run `python test_setup.py` to diagnose problems
3. Check BayesFlow documentation: https://bayesflow.readthedocs.io/
4. Verify Python version: `python --version` (should be 3.8+)
5. Check package versions: `pip list | grep -E "bayesflow|torch|keras"`

---

## Summary of Commands

```bash
# Setup (one time)
cd "/media/mohammad/New Volume/DoctoralSharif/Articles/Matin/train_joint_models"
python3 -m venv bayesflow_env
source bayesflow_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python test_setup.py

# Daily use (every session)
source bayesflow_env/bin/activate
python main.py  # or your script
deactivate  # when done
```

---

**Status**: Ready to use once environment is set up!  
**Last Updated**: January 30, 2026
