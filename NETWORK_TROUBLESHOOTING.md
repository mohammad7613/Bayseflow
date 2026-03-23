# Network Troubleshooting & Offline Installation Guide

## Your Current Issue

The errors show:
```
[Errno -2] Name or service not known
Could not find a version that satisfies the requirement bayesflow>=1.0.0
```

This is a **network connectivity issue**, not a package problem.

---

## Quick Fixes (Try These First)

### 1. Check Network Connectivity

```bash
# Test internet connection
ping -c 3 google.com

# Test PyPI connectivity
ping -c 3 pypi.org
curl -I https://pypi.org/simple/

# Test DNS
nslookup pypi.org
```

If ping fails, you have a network issue to resolve first.

### 2. Check Proxy Settings

If you're behind a corporate proxy/firewall:

```bash
# Check if proxy is set
echo $http_proxy
echo $https_proxy

# If you need to set proxy (ask your IT department for values)
export http_proxy=http://proxy.your-company.com:port
export https_proxy=http://proxy.your-company.com:port

# Make persistent
echo 'export http_proxy=http://proxy.your-company.com:port' >> ~/.bashrc
echo 'export https_proxy=http://proxy.your-company.com:port' >> ~/.bashrc
```

### 3. Try Alternative PyPI Mirror

```bash
# Activate your environment
source bayesflow_env/bin/activate

# Use a mirror (example: Chinese mirror if you're in Asia)
pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple bayesflow

# Or try different mirrors
pip install --index-url https://mirrors.aliyun.com/pypi/simple/ bayesflow
```

### 4. Increase Timeout

```bash
pip install --timeout 100 bayesflow
```

---

## Solution 1: Use Your Working PyTorch Environment

**Good news**: PyTorch is already installed! Let's work with what you have:

```bash
# Activate environment
source bayesflow_env/bin/activate

# Check what's installed
pip list

# You already have these critical packages:
# - torch 2.7.1+cu118 ✓
# - numpy 2.3.5 (need to downgrade to <2.0)
# - pandas, scipy, matplotlib (if installed)
```

### Manual Installation When Network Returns

When your network is back, run these commands **one at a time**:

```bash
source bayesflow_env/bin/activate

# 1. Fix NumPy version (critical for BayesFlow)
pip install "numpy<2.0.0" --force-reinstall

# 2. Install BayesFlow
pip install bayesflow

# 3. Install Keras (PyTorch backend)
pip install keras

# 4. Install scientific packages
pip install scipy pandas matplotlib

# 5. Install Jupyter (optional)
pip install jupyter notebook ipykernel
```

---

## Solution 2: Offline Installation (If Network Issues Persist)

### Step 1: Download Packages on a Computer with Internet

On a computer with internet access:

```bash
# Create a directory for packages
mkdir bayesflow_packages
cd bayesflow_packages

# Download all packages
pip download bayesflow
pip download "numpy<2.0.0"
pip download keras
pip download scipy pandas matplotlib
pip download jupyter notebook ipykernel
pip download tqdm seaborn
```

### Step 2: Transfer to Your Machine

```bash
# Copy the bayesflow_packages folder to your machine via:
# - USB drive
# - Network share
# - scp/rsync
```

### Step 3: Install Offline

```bash
# On your machine
source bayesflow_env/bin/activate
cd bayesflow_packages

# Install from local packages
pip install --no-index --find-links . bayesflow
pip install --no-index --find-links . "numpy<2.0.0" --force-reinstall
pip install --no-index --find-links . keras scipy pandas matplotlib
pip install --no-index --find-links . jupyter notebook ipykernel
```

---

## Solution 3: Use Conda (Alternative Package Manager)

Conda has different mirrors and might work better:

```bash
# Install miniconda if not installed
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create environment
conda create -n bayesflow_env python=3.11

# Activate
conda activate bayesflow_env

# Install packages from conda-forge
conda install -c conda-forge numpy scipy pandas matplotlib jupyter

# Install PyTorch
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install remaining with pip
pip install bayesflow keras
```

---

## Solution 4: Minimal Setup (Just to Get Started)

If you just want to test if things work with minimal packages:

```bash
source bayesflow_env/bin/activate

# You already have PyTorch, let's just fix NumPy
python -m pip install --upgrade "numpy==1.26.4"

# Try to import (this will tell us what else is missing)
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
```

---

## Troubleshooting Network Issues

### Check 1: DNS Resolution

```bash
# Add public DNS servers temporarily
sudo nano /etc/resolv.conf

# Add these lines:
nameserver 8.8.8.8
nameserver 8.8.4.4
```

### Check 2: Firewall

```bash
# Check if firewall is blocking
sudo ufw status

# If needed, allow HTTPS
sudo ufw allow 443/tcp
```

### Check 3: Network Manager

```bash
# Restart network
sudo systemctl restart NetworkManager

# Or
sudo service network-manager restart
```

### Check 4: VPN/Proxy

```bash
# Disable VPN temporarily
# Check if you're behind institutional proxy
env | grep -i proxy
```

---

## What You Need to Run BayesFlow (Minimum)

Based on what's already installed, you need:

1. ✅ **PyTorch** 2.7.1+cu118 - **ALREADY INSTALLED**
2. ⚠️ **NumPy** - Need version <2.0 (you have 2.3.5)
3. ❌ **BayesFlow** - Main package (need to install)
4. ❌ **Keras** - Neural networks (need to install)
5. ⚠️ **scipy, pandas, matplotlib** - May be installed

---

## Immediate Action Plan

### Step 1: Wait for Network or Fix Connectivity

```bash
# Keep checking until this works
ping -c 3 pypi.org
```

### Step 2: Once Network Works

```bash
# Activate environment
cd "/media/mohammad/New Volume/DoctoralSharif/Articles/Matin/train_joint_models"
source bayesflow_env/bin/activate

# Install critical packages
pip install "numpy==1.26.4" --force-reinstall
pip install bayesflow keras

# Test import
python -c "import bayesflow as bf; print('BayesFlow version:', bf.__version__)"
```

### Step 3: Run Test

```bash
python test_setup.py
```

---

## Alternative: Use Pre-configured Cloud Environment

If network issues persist, consider:

1. **Google Colab** (free GPU, pre-installed packages)
   - Upload your code
   - Install only BayesFlow: `!pip install bayesflow`
   - Everything else pre-installed

2. **University Cluster** (if available)
   - Often has better network
   - May have packages pre-installed

3. **Docker Container** (advanced)
   - Use pre-built container with all packages

---

## Check Current Status

Run this to see what you already have:

```bash
source bayesflow_env/bin/activate

python << 'EOF'
import sys
print(f"Python: {sys.version}")

packages = ['torch', 'numpy', 'scipy', 'pandas', 'matplotlib', 'keras']
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {pkg}: {version}")
    except ImportError:
        print(f"✗ {pkg}: NOT INSTALLED")

print("\nTrying BayesFlow...")
try:
    import bayesflow as bf
    print(f"✓ bayesflow: {bf.__version__}")
except ImportError:
    print("✗ bayesflow: NOT INSTALLED (this is expected)")
EOF
```

---

## Summary

**Your Problem**: Network connectivity, NOT package compatibility

**Your System**: 
- ✅ Python 3.12.3
- ✅ NVIDIA GPU (CUDA 13.0/Driver 580.95.05)
- ✅ PyTorch 2.7.1+cu118 already installed

**What's Missing**:
1. BayesFlow (need internet to download)
2. Proper NumPy version (need to downgrade from 2.3.5 to 1.26.4)
3. Keras (need internet to download)

**Next Steps**:
1. Fix network connectivity (check DNS, proxy, firewall)
2. OR use offline installation method
3. OR try conda instead of pip
4. Once packages are installed, run `python test_setup.py`

---

## Contact IT Support

If this is a university/company computer, contact IT about:
- PyPI access (pypi.org)
- DNS configuration
- Proxy settings for pip
- Firewall rules for HTTPS (port 443)

Show them this error message:
```
NewConnectionError: Failed to establish a new connection: [Errno -2] Name or service not known
```
