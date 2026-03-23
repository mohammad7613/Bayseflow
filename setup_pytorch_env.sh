#!/bin/bash

# =============================================================================
# BayesFlow Environment Setup Script (PyTorch Backend)
# =============================================================================
# This script creates a Python virtual environment with PyTorch backend
# Compatible with Python 3.8-3.12 (NO TensorFlow required)
# =============================================================================

set -e  # Exit on error

echo "========================================"
echo "BayesFlow Setup (PyTorch Backend)"
echo "========================================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Check if nvidia-smi exists (GPU check)
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
    echo "✓ NVIDIA Driver: $CUDA_VERSION"
    
    # Determine CUDA version for PyTorch
    DRIVER_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)
    if [ "$DRIVER_MAJOR" -ge 520 ]; then
        PYTORCH_CUDA="cu121"
        echo "  → Using PyTorch with CUDA 12.1"
    else
        PYTORCH_CUDA="cu118"
        echo "  → Using PyTorch with CUDA 11.8"
    fi
else
    echo "⚠ No GPU detected, will use CPU-only PyTorch"
    PYTORCH_CUDA="cpu"
fi

echo ""
echo "[1/6] Creating virtual environment..."
python3 -m venv bayesflow_env

echo "[2/6] Activating virtual environment..."
source bayesflow_env/bin/activate

echo "[3/6] Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "[4/6] Installing PyTorch..."
if [ "$PYTORCH_CUDA" = "cpu" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}
fi

echo "[5/6] Verifying PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo "[6/6] Installing remaining dependencies..."
pip install -r requirements_pytorch.txt

echo ""
echo "========================================"
echo "✓ Installation Complete!"
echo "========================================"
echo ""
echo "To activate the environment:"
echo "  source bayesflow_env/bin/activate"
echo ""
echo "Set Keras backend to PyTorch:"
echo "  export KERAS_BACKEND=torch"
echo ""
echo "Or add to your ~/.bashrc:"
echo "  echo 'export KERAS_BACKEND=torch' >> ~/.bashrc"
echo ""
echo "To verify installation:"
echo "  python3 test_setup.py"
echo ""
