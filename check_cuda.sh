#!/bin/bash
# Check CUDA and NVIDIA driver compatibility before setup
# Run: bash check_cuda.sh

echo "=========================================="
echo "CUDA and Driver Compatibility Check"
echo "=========================================="
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠ nvidia-smi not found"
    echo "Either:"
    echo "  1. No NVIDIA GPU present (will use CPU)"
    echo "  2. NVIDIA drivers not installed"
    echo ""
    echo "For CPU-only setup, this is fine."
    echo "For GPU setup, install NVIDIA drivers first."
    exit 0
fi

echo "✓ NVIDIA drivers detected"
echo ""

# Get driver version
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
echo "Driver Version: $DRIVER_VERSION"

# Get CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo "CUDA Version (from driver): $CUDA_VERSION"
echo ""

# Get GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Check CUDA major version
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)

echo "=========================================="
echo "Compatibility Assessment"
echo "=========================================="
echo ""

# Provide PyTorch compatibility info
if [ "$CUDA_MAJOR" -ge 12 ]; then
    echo "✓ CUDA $CUDA_VERSION detected"
    echo ""
    echo "PyTorch Installation Options:"
    echo ""
    echo "Option 1 - CUDA 12.1 (Recommended):"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    echo ""
    echo "Option 2 - CUDA 11.8 (Stable, works with CUDA 12+ via compatibility):"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    echo ""
elif [ "$CUDA_MAJOR" -eq 11 ]; then
    echo "✓ CUDA $CUDA_VERSION detected"
    echo ""
    echo "PyTorch Installation:"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    echo ""
else
    echo "⚠ CUDA $CUDA_VERSION detected"
    echo ""
    echo "PyTorch may have limited support for this CUDA version."
    echo "Consider using CPU version:"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
    echo ""
fi

echo "=========================================="
echo "Python Version Check"
echo "=========================================="
echo ""

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "Python Version: $PYTHON_VERSION"
echo ""

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -eq 12 ]; then
    echo "⚠ Python 3.12 detected"
    echo ""
    echo "Important Notes:"
    echo "  - PyTorch: ✓ Fully supported"
    echo "  - BayesFlow: ✓ Should work"
    echo "  - Keras with PyTorch backend: ✓ Supported"
    echo "  - TensorFlow: ✗ Not available for Python 3.12"
    echo ""
    echo "Recommendation: Use PyTorch backend (TensorFlow not needed)"
    echo ""
elif [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ] && [ "$PYTHON_MINOR" -le 11 ]; then
    echo "✓ Python $PYTHON_VERSION is compatible with all packages"
    echo ""
else
    echo "⚠ Python $PYTHON_VERSION compatibility may vary"
    echo "   Recommended: Python 3.10 or 3.11"
    echo ""
fi

echo "=========================================="
echo "Recommendations"
echo "=========================================="
echo ""
echo "Based on your system:"
echo ""
echo "1. CUDA: Use CUDA 12.1 or 11.8 build of PyTorch"
echo "2. Python 3.12: Skip TensorFlow, use PyTorch backend for Keras"
echo "3. Proceed with: bash setup_environment.sh"
echo ""
echo "The setup script will automatically detect your GPU."
echo ""
