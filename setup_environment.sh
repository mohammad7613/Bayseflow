#!/bin/bash
# Automated setup script for BayesFlow environment
# Run: bash setup_environment.sh

echo "=========================================="
echo "BayesFlow Environment Setup"
echo "=========================================="
echo ""

# Check Python version
echo "[1/7] Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "✗ Python 3.8+ required, found: $PYTHON_VERSION"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if Python 3.12+ (needs different TensorFlow version)
if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 12 ]; then
    echo "✓ Python $PYTHON_VERSION detected (TensorFlow 2.16+ will be used)"
else
    echo "✓ Python $PYTHON_VERSION detected"
fi
echo ""

# Create virtual environment
echo "[2/7] Creating virtual environment..."
if [ -d "bayesflow_env" ]; then
    echo "⚠ Virtual environment already exists"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf bayesflow_env
        python3 -m venv bayesflow_env
        echo "✓ Virtual environment recreated"
    else
        echo "Using existing environment"
    fi
else
    python3 -m venv bayesflow_env
    echo "✓ Virtual environment created"
fi
echo ""

# Activate environment
echo "[3/7] Activating environment..."
source bayesflow_env/bin/activate

if [ $? -ne 0 ]; then
    echo "✗ Failed to activate environment"
    exit 1
fi

echo "✓ Environment activated"
echo ""

# Upgrade pip
echo "[4/7] Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet
echo "✓ pip upgraded"
echo ""

# Detect GPU
echo "[5/7] Detecting hardware..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    GPU_AVAILABLE=true
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "  CUDA Version: $CUDA_VERSION"
else
    echo "⚠ No NVIDIA GPU detected (will use CPU)"
    GPU_AVAILABLE=false
fi
echo ""

# Install PyTorch
echo "[6/7] Installing PyTorch..."
if [ "$GPU_AVAILABLE" = true ]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
else
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
fi
echo "✓ PyTorch installed"
echo ""

# Install remaining dependencies
echo "[7/7] Installing remaining dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "✗ Failed to install dependencies"
    echo ""
    echo "This might be due to:"
    echo "  1. Network connectivity issues (try again)"
    echo "  2. Python version compatibility (you have Python $PYTHON_VERSION)"
    echo "  3. Missing system packages"
    echo ""
    echo "Try running manually:"
    echo "  source bayesflow_env/bin/activate"
    echo "  pip install -r requirements.txt"
    echo ""
    exit 1
fi

echo "✓ All dependencies installed"
echo ""

# Verify installation
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="
echo ""

python test_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Setup Complete!"
    echo "=========================================="
    echo ""
    echo "To activate the environment in the future:"
    echo "  source bayesflow_env/bin/activate"
    echo ""
    echo "To start training:"
    echo "  python main.py"
    echo ""
    echo "To use Jupyter notebooks:"
    echo "  jupyter notebook"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "⚠ Setup completed with warnings"
    echo "=========================================="
    echo ""
    echo "Some tests failed. Check the output above."
    echo "You may need to:"
    echo "  1. Install missing system dependencies"
    echo "  2. Check Python version compatibility"
    echo "  3. Review ENVIRONMENT_SETUP.md for troubleshooting"
    echo ""
fi
