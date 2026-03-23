#!/bin/bash
# Minimal installation check - works with existing PyTorch
# Run this to see what you already have and what's missing

echo "=========================================="
echo "BayesFlow Environment Status Check"
echo "=========================================="
echo ""

# Check if virtual environment exists and activate it
if [ ! -d "bayesflow_env" ]; then
    echo "✗ Virtual environment not found!"
    echo "Please create it first: python3 -m venv bayesflow_env"
    exit 1
fi

source bayesflow_env/bin/activate

echo "[1/3] Checking Python and system..."
echo "  Python: $(python --version)"
echo "  pip: $(pip --version | awk '{print $2}')"
echo ""

echo "[2/3] Checking installed packages..."
echo ""

python << 'PYEOF'
import sys

# Check critical packages
packages_check = {
    'torch': {'min_version': '2.0.0', 'critical': True},
    'numpy': {'min_version': '2.2.6', 'critical': True},
    'scipy': {'min_version': '1.10.0', 'critical': False},
    'pandas': {'min_version': '2.0.0', 'critical': False},
    'matplotlib': {'min_version': '3.7.0', 'critical': False},
    'bayesflow': {'min_version': '2.0.0', 'critical': True}
}
# Note: Keras is NOT checked - BayesFlow 2.0+ bundles it internally
# Note: BayesFlow 2.0.8+ requires NumPy >= 2.2.6 (changed from older versions!)

installed = []
missing_critical = []
missing_optional = []
version_issues = []

for pkg_name, requirements in packages_check.items():
    try:
        mod = __import__(pkg_name)
        version = getattr(mod, '__version__', 'unknown')
        
        status = '✓'
        if pkg_name == 'numpy':
            try:
                major, minor, _ = version.split('.')[:3]
                if int(major) >= 2:
                    status = '⚠'
                    version_issues.append(f"{pkg_name}: version {version} (need <2.0.0)")
            except:
                pass
        
        installed.append(f"{status} {pkg_name}: {version}")
        
    except ImportError:
        if requirements['critical']:
            missing_critical.append(pkg_name)
            installed.append(f"✗ {pkg_name}: NOT INSTALLED (CRITICAL)")
        else:
            missing_optional.append(pkg_name)
            installed.append(f"⚠ {pkg_name}: not installed (optional)")

# Print results
for item in installed:
    print(f"  {item}")

print("\n" + "="*50)
print("Summary")
print("="*50)

if not missing_critical and not version_issues:
    print("✓ All critical packages installed and compatible!")
elif missing_critical:
    print(f"✗ Missing CRITICAL packages: {', '.join(missing_critical)}")
if version_issues:
    print("⚠ Version issues detected:")
    for issue in version_issues:
        print(f"  - {issue}")
if missing_optional:
    print(f"⚠ Optional packages missing: {', '.join(missing_optional)}")

# Check PyTorch CUDA
if 'torch' in [p.split(':')[0].strip('✓ ') for p in installed if '✓' in p]:
    print("\n" + "="*50)
    print("PyTorch CUDA Status")
    print("="*50)
    import torch
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"  GPU 0: {torch.cuda.get_device_name(0)}")

# Provide next steps
print("\n" + "="*50)
print("Next Steps")
print("="*50)

if missing_critical:
    print("\n1. FIX NETWORK CONNECTION FIRST")
    print("   Test: ping -c 3 pypi.org")
    print("\n2. Once network works, install missing packages:")
    for pkg in missing_critical:
        if pkg == 'numpy':
            print(f"   pip install 'numpy<2.0.0' --force-reinstall")
        else:
            print(f"   pip install {pkg}")
    print("\n3. Then run: python test_setup.py")
    
elif version_issues:
    print("\n1. Fix version issues:")
    print("   pip install 'numpy<2.0.0' --force-reinstall")
    print("\n2. Then run: python test_setup.py")
    
else:
    print("\n✓ Ready to use!")
    print("   Run: python test_setup.py")

print("\n" + "="*50)

PYEOF

echo ""
echo "[3/3] Testing network connectivity..."
if ping -c 1 -W 2 pypi.org > /dev/null 2>&1; then
    echo "  ✓ Network: PyPI is reachable"
    echo ""
    echo "You can now install missing packages!"
else
    echo "  ✗ Network: Cannot reach PyPI"
    echo ""
    echo "NETWORK ISSUE DETECTED!"
    echo "See NETWORK_TROUBLESHOOTING.md for solutions"
    echo ""
    echo "Quick fixes to try:"
    echo "  1. Check VPN/proxy settings"
    echo "  2. Try: sudo systemctl restart NetworkManager"
    echo "  3. Check firewall: sudo ufw status"
    echo "  4. Use alternative mirror (when network works):"
    echo "     pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple bayesflow"
fi

echo ""
echo "=========================================="
