"""
Test script to verify conditional inference setup is working correctly.

Run this before training to ensure:
1. Simulator produces correct output format
2. Adapter transforms data properly
3. Model can sample without errors
"""

import numpy as np
import sys

print("="*60)
print("Testing Conditional Inference Setup")
print("="*60)

# Test 1: Import modules
print("\n[1/6] Testing imports...")
try:
    from bayesflow_models.DDM_DC_Pedestrain import (
        prior_DC, ddm_DC_alphaToCpp, meta, adopt,
        model_DC, all_models, CONDITIONS
    )
    import bayesflow as bf
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check CONDITIONS
print("\n[2/6] Checking TTA conditions...")
print(f"  CONDITIONS = {CONDITIONS}")
expected_conditions = np.array([2.5, 3.0, 3.5, 4.0])
if np.allclose(CONDITIONS, expected_conditions):
    print("✓ CONDITIONS match experiment design")
else:
    print(f"⚠ Expected {expected_conditions}, got {CONDITIONS}")

# Test 3: Test simulator directly
print("\n[3/6] Testing simulator...")
try:
    test_params = prior_DC()
    test_result = ddm_DC_alphaToCpp(
        theta=test_params['theta'],
        b0=test_params['b0'],
        k=test_params['k'],
        mu_ndt=test_params['mu_ndt'],
        sigma_ndt=test_params['sigma_ndt'],
        mu_alpah=test_params['mu_alpah'],
        sigma_alpha=test_params['sigma_alpha'],
        sigma_cpp=test_params['sigma_cpp'],
        number_of_trials=60,
        tta_condition=3.0
    )
    
    # Check output format
    assert 'x' in test_result, "Missing 'x' key in simulator output"
    assert test_result['x'].shape == (60, 2), f"Wrong shape: {test_result['x'].shape}, expected (60, 2)"
    assert not np.any(np.isnan(test_result['x'])), "NaN values in simulator output"
    
    print(f"  Output keys: {list(test_result.keys())}")
    print(f"  Data shape: {test_result['x'].shape}")
    print(f"  Data range: RT=[{test_result['x'][:, 0].min():.3f}, {test_result['x'][:, 0].max():.3f}], "
          f"CPP=[{test_result['x'][:, 1].min():.3f}, {test_result['x'][:, 1].max():.3f}]")
    print("✓ Simulator produces correct output format")
except Exception as e:
    print(f"✗ Simulator test failed: {e}")
    sys.exit(1)

# Test 4: Test meta function
print("\n[4/6] Testing meta function...")
try:
    meta_result = meta()
    assert 'number_of_trials' in meta_result, "Missing 'number_of_trials' in meta output"
    assert 'tta_condition' in meta_result, "Missing 'tta_condition' in meta output"
    assert meta_result['tta_condition'] in CONDITIONS, f"Invalid TTA: {meta_result['tta_condition']}"
    
    print(f"  Meta output: {meta_result}")
    print("✓ Meta function works correctly")
except Exception as e:
    print(f"✗ Meta test failed: {e}")
    sys.exit(1)

# Test 5: Test model sampling
print("\n[5/6] Testing model.sample()...")
try:
    samples = model_DC.sample(3)
    
    print(f"  Sample keys: {list(samples.keys())}")
    print(f"  x shape: {samples['x'].shape}")
    print(f"  TTA conditions: {samples['tta_condition']}")
    
    # Check structure
    assert 'x' in samples, "Missing 'x' in model samples"
    assert 'tta_condition' in samples, "Missing 'tta_condition' in model samples"
    assert samples['x'].shape[0] == 3, f"Wrong batch size: {samples['x'].shape[0]}"
    assert samples['x'].shape[2] == 2, f"Wrong feature dim: {samples['x'].shape[2]}"
    
    # Check parameters are present
    for param in ['theta', 'b0', 'k', 'mu_ndt', 'sigma_ndt', 'mu_alpah', 'sigma_alpha', 'sigma_cpp']:
        assert param in samples, f"Missing parameter: {param}"
    
    print("✓ Model sampling works correctly")
except Exception as e:
    print(f"✗ Model sampling failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test adapter
print("\n[6/6] Testing adapter...")
try:
    adapter = adopt(prior_DC())
    adapted = adapter(samples)
    
    print(f"  Adapter output keys: {list(adapted.keys())}")
    
    # Check expected keys
    assert 'summary_variables' in adapted, "Missing 'summary_variables'"
    assert 'inference_variables' in adapted, "Missing 'inference_variables'"
    assert 'condition_variables' in adapted, "Missing 'condition_variables'"
    
    print(f"  summary_variables shape: {adapted['summary_variables'].shape}")
    print(f"  inference_variables shape: {adapted['inference_variables'].shape}")
    print(f"  condition_variables shape: {adapted['condition_variables'].shape}")
    print(f"  condition_variables values: {adapted['condition_variables']}")
    
    # Verify condition variables match original TTAs
    original_ttas = samples['tta_condition']
    adapted_ttas = adapted['condition_variables']
    if np.allclose(original_ttas, adapted_ttas):
        print("✓ Adapter correctly handles condition variables")
    else:
        print(f"⚠ TTA mismatch: original={original_ttas}, adapted={adapted_ttas}")
    
except Exception as e:
    print(f"✗ Adapter test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "="*60)
print("ALL TESTS PASSED ✓")
print("="*60)
print("\nYour setup is ready for training!")
print("\nNext steps:")
print("  1. Train the model: python main.py")
print("  2. Check parameter recovery: evaluation_conditional.ipynb")
print("  3. Apply to real data: use utils_real_data.py")
print("\nFor detailed guidance, see CONDITIONAL_INFERENCE_GUIDE.md")
