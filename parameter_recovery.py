#!/usr/bin/env python3
"""
Parameter Recovery Test for DDM_DC Model
========================================

This script validates the Bayesflow posterior network's ability to recover
ground-truth parameters from synthetic data. This is Phase 1 of the inference pipeline.

Key Points:
- Tests conditional inference: p(θ | data, TTA)
- Generates synthetic data from known parameter values
- Trains posterior network on synthetic data
- Evaluates recovery accuracy (bias, credible intervals, efficiency)
- Produces diagnostic plots

Usage:
    python parameter_recovery_test.py --n_test_params 50 --epochs 5
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
import bayesflow as bf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import pandas as pd
from time import time
from pathlib import Path
import argparse

from bayesflow_models.DDM_DC_Pedestrain import (
    prior_DC,
    ddm_DC_alphaToCpp,
    meta,
    adopt,
    CONDITIONS,
)

# ============================================================================
# Configuration
# ============================================================================

class RecoveryConfig:
    """Configuration for parameter recovery experiments."""
    
    # Synthetic training
    n_simulations_train = 10000  # Simulations for network training
    epochs_train = 10
    batch_size = 32
    
    # Test/recovery evaluation
    n_test_params = 50  # True parameter sets to test
    n_posterior_samples = 5000  # Posterior samples per test
    
    # Network architecture
    summary_dim = 15
    num_coupling_layers = 6
    
    # Save paths
    results_dir = Path("results/parameter_recovery")
    checkpoint_dir = Path("trained_model/parameter_recovery_checkpoints")
    
    def __post_init_setup__(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Core Recovery Functions
# ============================================================================

def generate_ground_truth_parameters(n_samples: int, rng=None) -> dict:
    """
    Generate ground-truth parameter sets from the prior.
    
    Args:
        n_samples: Number of parameter sets to generate
        rng: Random number generator (default: use prior_DC default)
    
    Returns:
        Dictionary with keys = parameter names, values = (n_samples,) arrays
    """
    ground_truth_params = {}
    
    for i in range(n_samples):
        sample = prior_DC()
        for param_name, param_value in sample.items():
            if i == 0:
                ground_truth_params[param_name] = []
            ground_truth_params[param_name].append(param_value)
    
    # Convert to numpy arrays
    ground_truth_params = {
        k: np.array(v) for k, v in ground_truth_params.items()
    }
    
    return ground_truth_params


def generate_synthetic_data_for_ground_truth(
    ground_truth_params: dict,
    num_trials_per_condition: int = 60,
) -> dict:
    """
    Generate synthetic data for each ground-truth parameter set.
    
    For each parameter set, we generate data for EACH TTA condition separately
    (matching the experimental design).
    
    Args:
        ground_truth_params: Dict of parameter arrays, shape (n_params,)
        num_trials_per_condition: Trials per TTA condition
    
    Returns:
        Dictionary indexed by (param_idx, tta_str):
            {
                'data': (num_trials, 2) array,          # [RT, CPP]
                'tta_condition': scalar,                 # TTA value
                'param_set_idx': int,
            }
    """
    n_params = len(list(ground_truth_params.values())[0])
    synthetic_data = {}
    
    param_names = sorted(ground_truth_params.keys())
    
    for param_idx in range(n_params):
        # Extract parameters for this set
        theta_dict = {name: ground_truth_params[name][param_idx] 
                      for name in param_names}
        
        for tta in CONDITIONS:
            # Generate synthetic data for this param set × TTA combination
            output = ddm_DC_alphaToCpp(
                **theta_dict,
                number_of_trials=num_trials_per_condition,
                tta_condition=tta,
                dt=0.005,
            )
            
            key = (param_idx, f"tta_{tta:.1f}")
            synthetic_data[key] = {
                'data': output['x'],
                'tta_condition': tta,
                'param_set_idx': param_idx,
            }
    
    return synthetic_data


def aggregate_data_by_param_set(
    synthetic_data: dict,
    combine_all_ttas: bool = False,
) -> dict:
    """
    Organize synthetic data for inference.
    
    Args:
        synthetic_data: Dict indexed by (param_idx, tta_str)
        combine_all_ttas: If True, concatenate all TTA data for each param set.
                          If False, keep separate by TTA.
    
    Returns:
        Dictionary indexed by param_set_idx with organized data.
    """
    aggregated = {}
    
    # Get unique param indices
    param_indices = sorted(set(key[0] for key in synthetic_data.keys()))
    
    for param_idx in param_indices:
        if combine_all_ttas:
            # Concatenate data from all TTAs
            all_data = []
            all_ttas = []
            for key in sorted(synthetic_data.keys()):
                if key[0] == param_idx:
                    all_data.append(synthetic_data[key]['data'])
                    all_ttas.append(synthetic_data[key]['tta_condition'])
            
            aggregated[param_idx] = {
                'data': np.vstack(all_data),  # shape (240, 2) for 4 TTAs×60 trials
                'tta_conditions': np.array(all_ttas),  # Repeat each 60 times
            }
        else:
            # Keep separate by TTA
            aggregated[param_idx] = {}
            for key in sorted(synthetic_data.keys()):
                if key[0] == param_idx:
                    _, tta_str = key
                    aggregated[param_idx][tta_str] = synthetic_data[key]['data']
    
    return aggregated


def validate_synthetic_data(synthetic_data: dict, ground_truth_params: dict):
    """
    Validate synthetic data dimensions and ranges.
    
    Args:
        synthetic_data: Generated synthetic data dict
        ground_truth_params: Ground truth parameter dict
    """
    print("\n" + "="*70)
    print("SYNTHETIC DATA VALIDATION")
    print("="*70)
    
    # Check data shapes
    sample_key = list(synthetic_data.keys())[0]
    data_shape = synthetic_data[sample_key]['data'].shape
    print(f"\nData shape per (param_set, TTA): {data_shape}")
    assert data_shape[1] == 2, "Data should have 2 columns (RT, CPP)"
    assert data_shape[0] > 0, "Should have at least 1 trial"
    
    # Check parameter ranges
    print(f"\nParameter ranges (from ground truth):")
    for param_name, values in ground_truth_params.items():
        print(f"  {param_name:15s}: [{values.min():.4f}, {values.max():.4f}]")
    
    # Check data ranges
    all_rts = []
    all_cpps = []
    for data_dict in synthetic_data.values():
        all_rts.append(data_dict['data'][:, 0])
        all_cpps.append(data_dict['data'][:, 1])
    
    all_rts = np.concatenate(all_rts)
    all_cpps = np.concatenate(all_cpps)
    
    print(f"\nResponse time ranges:")
    print(f"  RT: [{all_rts.min():.4f}, {all_rts.max():.4f}]")
    print(f"  CPP: [{all_cpps.min():.4f}, {all_cpps.max():.4f}]")
    
    print(f"\nTotal synthetic data points: {len(all_rts)}")
    print(f"TTA conditions: {CONDITIONS}")
    print("✓ Synthetic data validation passed\n")


def build_posterior_network(n_dims: int = 8, summary_dim: int = 15) -> bf.networks.CouplingFlow:
    """
    Build the posterior network for conditional inference.
    
    Args:
        n_dims: Number of parameters (8 for DDM_DC)
        summary_dim: Output dimension of summary network
    
    Returns:
        CouplingFlow network for p(θ | data, TTA)
    """
    inference_network = bf.networks.CouplingFlow(
        num_dimensions=n_dims,
        conditional_shape=(1,),  # TTA is 1-d condition
        num_coupling_layers=6,
        num_dense_layers=3,
        activation='relu',
        num_mixture_components=3,
    )
    
    return inference_network


def build_summary_network(summary_dim: int = 15) -> bf.networks.SetTransformer:
    """
    Build the summary network for data compression.
    
    Uses SetTransformer to handle variable-length trial data.
    
    Args:
        summary_dim: Output dimension
    
    Returns:
        SetTransformer network
    """
    summary_network = bf.networks.SetTransformer(
        summary_dim=summary_dim,
        num_blocks=3,
        num_heads=4,
    )
    
    return summary_network


def train_posterior_network(
    model,
    n_simulations: int = 10000,
    epochs: int = 10,
    batch_size: int = 32,
    device=None,
    verbose: bool = True,
):
    """
    Train the posterior network on synthetic data.
    
    Args:
        model: Tuple of (simulator, adapter)
        n_simulations: Total simulations for training
        epochs: Number of full passes through training data
        batch_size: Batch size for gradient updates
        device: Torch device (default: auto-detect)
        verbose: Print progress
    
    Returns:
        Tuple of (workflow, history)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"POSTERIOR NETWORK TRAINING")
        print(f"{'='*70}")
        print(f"Device: {device}")
        print(f"Total simulations: {n_simulations}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Batches per epoch: {n_simulations // epochs // batch_size}")
    
    simulator, adapter = model
    
    # Build networks
    summary_net = build_summary_network(summary_dim=15)
    inference_net = build_posterior_network(n_dims=8, summary_dim=15)
    
    # Create workflow
    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inverse_transform_net=inference_net,
        summary_net=summary_net,
    )
    
    # Custom optimizer
    optimizer = torch.optim.AdamW(
        workflow.approximator.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    workflow.optimizer = optimizer
    
    # Training
    start_time = time()
    
    history = workflow.fit_online(
        epochs=epochs,
        batch_size=batch_size,
        num_batches_per_epoch=n_simulations // epochs // batch_size,
        max_queue_size=100,
        verbose=verbose,
    )
    
    elapsed_time = time() - start_time
    
    if verbose:
        print(f"\n✓ Training completed in {elapsed_time:.1f}s")
        print(f"  Mean loss (last 10 batches): {history.loss[-10:].mean():.4f}")
    
    return workflow, history


def posterior_sample_from_synthetic_data(
    workflow,
    ground_truth_params: dict,
    synthetic_data: dict,
    n_posterior_samples: int = 5000,
) -> dict:
    """
    Draw posterior samples from the trained network.
    
    For each ground-truth parameter set with synthetic data,
    generate posterior distribution samples.
    
    Args:
        workflow: Trained BayesFlow workflow
        ground_truth_params: Ground truth parameter dict
        synthetic_data: Synthetic data dict indexed by (param_idx, tta_str)
        n_posterior_samples: Number of posterior samples to draw
    
    Returns:
        Dictionary indexed by param_idx with posterior samples and diagnostics
    """
    n_params = len(list(ground_truth_params.values())[0])
    posterior_samples = {}
    
    for param_idx in range(n_params):
        # For this param set, we test on ONE randomly selected TTA
        # (In practice, you could test on all or average across TTAs)
        test_tta = np.random.choice(CONDITIONS)
        key = (param_idx, f"tta_{test_tta:.1f}")
        
        test_data = synthetic_data[key]
        
        # Format data for network
        summary_vars = test_data['data'].astype(np.float32)  # (num_trials, 2)
        condition_vars = np.array([test_tta], dtype=np.float32)  # (1,)
        
        # Get inference variables (ground truth for this set)
        param_names = sorted(ground_truth_params.keys())
        inference_vars = np.array(
            [ground_truth_params[name][param_idx] for name in param_names],
            dtype=np.float32
        )
        
        # Prepare input dict for network
        input_dict = {
            'summary_variables': summary_vars[np.newaxis, ...],  # Add batch dim
            'condition_variables': condition_vars[np.newaxis, ...],
        }
        
        # Generate posterior samples
        posterior_dist = workflow.approximator(
            torch.from_numpy(input_dict['summary_variables']).to(workflow.device),
            torch.from_numpy(input_dict['condition_variables']).to(workflow.device),
        )
        
        samples = posterior_dist.sample((n_posterior_samples,)).cpu().detach().numpy()
        
        posterior_samples[param_idx] = {
            'samples': samples,  # shape (n_posterior_samples, 8)
            'ground_truth': inference_vars,
            'tta_condition': test_tta,
            'test_data': test_data['data'],
        }
    
    return posterior_samples


def compute_recovery_diagnostics(posterior_samples: dict, param_names: list) -> pd.DataFrame:
    """
    Compute parameter recovery statistics.
    
    Args:
        posterior_samples: Dict indexed by param_idx with posterior samples
        param_names: List of parameter names
    
    Returns:
        DataFrame with recovery diagnostics
    """
    diagnostics = []
    
    for param_idx, posterior_dict in posterior_samples.items():
        samples = posterior_dict['samples']  # (n_samples, 8)
        ground_truth = posterior_dict['ground_truth']  # (8,)
        
        for param_i, param_name in enumerate(param_names):
            param_samples = samples[:, param_i]
            param_truth = ground_truth[param_i]
            
            # Compute statistics
            posterior_mean = param_samples.mean()
            posterior_std = param_samples.std()
            posterior_median = np.median(param_samples)
            
            # Quantiles for credible interval
            hpd_lower = np.percentile(param_samples, 2.5)
            hpd_upper = np.percentile(param_samples, 97.5)
            
            # Recovery metrics
            bias = posterior_mean - param_truth
            rmse = np.sqrt((posterior_mean - param_truth) ** 2)
            includes_truth = (hpd_lower <= param_truth <= hpd_upper)
            
            diagnostics.append({
                'param_set_idx': param_idx,
                'parameter': param_name,
                'ground_truth': param_truth,
                'posterior_mean': posterior_mean,
                'posterior_std': posterior_std,
                'posterior_median': posterior_median,
                'hpd_lower': hpd_lower,
                'hpd_upper': hpd_upper,
                'bias': bias,
                'rmse': rmse,
                'includes_truth': includes_truth,
            })
    
    return pd.DataFrame(diagnostics)


def plot_recovery_diagnostics(diagnostics_df: pd.DataFrame, param_names: list, save_path: Path):
    """
    Create diagnostic plots for parameter recovery.
    
    Plots include:
    1. Recovery scatter plots (ground truth vs. posterior mean)
    2. Bias analysis
    3. Coverage (credible interval inclusion)
    4. Efficiency (posterior standard deviation)
    
    Args:
        diagnostics_df: DataFrame from compute_recovery_diagnostics
        param_names: List of parameter names
        save_path: Where to save figure
    """
    n_params = len(param_names)
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, n_params, figure=fig, hspace=0.35, wspace=0.3)
    
    for param_i, param_name in enumerate(param_names):
        param_data = diagnostics_df[diagnostics_df['parameter'] == param_name]
        
        # Row 1: Recovery scatter plot
        ax = fig.add_subplot(gs[0, param_i])
        gt = param_data['ground_truth'].values
        pm = param_data['posterior_mean'].values
        ps = param_data['posterior_std'].values
        
        ax.errorbar(gt, pm, yerr=1.96*ps, fmt='o', alpha=0.6, markersize=6)
        
        # Add diagonal line (perfect recovery)
        lims = [min(gt.min(), pm.min()), max(gt.max(), pm.max())]
        ax.plot(lims, lims, 'r--', lw=2, alpha=0.5, label='Perfect recovery')
        ax.set_xlabel('Ground Truth', fontsize=9)
        ax.set_ylabel('Posterior Mean ± 1.96 SD', fontsize=9)
        ax.set_title(param_name, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Row 2: Bias
        ax = fig.add_subplot(gs[1, param_i])
        bias = param_data['bias'].values
        ax.hist(bias, bins=15, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', lw=2, label='Zero bias')
        ax.set_xlabel('Bias (Posterior - Truth)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title(f'{param_name} Bias', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Row 3: Coverage
        ax = fig.add_subplot(gs[2, param_i])
        coverage = param_data['includes_truth'].values
        coverage_pct = coverage.mean() * 100
        
        covers = coverage.sum()
        doesnt_cover = (~coverage).sum()
        
        bars = ax.bar(['Includes\nTruth', 'Misses\nTruth'], 
                      [covers, doesnt_cover],
                      color=['green', 'red'], alpha=0.7, edgecolor='black')
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title(f'{param_name} Coverage: {coverage_pct:.1f}%', fontsize=10)
        ax.set_ylim(0, len(param_data) * 1.1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Recovery diagnostics plot saved: {save_path}")
    plt.close()


def plot_training_loss(history, save_path: Path):
    """Plot training loss over epochs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    loss = history.loss
    ax.plot(loss, lw=2, alpha=0.7, label='Training loss')
    ax.fill_between(range(len(loss)), loss, alpha=0.3)
    
    ax.set_xlabel('Batch', fontsize=11)
    ax.set_ylabel('Loss (KL divergence)', fontsize=11)
    ax.set_title('Posterior Network Training Loss', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training loss plot saved: {save_path}")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main(args):
    """Main parameter recovery experiment."""
    
    config = RecoveryConfig()
    config.n_test_params = args.n_test_params
    config.epochs_train = args.epochs
    config.n_simulations_train = args.n_sim
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"BAYESFLOW PARAMETER RECOVERY TEST - DDM_DC MODEL")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"TTA conditions: {CONDITIONS}")
    print(f"Test parameter sets: {config.n_test_params}")
    print(f"Training simulations: {config.n_simulations_train}")
    print(f"Training epochs: {config.epochs_train}")
    
    # Step 1: Generate ground-truth parameters
    print(f"\n[1/5] Generating ground-truth parameter sets...")
    ground_truth_params = generate_ground_truth_parameters(config.n_test_params)
    
    # Step 2: Generate synthetic data
    print(f"[2/5] Generating synthetic data...")
    synthetic_data = generate_synthetic_data_for_ground_truth(
        ground_truth_params,
        num_trials_per_condition=60,
    )
    validate_synthetic_data(synthetic_data, ground_truth_params)
    
    # Step 3: Train posterior network
    print(f"[3/5] Training posterior network...")
    simulator = bf.simulators.make_simulator(
        [prior_DC, ddm_DC_alphaToCpp],
        meta_fn=meta,
    )
    adapter = adopt(prior_DC())
    model = (simulator, adapter)
    
    workflow, history = train_posterior_network(
        model,
        n_simulations=config.n_simulations_train,
        epochs=config.epochs_train,
        batch_size=config.batch_size,
        device=device,
        verbose=True,
    )
    
    # Step 4: Generate posteriors and diagnostics
    print(f"\n[4/5] Computing posterior samples and diagnostics...")
    posterior_samples = posterior_sample_from_synthetic_data(
        workflow,
        ground_truth_params,
        synthetic_data,
        n_posterior_samples=config.n_posterior_samples,
    )
    
    param_names = sorted(ground_truth_params.keys())
    diagnostics_df = compute_recovery_diagnostics(posterior_samples, param_names)
    
    # Step 5: Generate plots and summary statistics
    print(f"[5/5] Generating diagnostic plots...")
    
    # Summary statistics table
    print(f"\n{'='*70}")
    print("RECOVERY SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    summary_by_param = diagnostics_df.groupby('parameter').agg({
        'bias': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'posterior_std': 'mean',
        'includes_truth': lambda x: (x.sum() / len(x)) * 100,
    }).round(4)
    
    print(summary_by_param)
    print(f"\nOverall coverage: {diagnostics_df['includes_truth'].mean()*100:.1f}%")
    
    # Save plots
    plot_training_loss(history, config.results_dir / "training_loss.png")
    plot_recovery_diagnostics(diagnostics_df, param_names, 
                             config.results_dir / "recovery_diagnostics.png")
    
    # Save diagnostics to CSV
    diagnostics_path = config.results_dir / "recovery_diagnostics.csv"
    diagnostics_df.to_csv(diagnostics_path, index=False)
    print(f"\n✓ Diagnostics saved: {diagnostics_path}")
    
    # Save workflow
    checkpoint_path = config.checkpoint_dir / "posterior_network.pt"
    torch.save({
        'model_state_dict': workflow.approximator.state_dict(),
        'optimizer_state_dict': workflow.optimizer.state_dict() if hasattr(workflow, 'optimizer') else None,
        'config': vars(config),
    }, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    print(f"\n{'='*70}")
    print("PARAMETER RECOVERY TEST COMPLETE")
    print(f"Results saved to: {config.results_dir}")
    print(f"{'='*70}\n")
    
    return workflow, diagnostics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parameter recovery test for DDM_DC Bayesflow model"
    )
    parser.add_argument("--n_test_params", type=int, default=50,
                       help="Number of ground-truth parameter sets to test (default: 50)")
    parser.add_argument("--n_sim", type=int, default=10000,
                       help="Number of simulations for training (default: 10000)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training (default: 32)")
    
    args = parser.parse_args()
    
    workflow, diagnostics = main(args)
