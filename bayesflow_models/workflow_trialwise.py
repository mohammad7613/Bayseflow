"""
Complete Workflow for Trial-Wise DDM_DC Model Training, Recovery Analysis, and Real Data Inference

PHASES:
1. TRAINING: Train amortizers on simulated data
2. RECOVERY ANALYSIS: Evaluate parameter recovery performance
3. REAL DATA INFERENCE: Estimate 8 parameters from subject behavior (CPP, RT)
4. PIPELINE: Automated inference on new real data

Full workflow with checkpointing and GPU support.
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import bayesflow as bf
import torch
import numpy as np
import pandas as pd
from scipy import stats
from time import time
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime

from bayesflow_models.DDM_DC_Pedestrain_TrialWise import all_models
from bayesflow_models.train import train_amortizer_resume
from bayesflow_models.utils_real_data import *  # Assuming utils_real_data has subject data loading

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "training": {
        "n_sim": 100,          # Simulations per epoch
        "epochs": 1,          # Total epochs to train
        "batch_size": 32,
        "resume_epochs": 2,    # Additional epochs when resuming
    },
    "recovery": {
        "n_test_sims": 50,    # Simulations for recovery evaluation
        "n_posterior_samples": 50  # MCMC samples for posterior
    },
    "paths": {
        "checkpoints": "trained_model1/checkpoints",
        "results": "results",
        "real_data": "real_data",
        "logs": "logs"
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_directories():
    """Ensure all required directories exist"""
    for key, path in CONFIG["paths"].items():
        Path(path).mkdir(parents=True, exist_ok=True)
    print("✓ Directories setup complete")

def get_available_models():
    """List all available models"""
    print("\nAvailable Models:")
    for i, model_name in enumerate(all_models.keys(), 1):
        print(f"  {i}. {model_name}")
    return list(all_models.keys())

def log_info(message, phase="INFO"):
    """Log messages with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] [{phase}] {message}"
    print(log_msg)
    
    # Save to log file (create directory if needed)
    log_dir = CONFIG["paths"]["logs"]
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, "workflow.log")
    with open(log_file, "a") as f:
        f.write(log_msg + "\n")

# ============================================================================
# PHASE 1: TRAINING
# ============================================================================

def train_single_model(model_name, config=None, resume=False):
    """
    Train or resume training a single model.
    
    Args:
        model_name: Name of the model from all_models dict
        config: Optional config override
        resume: If True, resume from existing checkpoint
    
    Returns:
        history: Training history
    """
    if config is None:
        config = CONFIG["training"]
    
    if model_name not in all_models:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(all_models.keys())}")
    
    log_info(f"{'Resuming' if resume else 'Starting'} training for {model_name}", "TRAINING")
    
    model = all_models[model_name]
    
    start_time = time()
    epochs = config["resume_epochs"] if resume else config["epochs"]
    
    history = train_amortizer_resume(
        model=model,
        model_name=model_name,
        n_sim=config["n_sim"],
        epochs=epochs,
        batch_size=config["batch_size"],
        initial_lr=5e-4,
        checkpoint_dir=CONFIG["paths"]["checkpoints"]
    )
    
    elapsed = time() - start_time
    log_info(f"Training {model_name} completed in {elapsed:.2f}s", "TRAINING")
    
    return history

def train_all_models(config=None, resume_existing=True):
    """
    Train all models in sequence.
    
    Args:
        config: Optional config override
        resume_existing: If True, resume from checkpoint if available
    
    Returns:
        histories: Dict of model_name -> history
    """
    if config is None:
        config = CONFIG["training"]
    
    histories = {}
    model_names = list(all_models.keys())
    
    log_info(f"Training {len(model_names)} models", "TRAINING")
    
    for model_name in model_names:
        checkpoint_path = os.path.join(CONFIG["paths"]["checkpoints"], f"{model_name}.keras")
        resume = resume_existing and os.path.exists(checkpoint_path)
        
        try:
            history = train_single_model(model_name, config, resume=resume)
            histories[model_name] = history
        except Exception as e:
            log_info(f"Error training {model_name}: {str(e)}", "ERROR")
            continue
    
    log_info(f"Training complete. {len(histories)}/{len(model_names)} models trained", "TRAINING")
    return histories

# ============================================================================
# PHASE 2: RECOVERY ANALYSIS
# ============================================================================

def load_trained_model(model_name):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_name: Name of the model
    
    Returns:
        workflow: BayesFlow BasicWorkflow ready for inference
    """
    checkpoint_path = os.path.join(CONFIG["paths"]["checkpoints"], f"{model_name}.keras")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    log_info(f"Loading model {model_name} from {checkpoint_path}", "RECOVERY")
    
    model = all_models[model_name]
    simulator, adapter = model
    
    # Load the trained amortizer
    loaded_amortizer = keras.models.load_model(
        checkpoint_path,
        custom_objects={
            "SetTransformer": bf.networks.SetTransformer,
            "CouplingFlow": bf.networks.CouplingFlow,
        }
    )
    
    # Build a compatible workflow scaffold, then replace with loaded amortizer.
    # BayesFlow 2.0.8 cannot infer networks from None here.
    summary_network = bf.networks.SetTransformer(summary_dim=10)
    inference_network = bf.networks.CouplingFlow()

    workflow = bf.workflows.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=inference_network,
        summary_network=summary_network,
    )
    
    # Replace the networks with the loaded ones
    workflow.approximator = loaded_amortizer
    workflow.approximator.to(CONFIG["device"])
    
    return workflow

def run_recovery_analysis(model_name, n_test=None):
    """
    Run recovery analysis on a trained model.
    Simulates data and checks if the model can recover true parameters.
    
    Args:
        model_name: Name of the model
        n_test: Number of test simulations (uses config default if None)
    
    Returns:
        results: Dict with recovery metrics
    """
    if n_test is None:
        n_test = CONFIG["recovery"]["n_test_sims"]
    
    log_info(f"Running recovery analysis for {model_name} ({n_test} sims)", "RECOVERY")
    
    try:
        workflow = load_trained_model(model_name)
        model = all_models[model_name]
        simulator, adapter = model
        
        # Generate test data (BayesFlow 2.x simulators expose .sample()).
        if hasattr(simulator, "sample"):
            test_data = simulator.sample(batch_size=n_test)
        else:
            # Backward compatibility for callable simulators.
            test_data = simulator(batch_size=n_test)
        
        # Run posterior sampling using BayesFlow workflow API (v2.x).
        with torch.no_grad():
            posterior_dict = workflow.sample(
                num_samples=CONFIG["recovery"]["n_posterior_samples"],
                conditions=test_data,
            )

        param_names = list(posterior_dict.keys())
        posterior_samples = np.concatenate(
            [posterior_dict[name] for name in param_names], axis=-1
        )

        # Keep true parameters aligned with sampled parameter names.
        true_params = {name: test_data[name] for name in param_names if name in test_data}
        
        # Calculate recovery metrics (e.g., correlation, RMSE)
        results = {
            "model_name": model_name,
            "n_test": n_test,
            "timestamp": datetime.now().isoformat(),
            "param_names": param_names,
            "posterior_samples": posterior_samples,
            "true_params": true_params,
        }
        
        # Save results
        results_path = os.path.join(CONFIG["paths"]["results"], f"{model_name}_recovery.npz")
        arrays_to_save = {"posterior_samples": posterior_samples}
        for param_name, values in true_params.items():
            if isinstance(values, np.ndarray):
                arrays_to_save[f"true_{param_name}"] = values
        np.savez(results_path, **arrays_to_save)
        
        # Save JSON metadata
        json_results = {
            "model_name": model_name,
            "n_test": n_test,
            "timestamp": results["timestamp"],
            "posterior_shape": str(posterior_samples.shape) if hasattr(posterior_samples, 'shape') else str(type(posterior_samples))
        }
        json_path = os.path.join(CONFIG["paths"]["results"], f"{model_name}_recovery.json")
        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=2)
        
        log_info(f"Recovery analysis saved to {results_path}", "RECOVERY")
        return results
    
    except Exception as e:
        log_info(f"Error in recovery analysis: {str(e)}", "ERROR")
        raise

def run_recovery_all_models():
    """Run recovery analysis for all trained models"""
    model_names = list(all_models.keys())
    results_dict = {}
    
    for model_name in model_names:
        checkpoint_path = os.path.join(CONFIG["paths"]["checkpoints"], f"{model_name}.keras")
        if os.path.exists(checkpoint_path):
            try:
                results = run_recovery_analysis(model_name)
                results_dict[model_name] = results
            except Exception as e:
                log_info(f"Skipped recovery for {model_name}: {str(e)}", "RECOVERY")
    
    return results_dict

# ============================================================================
# PHASE 3: REAL DATA INFERENCE PIPELINE
# ============================================================================

def prepare_real_data_for_inference(subject_data, model_name):
    """
    Prepare real subject data for inference.
    
    Args:
        subject_data: Dict with 'cpp' and 'reaction_times' (or 'rt')
        model_name: Name of the model to infer with
    
    Returns:
        structured_data: Data in format expected by the adapter
    """
    # Extract behavioral data
    cpp = subject_data.get('cpp') or subject_data.get('CPP')
    rt = subject_data.get('reaction_times') or subject_data.get('rt') or subject_data.get('RT')
    
    if cpp is None or rt is None:
        raise ValueError("Subject data must contain 'cpp'/'CPP' and 'reaction_times'/'rt'/'RT'")
    
    # Stack as (n_trials, 2)
    n_trials = len(cpp)
    x = np.column_stack([rt, cpp]).astype(np.float32)
    
    # You may need to add number_of_trials and other expected inputs
    # depending on your adapter
    structured_data = {
        'x': x,
        'number_of_trials': np.array([n_trials], dtype=np.int32)
    }
    
    return structured_data

def infer_subject_parameters(subject_data, model_name, n_samples=None):
    """
    Infer the 8 DDM parameters for a single subject given their behavioral data.
    
    Args:
        subject_data: Dict with 'cpp' and 'reaction_times'
        model_name: Name of the trained model to use
        n_samples: Number of posterior samples to draw
    
    Returns:
        result: Dict with inferred parameters and statistics
    """
    if n_samples is None:
        n_samples = CONFIG["recovery"]["n_posterior_samples"]
    
    log_info(f"Inferring parameters for subject with model {model_name}", "INFERENCE")
    
    try:
        # Load model
        workflow = load_trained_model(model_name)
        
        # Prepare data
        prepared_data = prepare_real_data_for_inference(subject_data, model_name)
        
        # Run amortized inference
        with torch.no_grad():
            posterior_samples = workflow.infer_amortized(prepared_data, n_samples=n_samples)
        
        # Calculate posterior statistics
        if isinstance(posterior_samples, torch.Tensor):
            posterior_samples = posterior_samples.cpu().numpy()
        
        # Assuming 8 parameters: [v, a, t0, z, ndt, dc, sv, st]
        param_names = ['v', 'a', 't0', 'z', 'ndt', 'dc', 'sv', 'st']
        
        if posterior_samples.shape[-1] != 8:
            log_info(f"Warning: Expected 8 parameters, got {posterior_samples.shape[-1]}", "INFERENCE")
        
        result = {
            'posterior_samples': posterior_samples,
            'posterior_mean': np.mean(posterior_samples, axis=0),
            'posterior_std': np.std(posterior_samples, axis=0),
            'posterior_median': np.median(posterior_samples, axis=0),
            'param_names': param_names[:posterior_samples.shape[-1]],
            'n_trials': len(subject_data.get('cpp') or subject_data.get('CPP')),
            'timestamp': datetime.now().isoformat(),
        }
        
        log_info(f"Inference complete. Posterior shape: {result['posterior_samples'].shape}", "INFERENCE")
        return result
    
    except Exception as e:
        log_info(f"Error during inference: {str(e)}", "ERROR")
        raise

def infer_batch_subjects(subject_list, model_name):
    """
    Infer parameters for multiple subjects.
    
    Args:
        subject_list: List of dicts, each with 'cpp', 'reaction_times', 'subject_id'
        model_name: Name of the trained model
    
    Returns:
        results_df: DataFrame with inferred parameters for all subjects
    """
    all_results = []
    
    log_info(f"Starting batch inference for {len(subject_list)} subjects", "INFERENCE")
    
    for i, subject_data in enumerate(subject_list):
        subject_id = subject_data.get('subject_id', f'S{i+1}')
        
        try:
            result = infer_subject_parameters(subject_data, model_name)
            
            # Flatten results for DataFrame
            row = {'subject_id': subject_id}
            for j, param_name in enumerate(result['param_names']):
                row[f'{param_name}_mean'] = result['posterior_mean'][j]
                row[f'{param_name}_std'] = result['posterior_std'][j]
                row[f'{param_name}_median'] = result['posterior_median'][j]
            
            all_results.append(row)
            log_info(f"Completed inference for {subject_id}", "INFERENCE")
        
        except Exception as e:
            log_info(f"Failed to infer {subject_id}: {str(e)}", "ERROR")
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_path = os.path.join(CONFIG["paths"]["results"], f"batch_inference_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    results_df.to_csv(results_path, index=False)
    log_info(f"Batch results saved to {results_path}", "INFERENCE")
    
    return results_df

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def run_complete_workflow(phases=['train', 'recovery', 'inference'], models=None):
    """
    Run the complete workflow.
    
    Args:
        phases: List of phases to run: ['train', 'recovery', 'inference']
        models: List of model names to process (None = all models)
    """
    setup_directories()
    
    if models is None:
        models = list(all_models.keys())
    
    results = {}
    
    # PHASE 1: TRAINING
    if 'train' in phases:
        log_info("=" * 60, "MAIN")
        log_info("PHASE 1: TRAINING TRIALWISE MODELS", "MAIN")
        log_info("=" * 60, "MAIN")
        
        train_config = CONFIG["training"].copy()
        training_results = train_all_models(config=train_config)
        results['training'] = training_results
    
    # PHASE 2: RECOVERY ANALYSIS
    if 'recovery' in phases:
        log_info("=" * 60, "MAIN")
        log_info("PHASE 2: RECOVERY ANALYSIS", "MAIN")
        log_info("=" * 60, "MAIN")
        
        recovery_results = {}
        for model_name in models:
            checkpoint_path = os.path.join(CONFIG["paths"]["checkpoints"], f"{model_name}.keras")
            if os.path.exists(checkpoint_path):
                try:
                    recovery_results[model_name] = run_recovery_analysis(model_name)
                except Exception as e:
                    log_info(f"Recovery failed for {model_name}: {str(e)}", "ERROR")
        
        results['recovery'] = recovery_results
    
    # PHASE 3: REAL DATA INFERENCE (example with dummy data)
    if 'inference' in phases:
        log_info("=" * 60, "MAIN")
        log_info("PHASE 3: REAL DATA INFERENCE", "MAIN")
        log_info("=" * 60, "MAIN")
        
        # Example: Load real data and run inference
        # You should adapt this to load your actual subject data
        log_info("To run inference on real data, call infer_subject_parameters() or infer_batch_subjects()", "INFERENCE")
        
        results['inference'] = "Ready for real data inference"
    
    log_info("=" * 60, "MAIN")
    log_info("WORKFLOW COMPLETE", "MAIN")
    log_info("=" * 60, "MAIN")
    
    return results

# ============================================================================
# QUICK INFERENCE FUNCTIONS (for interactive use)
# ============================================================================

def quick_infer(cpp_values, rt_values, model_name='model_DC_TrialWise'):
    """
    Quick inference for testing. Pass arrays directly.
    
    Args:
        cpp_values: Array of CPP values (n_trials,)
        rt_values: Array of reaction times (n_trials,)
        model_name: Model to use
    
    Returns:
        result: Inferred parameters dict
    """
    subject_data = {
        'cpp': np.array(cpp_values, dtype=np.float32),
        'reaction_times': np.array(rt_values, dtype=np.float32),
    }
    return infer_subject_parameters(subject_data, model_name)

if __name__ == "__main__":
    import sys
    
    print(f"Using device: {CONFIG['device']}")
    
    if len(sys.argv) > 1:
        phase = sys.argv[1]  # 'train', 'recovery', or 'inference'
        run_complete_workflow(phases=[phase])
    else:
        # Default: run all phases
        run_complete_workflow(phases=['train', 'recovery'])
