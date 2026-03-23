"""
Main entry point for DDM-DC Trial-Wise Workflow

Supports three phases:
1. train    - Train all models from scratch or resume
2. recovery - Run parameter recovery analysis
3. inference - Prepare for real data inference

Usage:
    python main_workflow.py train           # Train all models
    python main_workflow.py recovery        # Run recovery analysis
    python main_workflow.py inference       # Prepare inference
    python main_workflow.py all             # Run all phases
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import sys
import torch
from bayesflow_models.workflow_trialwise import (
    run_complete_workflow,
    get_available_models,
    log_info,
    CONFIG,
    infer_subject_parameters,
    infer_batch_subjects,
    quick_infer
)

def print_status():
    """Print current system and Config status"""
    print("\n" + "=" * 70)
    print("DDM-DC TRIAL-WISE MODEL WORKFLOW")
    print("=" * 70)
    print(f"\n✓ Device: {CONFIG['device']}")
    print(f"✓ Environment: KERAS_BACKEND=torch")
    print(f"\n📁 Directories:")
    for key, path in CONFIG["paths"].items():
        print(f"   - {key}: {path}")
    
    print(f"\n🤖 Available Models:")
    models = get_available_models()
    
    print(f"\n⚙️  Configuration:")
    print(f"   Training:")
    for key, val in CONFIG["training"].items():
        print(f"     - {key}: {val}")
    print(f"   Recovery:")
    for key, val in CONFIG["recovery"].items():
        print(f"     - {key}: {val}")
    print("\n" + "=" * 70 + "\n")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print_status()
    
    if len(sys.argv) > 1:
        phase = sys.argv[1].lower()
        
        if phase == 'train':
            log_info("Starting TRAINING phase", "MAIN")
            run_complete_workflow(phases=['train'])
        
        elif phase == 'recovery':
            log_info("Starting RECOVERY ANALYSIS phase", "MAIN")
            run_complete_workflow(phases=['recovery'])
        
        elif phase == 'inference':
            log_info("INFERENCE phase ready - use infer_subject_parameters() or infer_batch_subjects()", "MAIN")
            print("\nExample usage:")
            print("  from main_workflow import quick_infer")
            print("  result = quick_infer(")
            print("      cpp_values=[0.5, 0.8, 0.3, ...],")
            print("      rt_values=[0.45, 0.52, 0.38, ...],")
            print("      model_name='model_DC_TrialWise'")
            print("  )")
        
        elif phase == 'all':
            log_info("Starting COMPLETE WORKFLOW (train + recovery)", "MAIN")
            run_complete_workflow(phases=['train', 'recovery'])
        
        else:
            print(f"Unknown phase: {phase}")
            print("\nValid phases: train, recovery, inference, all")
            sys.exit(1)
    
    else:
        print("Usage: python main_workflow.py [phase]")
        print("\nPhases:")
        print("  train     - Train all models (or resume from checkpoint)")
        print("  recovery  - Run parameter recovery analysis")
        print("  inference - Prepare for real data inference")
        print("  all       - Run training + recovery analysis")
        print("\nExample:")
        print("  python main_workflow.py train")
        print("  python main_workflow.py recovery")
        print("  python main_workflow.py all")

if __name__ == "__main__":
    main()
