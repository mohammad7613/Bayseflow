import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import bayesflow as bf
import torch
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats
from time import time
import matplotlib.pyplot as plt
import random
import os

from bayesflow_models.DDM_DC_Pedestrain_TrialWise import all_models
from bayesflow_models.train import train_amortizer, train_amortizer_load, train_amortizer_resume

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    for model_name, model in all_models.items():
        print(f"\n=== Training {model_name} ===")
            # Save updated model
        base_dir = os.getcwd()
        save_dir = os.path.join(base_dir, "trained_model1", "checkpoints")
        train_amortizer_resume(model, model_name, n_sim=10, epochs=10, checkpoint_dir=save_dir)
 

if __name__ == "__main__":
    main()
