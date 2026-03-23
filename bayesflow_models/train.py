import os

os.environ.setdefault("KERAS_BACKEND", "torch")

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




# def train_amortizer(model, model_name, n_sim=10000, epochs=1):
#     summary_network = bf.networks.SetTransformer(summary_dim=10)
#     inference_network = bf.networks.CouplingFlow()
#     workflow = bf.BasicWorkflow(
#     simulator=model[0],
#     adapter=model[1],
#     inference_network=inference_network,
#     summary_network=summary_network,
#     )
#     history = workflow.fit_online(epochs=epochs, batch_size=32, num_batches_per_epoch=n_sim, max_queue_size=100)    

#     # Save

#     base_dir = os.getcwd()
#     save_dir = os.path.join(base_dir, "trained_model", "checkpoints")
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, model_name + ".keras")
#     workflow.approximator.save(filepath=save_path)
    

#     return history


def train_amortizer(model, model_name, n_sim=10000, epochs=1):
    summary_network = bf.networks.SetTransformer(summary_dim=10)
    inference_network = bf.networks.CouplingFlow()
    workflow = bf.BasicWorkflow(
    simulator=model[0],
    adapter=model[1],
    inference_network=inference_network,
    summary_network=summary_network,
    )
    # Use AdamW as the default (to match BayesFlow default under the hood).

    optimizer = torch.optim.AdamW(
        workflow.approximator.parameters(),
        lr=0.001,
        weight_decay=0.0
    )
    workflow.optimizer = optimizer
    history = workflow.fit_online(epochs=epochs, batch_size=32, num_batches_per_epoch=n_sim, max_queue_size=100)    

    # Save

    base_dir = os.getcwd()
    save_dir = os.path.join(base_dir, "trained_model", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name + ".keras")
    # workflow.approximator.save(filepath=save_path)
    # Save both model and optimizer states (PyTorch-style)
    save_dict = {
        'model_state_dict': workflow.approximator.state_dict(),
        'optimizer_state_dict': workflow.optimizer.state_dict(),
        'model_name': model_name
    }
    print("hi new save method")
    torch.save(save_dict, os.path.join(save_path + "_checkpoint.pt"))
    print(f"✅ Saved model and optimizer for {model_name}")

    return history

# def train_amortizer_load(model, model_name, n_sim=1000, epochs=10, checkpoint_path=None):
#     """
#     Train an amortizer or continue training from a checkpoint using BasicWorkflow.
    
#     Args:
#         model: Tuple of (simulator, configurator)
#         model_name: Name of the model (e.g., 'model_1a')
#         n_sim: Number of simulations per epoch
#         epochs: Number of training epochs
#         checkpoint_path: Path to a .keras checkpoint file (optional)
#     Returns:
#         history: Training history
#     """
#     simulator, configurator = model
    
#     # Define networks
#     summary_network = bf.networks.SetTransformer(summary_dim=10)
#     inference_network = bf.networks.CouplingFlow()
    
#     # Initialize approximator
#     if checkpoint_path is not None:
#         try:
#             print(f"Loading checkpoint: {checkpoint_path}")
#             approximator = keras.models.load_model(checkpoint_path, custom_objects={
#                 'SetTransformer': bf.networks.SetTransformer,
#                 'CouplingFlow': bf.networks.CouplingFlow
#             })
#             print(f"✅ Loaded checkpoint for {model_name}")
#         except Exception as e:
#             print(f"Error loading checkpoint: {e}")
#             raise ValueError(f"Failed to load checkpoint: {checkpoint_path}")
#     else:
#         approximator = bf.amortizers.AmortizedPosterior(
#             inference_net=inference_network,
#             summary_net=summary_network
#         )
    
#     # Create workflow
#     workflow = bf.workflows.BasicWorkflow(
#         simulator=simulator,
#         adapter=configurator,
#         inference_network=inference_network,
#         summary_network=summary_network
#     )
    
#     # Move to GPU
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     workflow.approximator.to(device)
#     print(f"Training on device: {device}")
    
#     # Train
#     history = workflow.fit_online(
#         epochs=epochs,
#         batch_size=32,
#         num_batches_per_epoch=n_sim,
#         max_queue_size=100
#     )
    
#     # Save updated model
#     base_dir = os.getcwd()
#     save_dir = os.path.join(base_dir, "trained_model", "checkpoints")
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, f"{model_name}_updated.keras")
#     workflow.approximator.save(filepath=save_path)
#     print(f"✅ Saved updated amortizer for {model_name} at {save_path}")
    
#     return history


def train_amortizer_load(model, model_name, n_sim=1000, epochs=10, checkpoint_path=None):
    """
    Train an amortizer or continue training from a checkpoint using BasicWorkflow.
    
    Args:
        model: Tuple of (simulator, configurator)
        model_name: Name of the model (e.g., 'model_1a')
        n_sim: Number of simulations per epoch
        epochs: Number of training epochs
        checkpoint_path: Path to a .keras checkpoint file (optional)
    Returns:
        history: Training history
    """
    simulator, configurator = model
    print("hi")
    # Define networks
    summary_network = bf.networks.SetTransformer(summary_dim=10)
    inference_network = bf.networks.CouplingFlow()

    # Initialize approximator
    approximator = bf.amortizers.AmortizedPosterior(
        inference_net=inference_network,
        summary_net=summary_network
    )

    # Create workflow
    workflow = bf.workflows.BasicWorkflow(
        simulator=simulator,
        adapter=configurator,
        inference_network=inference_network,
        summary_network=summary_network
    )

    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    workflow.approximator.to(device)
    print(f"Training on device: {device}")

    # Load weights and optimizer state if checkpoint is provided
    if checkpoint_path is not None:
        try:
            print(f"Loading checkpoint: {checkpoint_path}")
            workflow.approximator.load_weights(checkpoint_path)
            print(f"✅ Loaded weights from checkpoint for {model_name}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise ValueError(f"Failed to load checkpoint: {checkpoint_path}")

    # Train
    history = workflow.fit_online(
        epochs=epochs,
        batch_size=32,
        num_batches_per_epoch=n_sim,
        max_queue_size=100
    )

    # Save updated model
    base_dir = os.getcwd()
    save_dir = os.path.join(base_dir, "trained_model", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_updated.keras")
    workflow.approximator.save_weights(save_path)
    print(f"✅ Saved updated amortizer weights for {model_name} at {save_path}")

    return history




def train_amortizer_resume(
    model,                 # tuple (simulator, adapter)
    model_name: str,       # same name used when saving
    n_sim: int = 1_000,
    epochs: int = 10,      # how many *additional* epochs to run
    batch_size: int = 32,
    initial_lr: float = 5e-4,
    checkpoint_dir: str = "trained_model/checkpoints"
):
    """
    1) Looks for `<checkpoint_dir>/<model_name>.keras`.  
       • If found, does keras.models.load_model(...) to get `loaded_amortizer`.  
       • Otherwise, falls back to "from‐scratch" (and prints a warning).
    2) Builds a fresh BasicWorkflow, but passes in `optimizer=loaded_amortizer.optimizer`.  
       This makes sure that when BayesFlow calls `compile(...)` internally, it uses the restored optim.  
    3) Assigns `workflow.approximator = loaded_amortizer` before calling fit_online.  
    4) Runs another `epochs` of `fit_online(...)`, then re‐saves into the same `<model_name>.keras` path.
    """

    simulator, adapter = model
    checkpoint_save = os.path.join(checkpoint_dir, model_name + ".keras")

    # Step 2: Build a brand‐new BasicWorkflow, but inject our optimizer (loaded or freshly created)
    summary_net   = bf.networks.SetTransformer(summary_dim=10)
    inference_net = bf.networks.CouplingFlow()

    # Step 1: Try to load existing amortizer (weights + optimizer state)
    if os.path.exists(checkpoint_save):
        print(f"▶️ Loading existing amortizer from:\n   {checkpoint_save}")
        loaded_amortizer = keras.models.load_model(
            checkpoint_save,
            custom_objects={
                # BayesFlow’s custom layers/networks:
                "SetTransformer": bf.networks.SetTransformer,
                "CouplingFlow": bf.networks.CouplingFlow,
                # If you used any other custom Modules, list them here as well
            }
        )
        print("✅  Successfully loaded amortizer (weights + optimizer)")

        # Extract the loaded optimizer instance from the loaded Keras Model:
        loaded_optimizer = loaded_amortizer.optimizer
        if loaded_optimizer is None:
            raise RuntimeError(
                "Loaded amortizer has no .optimizer attached. "
                "That implies `save()` did not store optim state. "
                "Double‐check that you originally called `amortizer.save(...)`."
            )
        
        workflow = bf.workflows.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=inference_net,
        summary_network=summary_net,
        optimizer=loaded_optimizer,
        initial_learning_rate=initial_lr,
        checkpoint_filepath=None
        )
    else:
        print(f"⚠️ No checkpoint found at '{checkpoint_save}'.  Starting from scratch.")
        # Fall back to exactly the same steps as train_amortizer_from_scratch
        loaded_amortizer = None
        # loaded_optimizer = keras.optimizers.Adam(learning_rate=initial_lr)
        workflow = bf.BasicWorkflow(
            simulator=model[0],
            adapter=model[1],
            inference_network=inference_net,
            summary_network=summary_net,
        )
    # Move the amortizer inside this new workflow:
    if loaded_amortizer is not None:
        workflow.approximator = loaded_amortizer
        print("▶️  Plugged loaded amortizer into new BasicWorkflow")
    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    workflow.approximator.to(device)

    # Step 3: Run further online training
    print(f"▶️ Continuing training for {epochs} more epochs …")
    history = workflow.fit_online(
        epochs=epochs,
        batch_size=batch_size,
        num_batches_per_epoch=n_sim,
        max_queue_size=100
    )

    # Step 4: Re‐save the updated amortizer (now with new weights + updated optimizer state)
    os.makedirs(checkpoint_dir, exist_ok=True)
    workflow.approximator.save(filepath=checkpoint_save)
    print(f"✅  Saved updated amortizer (weights + optimizer) at:\n   {checkpoint_save}")

    return history
