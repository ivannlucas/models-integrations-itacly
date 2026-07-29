"""
Random hyperparameter search (Random Search).

Explores combinations of {num_layers, neurons, lr, activation} by training
a DynamicMLP for each configuration and selecting the best by validation MSE.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from src.training.model import DynamicMLP
from src.training.trainer import train_model
from src.utils.constants import FEATURES, TARGETS


def random_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    y_val_original: np.ndarray,
    scaler_y: MinMaxScaler,
    param_space: Dict[str, list] = None,
    n_iter: int = 60,
    epochs: int = 300,
    batch_size: int = 128,
    patience: int = 15,
    verbose: bool = True,
) -> Tuple[DynamicMLP, Dict, List[Dict]]:
    """
    Execute Random Search over the hyperparameter space.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Normalized training data.
    X_val, y_val : np.ndarray
        Normalized validation data.
    y_val_original : np.ndarray
        Validation data in original units (for real MAE computation).
    scaler_y : MinMaxScaler
        Fitted scaler to denormalize predictions.
    param_space : dict, optional
        Search space. Default:
        {'num_layers': [1,2,3,4], 'neurons': [16,32,64,128],
         'lr': [0.01,0.005,0.001,0.0005], 'activation': ['ReLU','Tanh']}
    n_iter : int
        Number of random combinations to try.
    epochs, batch_size, patience : int
        Training parameters per iteration.
    verbose : bool
        If True, prints progress.

    Returns
    -------
    tuple
        (best_model, best_params, search_history)
    """
    if param_space is None:
        param_space = {
            "num_layers": [1, 2, 3, 4],
            "neurons": [16, 32, 64, 128],
            "lr": [0.01, 0.005, 0.001, 0.0005],
            "activation": ["ReLU", "Tanh"],
        }

    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    best_val_loss = float("inf")
    best_params = {}
    best_model_state = None
    search_history = []

    if verbose:
        print(f"Starting Random Search: {n_iter} combinations...")

    start_time = time.time()

    for i in range(n_iter):
        # Random hyperparameter sampling
        params = {k: random.choice(v) for k, v in param_space.items()}

        model = DynamicMLP(
            input_size=input_size,
            output_size=output_size,
            num_layers=params["num_layers"],
            neurons=params["neurons"],
            activation=params["activation"],
        )

        # Train
        model, info = train_model(
            model, X_train, y_train, X_val, y_val,
            lr=params["lr"], epochs=epochs,
            batch_size=batch_size, patience=patience,
            verbose=False,
        )

        # Metrics in original units
        model.eval()
        with torch.no_grad():
            val_preds_scaled = model(torch.FloatTensor(X_val)).numpy()
        val_preds_real = scaler_y.inverse_transform(val_preds_scaled)

        mae_e = mean_absolute_error(y_val_original[:, 0], val_preds_real[:, 0])
        mae_t = mean_absolute_error(y_val_original[:, 1], val_preds_real[:, 1])

        if verbose:
            print(
                f"Iter [{i+1:>2}/{n_iter}] | "
                f"Layers: {params['num_layers']} | Neurons: {params['neurons']:>3} | "
                f"LR: {params['lr']:.4f} | Act: {params['activation']:<4} | "
                f"Epochs: {info['epochs_executed']:>3} | "
                f"Val MSE: {info['best_val_loss']:.6f} | "
                f"MAE E(kW): {mae_e:.4f} | MAE T(C): {mae_t:.4f}"
            )

        search_history.append({
            "iter": i + 1,
            "val_loss": info["best_val_loss"],
            "mae_e_val_kw": mae_e,
            "mae_t_val_c": mae_t,
            "epochs_executed": info["epochs_executed"],
            "params": params,
        })

        # Save global best model
        if info["best_val_loss"] < best_val_loss:
            best_val_loss = info["best_val_loss"]
            best_params = params
            best_model_state = copy.deepcopy(model.state_dict())

    elapsed = time.time() - start_time

    if verbose:
        print(f"\nRandom Search completed in {elapsed:.1f}s")
        print(f"Best Val MSE: {best_val_loss:.6f} | Params: {best_params}")

    # Reconstruct best model
    best_model = DynamicMLP(
        input_size=input_size,
        output_size=output_size,
        num_layers=best_params["num_layers"],
        neurons=best_params["neurons"],
        activation=best_params["activation"],
    )
    best_model.load_state_dict(best_model_state)
    best_model.eval()

    return best_model, best_params, search_history
