"""
Training loop with Early Stopping and Mini-batches.

Encapsulates the training logic from predict_model_t_h.ipynb.
"""

import copy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.training.model import DynamicMLP


def train_model(
    model: DynamicMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 0.0005,
    epochs: int = 300,
    batch_size: int = 128,
    patience: int = 15,
    verbose: bool = True,
) -> Tuple[DynamicMLP, Dict]:
    """
    Train a DynamicMLP model with early stopping and mini-batches.

    Parameters
    ----------
    model : DynamicMLP
        Model to train (already instantiated).
    X_train, y_train : np.ndarray
        Normalized training data (float32).
    X_val, y_val : np.ndarray
        Normalized validation data (float32).
    lr : float
        Learning rate for Adam optimizer.
    epochs : int
        Maximum number of epochs.
    batch_size : int
        Mini-batch size.
    patience : int
        Epochs without improvement before stopping.
    verbose : bool
        If True, prints progress.

    Returns
    -------
    tuple
        (model with best weights, dict with training info)
    """
    # Prepare tensors and DataLoader
    device = next(model.parameters()).device
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Control variables
    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0
    epochs_executed = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / n_batches)

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
        val_losses.append(val_loss)

        # --- Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        epochs_executed = epoch + 1

        if verbose and (epoch + 1) % 50 == 0:
            print(
                f"  Epoch [{epoch+1:>3}/{epochs}] | "
                f"Train MSE: {train_losses[-1]:.6f} | "
                f"Val MSE: {val_loss:.6f} | "
                f"Patience: {epochs_no_improve}/{patience}"
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
            break

    # Restore best weights
    model.load_state_dict(best_model_state)

    info = {
        "epochs_executed": epochs_executed,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    return model, info
