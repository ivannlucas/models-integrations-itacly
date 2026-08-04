from __future__ import annotations

import copy
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


LABEL_NAMES = {0: 'sano', 1: 'insectos', 2: 'moho_critico'}


@dataclass
class SequenceTrainingResult:
    model_type: str
    model: nn.Module
    checkpoint: dict[str, Any]
    history: pd.DataFrame
    metrics: dict[str, float]
    predictions: dict[str, np.ndarray]
    params: dict[str, Any]
    scaler: StandardScaler


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        model_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_classes: int,
    ):
        super().__init__()
        model_type = model_type.lower().strip()
        self.model_type = model_type
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)
        self.num_classes = int(num_classes)

        rnn_cls = nn.LSTM if model_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        rnn_out_size = self.hidden_size * (2 if self.bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(rnn_out_size, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last_state = out[:, -1, :]
        return self.head(last_state)


class _TensorSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


def set_global_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(cfg: dict[str, Any]) -> torch.device:
    requested = str(cfg.get('training', {}).get('device', 'cpu')).lower().strip()
    if requested in {'cuda', 'gpu'} and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    return scaler


def transform_sequences(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    flat = X.reshape(-1, X.shape[-1])
    scaled = scaler.transform(flat)
    return scaled.reshape(X.shape).astype(np.float32)


def build_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=num_classes).astype(np.float32)
    counts[counts == 0.0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32)


def build_model(model_type: str, input_size: int, num_classes: int, params: dict[str, Any]) -> SequenceClassifier:
    return SequenceClassifier(
        model_type=model_type,
        input_size=input_size,
        hidden_size=int(params['hidden_size']),
        num_layers=int(params.get('num_layers', 1)),
        dropout=float(params.get('dropout', 0.0)),
        bidirectional=bool(params.get('bidirectional', False)),
        num_classes=num_classes,
    )


def _make_loader(X: np.ndarray, y: np.ndarray | None, batch_size: int, *, shuffle: bool) -> DataLoader:
    dataset = _TensorSequenceDataset(X, y)
    return DataLoader(dataset, batch_size=max(1, int(batch_size)), shuffle=shuffle, drop_last=False)


@torch.no_grad()
def predict_sequence_proba(
    model: nn.Module,
    X: np.ndarray,
    *,
    batch_size: int = 128,
    device: torch.device | str = 'cpu',
) -> np.ndarray:
    if len(X) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    loader = _make_loader(X, None, batch_size, shuffle=False)
    model = model.to(device)
    model.eval()
    probs: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        proba = torch.softmax(logits, dim=1).detach().cpu().numpy()
        probs.append(proba)
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, 0), dtype=np.float32)


@torch.no_grad()
def evaluate_sequence_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 128,
    device: torch.device | str = 'cpu',
) -> dict[str, Any]:
    proba = predict_sequence_proba(model, X, batch_size=batch_size, device=device)
    pred = np.argmax(proba, axis=1) if len(proba) else np.zeros((0,), dtype=int)
    labels = sorted(np.unique(y).tolist()) if len(y) else [0, 1, 2]
    metrics = {
        'accuracy': float(accuracy_score(y, pred)) if len(y) else 0.0,
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)) if len(y) else 0.0,
        'f1_macro': float(f1_score(y, pred, average='macro', zero_division=0)) if len(y) else 0.0,
        'precision_macro': float(precision_score(y, pred, average='macro', zero_division=0)) if len(y) else 0.0,
        'recall_macro': float(recall_score(y, pred, average='macro', zero_division=0)) if len(y) else 0.0,
        'log_loss': float(log_loss(y, proba, labels=labels)) if len(y) else 0.0,
    }
    return {
        'metrics': metrics,
        'y_pred': pred,
        'y_proba': proba,
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    running = 0.0
    total = 0
    train_mode = optimizer is not None
    model.train(train_mode)

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

        batch_size = int(X_batch.shape[0])
        running += float(loss.item()) * batch_size
        total += batch_size

    return running / max(total, 1)


def train_sequence_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    params: dict[str, Any],
    cfg: dict[str, Any],
    scaler: StandardScaler,
    num_classes: int,
    class_weights: torch.Tensor | None = None,
    device: torch.device | str = 'cpu',
) -> SequenceTrainingResult:
    seq_train_cfg = cfg.get('sequence', {}).get('training', {})
    max_epochs = int(seq_train_cfg.get('max_epochs', 25))
    patience = int(seq_train_cfg.get('patience', 5))
    min_delta = float(seq_train_cfg.get('min_delta', 0.0))
    grad_clip = float(seq_train_cfg.get('grad_clip', 1.0))
    batch_size = int(params.get('batch_size', seq_train_cfg.get('batch_size', 32)))
    learning_rate = float(params.get('learning_rate', seq_train_cfg.get('learning_rate', 1e-3)))
    weight_decay = float(params.get('weight_decay', seq_train_cfg.get('weight_decay', 0.0)))

    X_train_s = transform_sequences(scaler, X_train)
    X_val_s = transform_sequences(scaler, X_val)

    model = build_model(model_type, X_train.shape[-1], num_classes, params).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    train_loader = _make_loader(X_train_s, y_train, batch_size, shuffle=True)
    val_loader = _make_loader(X_val_s, y_val, batch_size, shuffle=False)

    best_state = copy.deepcopy(model.state_dict())
    best_metric = -np.inf
    best_loss = np.inf
    best_epoch = 0
    bad_epochs = 0
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _run_epoch(model, val_loader, criterion, None, device)
        val_eval = evaluate_sequence_model(model, X_val_s, y_val, batch_size=batch_size, device=device)
        val_metrics = val_eval['metrics']
        selection_metric = float(val_metrics.get(cfg.get('sequence', {}).get('selection_metric', 'f1_macro'), val_metrics['f1_macro']))

        history_rows.append(
            {
                'epoch': epoch,
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                **{f'val_{k}': float(v) for k, v in val_metrics.items()},
            }
        )

        improvement = selection_metric > (best_metric + min_delta)
        if improvement:
            best_metric = selection_metric
            best_loss = float(val_loss)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    model.load_state_dict(best_state)
    final_eval = evaluate_sequence_model(model, X_val_s, y_val, batch_size=batch_size, device=device)
    final_metrics = final_eval['metrics']
    history = pd.DataFrame(history_rows)

    checkpoint = {
        'model_type': model_type,
        'state_dict': best_state,
        'model_config': {
            'input_size': int(X_train.shape[-1]),
            'hidden_size': int(params['hidden_size']),
            'num_layers': int(params.get('num_layers', 1)),
            'dropout': float(params.get('dropout', 0.0)),
            'bidirectional': bool(params.get('bidirectional', False)),
            'num_classes': int(num_classes),
        },
        'training_params': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'max_epochs': max_epochs,
            'patience': patience,
            'min_delta': min_delta,
            'grad_clip': grad_clip,
        },
        'selection_metric': cfg.get('sequence', {}).get('selection_metric', 'f1_macro'),
        'best_metric': float(best_metric),
        'best_loss': float(best_loss),
        'best_epoch': int(best_epoch),
        'params': params,
    }

    return SequenceTrainingResult(
        model_type=model_type,
        model=model,
        checkpoint=checkpoint,
        history=history,
        metrics=final_metrics,
        predictions={
            'y_pred': final_eval['y_pred'],
            'y_proba': final_eval['y_proba'],
        },
        params=params,
        scaler=scaler,
    )


def search_sequence_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    cfg: dict[str, Any],
    scaler: StandardScaler,
    num_classes: int,
    class_weights: torch.Tensor | None,
    device: torch.device | str,
    quick: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, SequenceTrainingResult]:
    search_cfg = cfg.get('sequence', {}).get('search', {}).get(model_type, {})
    param_grid = list(ParameterGrid(search_cfg)) if search_cfg else [{}]
    if quick:
        limit = int(cfg.get('sequence', {}).get('training', {}).get('quick_search_trials', 2))
        param_grid = param_grid[:max(1, limit)]

    if not param_grid:
        param_grid = [{}]

    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    best_result: SequenceTrainingResult | None = None
    best_score = -np.inf

    selection_metric = str(cfg.get('sequence', {}).get('selection_metric', 'f1_macro'))

    for trial_id, params in enumerate(param_grid, start=1):
        result = train_sequence_model(
            model_type,
            X_train,
            y_train,
            X_val,
            y_val,
            params=params,
            cfg=cfg,
            scaler=scaler,
            num_classes=num_classes,
            class_weights=class_weights,
            device=device,
        )
        metric_value = float(result.metrics.get(selection_metric, result.metrics.get('f1_macro', 0.0)))
        summary_rows.append(
            {
                'model_type': model_type,
                'trial_id': trial_id,
                'selection_metric': selection_metric,
                'selection_value': metric_value,
                'val_accuracy': result.metrics['accuracy'],
                'val_balanced_accuracy': result.metrics['balanced_accuracy'],
                'val_f1_macro': result.metrics['f1_macro'],
                'val_precision_macro': result.metrics['precision_macro'],
                'val_recall_macro': result.metrics['recall_macro'],
                'val_log_loss': result.metrics['log_loss'],
                'params': json.dumps(params, sort_keys=True, ensure_ascii=False),
            }
        )
        history = result.history.copy()
        history.insert(0, 'model_type', model_type)
        history.insert(1, 'trial_id', trial_id)
        fold_rows.extend(history.to_dict(orient='records'))

        if metric_value > best_score:
            best_score = metric_value
            best_result = result

    assert best_result is not None
    summary_df = pd.DataFrame(summary_rows).sort_values('selection_value', ascending=False).reset_index(drop=True)
    history_df = pd.DataFrame(fold_rows).sort_values(['trial_id', 'epoch']).reset_index(drop=True)
    return summary_df, history_df, best_result


def save_pickle(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_checkpoint(path: str | Path, checkpoint: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, p)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = 'cpu') -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    model_config = checkpoint['model_config']
    model = SequenceClassifier(
        model_type=checkpoint['model_type'],
        input_size=int(model_config['input_size']),
        hidden_size=int(model_config['hidden_size']),
        num_layers=int(model_config['num_layers']),
        dropout=float(model_config['dropout']),
        bidirectional=bool(model_config['bidirectional']),
        num_classes=int(model_config['num_classes']),
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    checkpoint['model'] = model
    return checkpoint
