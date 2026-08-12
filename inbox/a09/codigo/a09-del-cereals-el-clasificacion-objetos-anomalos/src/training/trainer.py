from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data_processing.preprocess import build_sequence_feature_frame, build_sliding_windows, split_sample_ids_stratified
from src.evaluation.evaluator import evaluate_with_sequence_models
from src.training.sequential import (
    build_class_weights,
    fit_scaler,
    load_checkpoint,
    predict_sequence_proba,
    save_checkpoint,
    save_pickle,
    search_sequence_model,
    set_global_seed,
    transform_sequences,
)


def _ensure_dirs(cfg: dict[str, Any]) -> None:
    for p in [
        cfg['paths']['predictions_dir'],
        cfg['paths']['splits_dir'],
        cfg['paths']['models_artifacts_dir'],
        cfg['paths']['models_metrics_dir'],
        cfg['paths']['metrics_figures_dir'],
    ]:
        Path(p).mkdir(parents=True, exist_ok=True)


def _export_split_frames(
    seq_df: pd.DataFrame,
    split_ids: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> None:
    group_col = cfg['data']['group_column']
    splits_dir = Path(cfg['paths']['splits_dir'])

    split_to_filename = {
        'train_sample_ids': 'model_train.csv',
        'validation_sample_ids': 'model_validation.csv',
        'test_sample_ids': 'model_test.csv',
    }

    for split_key, filename in split_to_filename.items():
        sample_ids = np.asarray(split_ids.get(split_key, []))
        split_df = seq_df[seq_df[group_col].isin(sample_ids)].copy()
        split_df.to_csv(splits_dir / filename, index=False)


def _load_checkpoint_model(path: Path):
    checkpoint = load_checkpoint(path, map_location='cpu')
    return checkpoint['model'], checkpoint


def _evaluate_on_split(
    model_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    scaler,
    *,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    model, checkpoint = _load_checkpoint_model(model_path)
    X_s = transform_sequences(scaler, X)
    proba = predict_sequence_proba(model, X_s, batch_size=batch_size, device=device)
    pred = np.argmax(proba, axis=1) if len(proba) else np.zeros((0,), dtype=int)
    return {
        'checkpoint': checkpoint,
        'y_pred': pred,
        'y_proba': proba,
    }


def train_pipeline(df: pd.DataFrame, cfg: dict[str, Any], quick: bool = False) -> dict[str, Any]:
    set_global_seed(int(cfg['training'].get('random_state', 42)))
    _ensure_dirs(cfg)

    group_col = cfg['data']['group_column']
    ts_col = cfg['data']['timestamp_column']
    target_col = cfg['data']['target_column']
    seq_cfg = cfg.get('sequence', {})
    selection_metric = str(seq_cfg.get('selection_metric', 'f1_macro'))
    batch_size_eval = int(seq_cfg.get('training', {}).get('eval_batch_size', 128))

    seq_df, feature_columns = build_sequence_feature_frame(df, cfg)
    split_ids = split_sample_ids_stratified(df, cfg)
    _export_split_frames(seq_df, split_ids, cfg)

    train_payload = build_sliding_windows(seq_df, cfg, feature_columns=feature_columns, sample_ids=split_ids['train_sample_ids'])
    val_payload = build_sliding_windows(seq_df, cfg, feature_columns=feature_columns, sample_ids=split_ids['validation_sample_ids'])
    test_payload = build_sliding_windows(seq_df, cfg, feature_columns=feature_columns, sample_ids=split_ids['test_sample_ids'])

    X_train, y_train = train_payload['X_seq'], train_payload['y_seq']
    X_val, y_val = val_payload['X_seq'], val_payload['y_seq']
    X_test, y_test = test_payload['X_seq'], test_payload['y_seq']

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError('El pipeline secuencial no ha generado ventanas suficientes en train/validation/test.')

    scaler = fit_scaler(X_train)
    num_classes = int(len(sorted(np.unique(seq_df[target_col].to_numpy(dtype=int)).tolist())))
    class_weights = build_class_weights(y_train, num_classes=num_classes)
    requested_device = str(cfg['training'].get('device', 'cpu')).lower().strip()
    device = 'cuda' if requested_device in {'cuda', 'gpu'} and torch.cuda.is_available() else 'cpu'
    device_mode = 'gpu' if device == 'cuda' else 'cpu'

    metrics_dir = Path(cfg['paths']['models_metrics_dir'])
    artifacts_dir = Path(cfg['paths']['models_artifacts_dir'])
    pred_dir = Path(cfg['paths']['predictions_dir'])
    figures_dir = Path(cfg['paths']['metrics_figures_dir'])

    save_pickle(scaler, artifacts_dir / 'sequence_scaler.pkl')

    lstm_summary, lstm_history, lstm_best = search_sequence_model(
        'lstm',
        X_train,
        y_train,
        X_val,
        y_val,
        cfg=cfg,
        scaler=scaler,
        num_classes=num_classes,
        class_weights=class_weights,
        device=device,
        quick=quick,
    )
    gru_summary, gru_history, gru_best = search_sequence_model(
        'gru',
        X_train,
        y_train,
        X_val,
        y_val,
        cfg=cfg,
        scaler=scaler,
        num_classes=num_classes,
        class_weights=class_weights,
        device=device,
        quick=quick,
    )

    lstm_path = artifacts_dir / 'lstm_best.pt'
    gru_path = artifacts_dir / 'gru_best.pt'
    save_checkpoint(lstm_path, lstm_best.checkpoint)
    save_checkpoint(gru_path, gru_best.checkpoint)

    lstm_score = float(lstm_summary.iloc[0]['selection_value'])
    gru_score = float(gru_summary.iloc[0]['selection_value'])
    if gru_score > lstm_score:
        winner_name = 'gru'
        winner_path = gru_path
        winner_result = gru_best
        winner_score = gru_score
    else:
        winner_name = 'lstm'
        winner_path = lstm_path
        winner_result = lstm_best
        winner_score = lstm_score

    final_winner_path = artifacts_dir / 'final_winner.pt'
    shutil.copy2(winner_path, final_winner_path)

    winner_eval = _evaluate_on_split(
        final_winner_path,
        X_test,
        y_test,
        scaler,
        batch_size=batch_size_eval,
        device=device,
    )
    lstm_eval = _evaluate_on_split(
        lstm_path,
        X_test,
        y_test,
        scaler,
        batch_size=batch_size_eval,
        device=device,
    )
    gru_eval = _evaluate_on_split(
        gru_path,
        X_test,
        y_test,
        scaler,
        batch_size=batch_size_eval,
        device=device,
    )

    eval_payload = evaluate_with_sequence_models(
        cfg,
        seq_df=seq_df,
        feature_columns=feature_columns,
        split_ids=split_ids,
        scaler=scaler,
        model_paths={
            'lstm': lstm_path,
            'gru': gru_path,
            'winner': final_winner_path,
        },
        X_test=X_test,
        y_test=y_test,
        test_meta=test_payload['window_meta'],
        selection_metric=selection_metric,
        device=device,
        batch_size=batch_size_eval,
        quick=quick,
    )

    bundle = {
        'pipeline_type': 'sequence_lstm_gru',
        'selection_metric': selection_metric,
        'feature_columns': feature_columns,
        'window_size': int(seq_cfg.get('window_size', 48)),
        'stride': int(seq_cfg.get('stride', 12)),
        'label_mode': str(seq_cfg.get('label_mode', 'last')),
        'pad_short_sequences': bool(seq_cfg.get('pad_short_sequences', False)),
        'split_sample_ids': {
            'train_sample_ids': split_ids['train_sample_ids'].tolist(),
            'validation_sample_ids': split_ids['validation_sample_ids'].tolist(),
            'test_sample_ids': split_ids['test_sample_ids'].tolist(),
        },
        'model_files': {
            'lstm': lstm_path.name,
            'gru': gru_path.name,
            'winner': final_winner_path.name,
        },
        'scaler_file': 'sequence_scaler.pkl',
        'best_model_name': winner_name,
        'best_model_metric': winner_score,
        'device_requested': requested_device,
        'device_used': device_mode,
        'validation_scores': {
            'lstm': float(lstm_score),
            'gru': float(gru_score),
        },
        'config_snapshot': cfg,
    }
    with open(artifacts_dir / 'model_bundle_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

    lstm_summary.to_csv(metrics_dir / 'lstm_search_summary.csv', index=False)
    gru_summary.to_csv(metrics_dir / 'gru_search_summary.csv', index=False)
    lstm_history.to_csv(metrics_dir / 'lstm_training_history.csv', index=False)
    gru_history.to_csv(metrics_dir / 'gru_training_history.csv', index=False)

    eval_payload['model_selection'] = {
        'lstm_validation_f1_macro': lstm_score,
        'gru_validation_f1_macro': gru_score,
        'winner': winner_name,
    }
    eval_payload['device_used'] = device_mode
    eval_payload['model_files'] = {
        'lstm': str(lstm_path),
        'gru': str(gru_path),
        'winner': str(final_winner_path),
    }
    eval_payload['bundle_file'] = str(artifacts_dir / 'model_bundle_metadata.json')
    eval_payload['scaler_file'] = str(artifacts_dir / 'sequence_scaler.pkl')
    return eval_payload
