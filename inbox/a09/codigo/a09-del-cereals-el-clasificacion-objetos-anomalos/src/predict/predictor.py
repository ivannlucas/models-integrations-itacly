from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_processing.preprocess import build_sequence_feature_frame, build_sliding_windows, validate_and_normalize_input_frame
from src.predict.postprocess import format_output
from src.training.sequential import load_checkpoint, load_pickle, predict_sequence_proba, transform_sequences
from src.utils.model_bundle import align_feature_columns, merge_runtime_config_with_bundle


def load_model(model_path: str | Path) -> dict[str, Any]:
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f'No existe modelo: {p}')
    return load_checkpoint(p, map_location='cpu')


def run_inference(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    scaler = model.get('scaler')
    if scaler is None:
        raise ValueError('El checkpoint no incluye scaler.')
    X_scaled = transform_sequences(scaler, X)
    return predict_sequence_proba(model['model'], X_scaled, batch_size=int(model.get('training_params', {}).get('batch_size', 128)), device='cpu')


def save_predictions(df_pred: pd.DataFrame, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    format_output(df_pred).to_csv(p, index=False)


def validate_inference_input_columns(df_input: pd.DataFrame, runtime_cfg: dict[str, Any]) -> pd.DataFrame:
    try:
        return validate_and_normalize_input_frame(df_input, runtime_cfg, require_target=False)
    except ValueError as exc:
        raise ValueError(f'Entrada de inferencia invaÂ´lida: {exc}') from exc


def predict_with_bundle(df_input: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, str]:
    artifacts_dir = Path(cfg['paths']['models_artifacts_dir'])
    meta_path = artifacts_dir / 'model_bundle_metadata.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'No existe metadata de bundle: {meta_path}. Ejecuta entrenamiento antes de inferir.')

    with open(meta_path, 'r', encoding='utf-8') as f:
        bundle = json.load(f)

    runtime_cfg = merge_runtime_config_with_bundle(cfg, bundle)
    df = validate_inference_input_columns(df_input, runtime_cfg)
    has_target = runtime_cfg['data']['target_column'] in df.columns

    seq_df, feature_columns = build_sequence_feature_frame(df, runtime_cfg, require_target=has_target)
    _, feature_columns = align_feature_columns(
        seq_df.loc[:, feature_columns],
        bundle.get('feature_columns'),
    )
    seq_windows = build_sliding_windows(
        seq_df,
        runtime_cfg,
        feature_columns=feature_columns,
        require_target=has_target,
    )

    scaler = load_pickle(artifacts_dir / bundle['scaler_file'])
    winner_model_path = artifacts_dir / bundle['model_files']['winner']
    model = load_model(winner_model_path)
    model['scaler'] = scaler

    proba = run_inference(model, seq_windows['X_seq'])
    pred = np.argmax(proba, axis=1) if len(proba) else np.zeros((0,), dtype=int)

    pred_dir = Path(cfg['paths']['predictions_dir'])
    out_path = pred_dir / 'predictions_sequence.csv'
    out = seq_windows['window_meta'].reset_index(drop=True).copy()
    if has_target and seq_windows['y_seq'].size:
        out['y_true'] = seq_windows['y_seq'].astype(int)
    out['pred'] = pred.astype(int)
    for i in range(proba.shape[1]):
        out[f'proba_{i}'] = proba[:, i]
    save_predictions(out, out_path)

    return {'sequence_predictions': str(out_path)}
