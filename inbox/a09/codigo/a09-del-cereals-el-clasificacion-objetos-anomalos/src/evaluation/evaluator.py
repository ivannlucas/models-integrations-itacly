from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from src.data_processing.preprocess import build_sequence_feature_frame, build_sliding_windows, split_sample_ids_stratified
from src.utils.model_bundle import align_feature_columns, merge_runtime_config_with_bundle
from src.training.sequential import evaluate_sequence_model, load_checkpoint, load_pickle, predict_sequence_proba, transform_sequences


CLASS_LABEL_NAMES = {0: 'sano', 1: 'insectos', 2: 'moho_critico'}


def ensure_out_dirs(cfg: dict[str, Any]) -> None:
    for p in [
        cfg['paths']['predictions_dir'],
        cfg['paths']['splits_dir'],
        cfg['paths']['models_artifacts_dir'],
        cfg['paths']['models_metrics_dir'],
        cfg['paths']['metrics_figures_dir'],
    ]:
        Path(p).mkdir(parents=True, exist_ok=True)


def _metric_from_outputs(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, selection_metric: str) -> float:
    selection_metric = str(selection_metric).lower().strip()
    labels = sorted(np.unique(y_true).tolist()) if len(y_true) else [0, 1, 2]
    metric_values = {
        'accuracy': float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)) if len(y_true) else 0.0,
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)) if len(y_true) else 0.0,
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)) if len(y_true) else 0.0,
        'log_loss': float(log_loss(y_true, y_proba, labels=labels)) if len(y_true) else 0.0,
    }
    return float(metric_values.get(selection_metric, metric_values['f1_macro']))


def _compute_permutation_importance(
    model,
    X_scaled: np.ndarray,
    y_true: np.ndarray,
    feature_columns: list[str],
    *,
    selection_metric: str,
    batch_size: int,
    device: str,
    random_state: int,
    n_repeats: int = 1,
) -> pd.DataFrame:
    if len(X_scaled) == 0 or not feature_columns:
        return pd.DataFrame(columns=['feature', 'baseline_score', 'permuted_score_mean', 'importance_mean', 'importance_std'])

    rng = np.random.default_rng(int(random_state))
    baseline_proba = predict_sequence_proba(model, X_scaled, batch_size=batch_size, device=device)
    baseline_pred = np.argmax(baseline_proba, axis=1) if len(baseline_proba) else np.zeros((0,), dtype=int)
    baseline_score = _metric_from_outputs(y_true, baseline_pred, baseline_proba, selection_metric)

    rows: list[dict[str, Any]] = []
    for feature_idx, feature_name in enumerate(feature_columns):
        repeated_scores: list[float] = []
        for _ in range(max(1, int(n_repeats))):
            X_perm = X_scaled.copy()
            flat = X_perm[:, :, feature_idx].reshape(-1)
            shuffled = rng.permutation(flat)
            X_perm[:, :, feature_idx] = shuffled.reshape(X_perm.shape[0], X_perm.shape[1])
            perm_proba = predict_sequence_proba(model, X_perm, batch_size=batch_size, device=device)
            perm_pred = np.argmax(perm_proba, axis=1) if len(perm_proba) else np.zeros((0,), dtype=int)
            perm_score = _metric_from_outputs(y_true, perm_pred, perm_proba, selection_metric)
            repeated_scores.append(float(perm_score))

        perm_mean = float(np.mean(repeated_scores)) if repeated_scores else baseline_score
        perm_std = float(np.std(repeated_scores, ddof=0)) if repeated_scores else 0.0
        rows.append(
            {
                'feature': feature_name,
                'baseline_score': baseline_score,
                'permuted_score_mean': perm_mean,
                'permuted_score_std': perm_std,
                'importance_mean': float(baseline_score - perm_mean),
                'importance_std': perm_std,
            }
        )

    importance_df = pd.DataFrame(rows).sort_values('importance_mean', ascending=False).reset_index(drop=True)
    return importance_df


def _plot_feature_importance(importance_df: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    if importance_df.empty:
        return

    plot_df = importance_df.head(max(1, int(top_n))).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(11.5, max(5.8, 0.42 * len(plot_df) + 1.6)))
    colors = ['#d55e00' if v >= 0 else '#999999' for v in plot_df['importance_mean']]
    ax.barh(plot_df['feature'], plot_df['importance_mean'], color=colors)
    ax.axvline(0.0, color='black', linewidth=1)
    ax.set_xlabel('Caida de la metrica principal al permutar la feature')
    ax.set_title('Importancia por permutacion | clasificador ganador')
    ax.grid(True, axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist()) if len(y_true) else [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    names = [CLASS_LABEL_NAMES.get(int(x), str(x)) for x in labels]
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    im = ax.imshow(cmn, cmap='Oranges', vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel('Clase predicha')
    ax.set_ylabel('Clase real')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(names, rotation=25, ha='right')
    ax.set_yticklabels(names)
    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            value = cmn[i, j]
            ax.text(j, i, f'{value:.0%}', ha='center', va='center', fontsize=9, color=('white' if value > 0.6 else 'black'))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_temporal_detection_trace(predictions_path: Path, out_path: Path) -> None:
    if not predictions_path.exists():
        return
    df = pd.read_csv(predictions_path)
    if df.empty or 'sample_id' not in df.columns or 'timestamp_end' not in df.columns:
        return

    def _score(group: pd.DataFrame) -> tuple[int, int, int, int]:
        true_change = int((group['y_true'] != group['y_true'].shift()).sum()) if 'y_true' in group.columns else 0
        pred_change = int((group['pred'] != group['pred'].shift()).sum()) if 'pred' in group.columns else 0
        true_nunique = int(group['y_true'].nunique()) if 'y_true' in group.columns else 0
        pred_nunique = int(group['pred'].nunique()) if 'pred' in group.columns else 0
        return (true_change, pred_change, true_nunique, pred_nunique)

    ranked = []
    for sample_id, group in df.groupby('sample_id'):
        ranked.append((_score(group), sample_id))
    ranked.sort(reverse=True)
    sample_id = ranked[0][1] if ranked else None
    trace = df[df['sample_id'] == sample_id].copy() if sample_id is not None else pd.DataFrame()
    if trace.empty:
        return

    trace['timestamp_end'] = pd.to_datetime(trace['timestamp_end'], errors='coerce')
    trace = trace.sort_values('timestamp_end')
    fig, ax = plt.subplots(figsize=(11, 5.2))

    if 'proba_1' in trace.columns:
        ax.plot(trace['timestamp_end'], trace['proba_1'], label='P(insectos)', color='#d55e00', linewidth=2)
    if 'proba_2' in trace.columns:
        ax.plot(trace['timestamp_end'], trace['proba_2'], label='P(moho_critico)', color='#7a5195', linewidth=2)
    if 'proba_0' in trace.columns:
        ax.plot(trace['timestamp_end'], trace['proba_0'], label='P(sano)', color='#1f77b4', linewidth=2)

    if 'y_true' in trace.columns:
        ax.step(trace['timestamp_end'], trace['y_true'], where='post', label='Clase real', color='black', alpha=0.5, linewidth=1.4)
    if 'pred' in trace.columns:
        ax.step(trace['timestamp_end'], trace['pred'], where='post', label='Clase predicha', color='#2ca02c', alpha=0.6, linewidth=1.4)

    ax.set_title(f'Evolucion temporal de una serie de hold-out | {sample_id}')
    ax.set_xlabel('timestamp_end')
    ax.set_ylabel('Probabilidad / clase')
    ax.set_ylim(-0.2, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper left', ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_initial_context_trace(predictions_path: Path, out_path: Path, n_windows: int = 8) -> None:
    if not predictions_path.exists():
        return
    df = pd.read_csv(predictions_path)
    if df.empty or 'sample_id' not in df.columns or 'window_index' not in df.columns:
        return

    candidates = []
    for sample_id, group in df.groupby('sample_id'):
        g = group.sort_values('window_index').head(n_windows).copy()
        if g.empty:
            continue
        early_critical = int(g['pred'].isin([1, 2]).sum()) if 'pred' in g.columns else 0
        early_change = int((g['pred'] != g['pred'].shift()).sum()) if 'pred' in g.columns else 0
        early_true_change = int((g['y_true'] != g['y_true'].shift()).sum()) if 'y_true' in g.columns else 0
        score = (early_critical, early_change, early_true_change, len(g))
        candidates.append((score, sample_id))
    if not candidates:
        return
    candidates.sort(reverse=True)
    sample_id = candidates[0][1]
    trace = df[df['sample_id'] == sample_id].sort_values('window_index').head(n_windows).copy()
    if trace.empty:
        return

    fig, (ax_prob, ax_cls) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, gridspec_kw={'height_ratios': [2.4, 1.0]})
    x = trace['window_index'].astype(int).to_list()

    if 'proba_0' in trace.columns:
        ax_prob.plot(x, trace['proba_0'], marker='o', label='P(sano)', color='#1f77b4', linewidth=2)
    if 'proba_1' in trace.columns:
        ax_prob.plot(x, trace['proba_1'], marker='o', label='P(insectos)', color='#d55e00', linewidth=2)
    if 'proba_2' in trace.columns:
        ax_prob.plot(x, trace['proba_2'], marker='o', label='P(moho_critico)', color='#7a5195', linewidth=2)
    ax_prob.set_ylabel('Probabilidad')
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.grid(True, alpha=0.25)
    ax_prob.legend(loc='upper left', ncol=3)

    if 'y_true' in trace.columns:
        ax_cls.plot(x, trace['y_true'], marker='o', linewidth=2, color='black', label='Clase real')
    if 'pred' in trace.columns:
        ax_cls.plot(x, trace['pred'], marker='o', linewidth=2, color='#2ca02c', label='Clase predicha')
    ax_cls.set_yticks([0, 1, 2])
    ax_cls.set_yticklabels(['sano', 'insectos', 'moho'])
    ax_cls.set_xlabel('window_index')
    ax_cls.set_ylabel('Clase')
    ax_cls.set_ylim(-0.3, 2.3)
    ax_cls.grid(True, alpha=0.25)
    ax_cls.legend(loc='upper left')

    fig.suptitle(f'Arranque temporal de una serie de hold-out | {sample_id}')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _classification_report_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist()) if len(y_true) else [0, 1, 2]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame(
        {
            'class_id': labels,
            'class_name': [CLASS_LABEL_NAMES.get(int(c), str(c)) for c in labels],
            'precision': p,
            'recall': r,
            'f1': f,
            'support': s,
        }
    )


def _build_acceptance_check(
    winner_metrics: dict[str, Any],
    winner_report_df: pd.DataFrame,
    acceptance_cfg: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    min_metric_map = {
        'f1_macro_min': 'f1_macro',
        'recall_macro_min': 'recall_macro',
        'accuracy_min': 'accuracy',
        'balanced_accuracy_min': 'balanced_accuracy',
        'precision_macro_min': 'precision_macro',
    }
    for cfg_key, metric_name in min_metric_map.items():
        if cfg_key not in acceptance_cfg:
            continue
        threshold = float(acceptance_cfg[cfg_key])
        actual = float(winner_metrics.get(metric_name, 0.0))
        checks.append(
            {
                'name': cfg_key,
                'metric': metric_name,
                'operator': '>=',
                'required': threshold,
                'actual': actual,
                'passed': bool(actual >= threshold),
            }
        )

    if 'log_loss_max' in acceptance_cfg:
        threshold = float(acceptance_cfg['log_loss_max'])
        actual = float(winner_metrics.get('log_loss', 0.0))
        checks.append(
            {
                'name': 'log_loss_max',
                'metric': 'log_loss',
                'operator': '<=',
                'required': threshold,
                'actual': actual,
                'passed': bool(actual <= threshold),
            }
        )

    class_recall_cfg = acceptance_cfg.get('class_recall_min', {})
    if isinstance(class_recall_cfg, dict) and not winner_report_df.empty:
        report_index = winner_report_df.set_index('class_name')
        for class_name, threshold_value in class_recall_cfg.items():
            if class_name not in report_index.index:
                checks.append(
                    {
                        'name': f'class_recall_min.{class_name}',
                        'metric': f'{class_name}.recall',
                        'operator': '>=',
                        'required': float(threshold_value),
                        'actual': None,
                        'passed': False,
                        'note': 'Clase no disponible en el reporte de evaluacion.',
                    }
                )
                continue

            actual = float(report_index.loc[class_name, 'recall'])
            threshold = float(threshold_value)
            checks.append(
                {
                    'name': f'class_recall_min.{class_name}',
                    'metric': f'{class_name}.recall',
                    'operator': '>=',
                    'required': threshold,
                    'actual': actual,
                    'passed': bool(actual >= threshold),
                }
            )

    return {
        'enabled': bool(acceptance_cfg),
        'overall_passed': bool(all(check['passed'] for check in checks)) if checks else True,
        'checks': checks,
    }


def _save_predictions(
    pred_dir: Path,
    model_name: str,
    test_meta: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Path:
    out = test_meta.reset_index(drop=True).copy()
    out['y_true'] = y_true.astype(int)
    out['pred'] = y_pred.astype(int)
    for i in range(y_proba.shape[1]):
        out[f'proba_{i}'] = y_proba[:, i]
    path = pred_dir / f'holdout_predictions_{model_name}.csv'
    out.to_csv(path, index=False)
    return path


def evaluate_with_sequence_models(
    cfg: dict[str, Any],
    *,
    seq_df: pd.DataFrame,
    feature_columns: list[str],
    split_ids: dict[str, np.ndarray],
    scaler,
    model_paths: dict[str, Path],
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_meta: pd.DataFrame,
    selection_metric: str,
    device: str,
    batch_size: int,
    quick: bool = False,
) -> dict[str, Any]:
    del seq_df, split_ids, quick
    ensure_out_dirs(cfg)
    metrics_dir = Path(cfg['paths']['models_metrics_dir'])
    figures_dir = Path(cfg['paths']['metrics_figures_dir'])
    pred_dir = Path(cfg['paths']['predictions_dir'])

    model_outputs: dict[str, dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []

    X_test_scaled = transform_sequences(scaler, X_test)
    for model_name, model_path in model_paths.items():
        if model_name == 'winner':
            continue
        checkpoint = load_checkpoint(model_path, map_location='cpu')
        model = checkpoint['model']
        result = evaluate_sequence_model(model, X_test_scaled, y_test, batch_size=batch_size, device=device)
        y_pred = result['y_pred']
        y_proba = result['y_proba']
        metrics = result['metrics']
        report_df = _classification_report_df(y_test, y_pred)
        report_df.to_csv(metrics_dir / f'{model_name}_class_report.csv', index=False)
        pred_path = _save_predictions(pred_dir, model_name, test_meta, y_test, y_pred, y_proba)
        _plot_confusion(y_test, y_pred, figures_dir / f'{model_name}_confusion.png', f'Matriz de confusion | {model_name.upper()}')
        model_outputs[model_name] = {
            'checkpoint': checkpoint,
            'metrics': metrics,
            'report_df': report_df,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'predictions_file': str(pred_path),
        }
        comparison_rows.append(
            {
                'model_name': model_name,
                'selection_metric': selection_metric,
                'selection_value': float(metrics.get(selection_metric, metrics.get('f1_macro', 0.0))),
                **metrics,
                'predictions_file': str(pred_path),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values('selection_value', ascending=False).reset_index(drop=True)
    comparison_df.to_csv(metrics_dir / 'model_comparison.csv', index=False)

    winner_name = comparison_df.iloc[0]['model_name'] if len(comparison_df) else 'winner'
    winner_output = model_outputs[winner_name]
    winner_metrics = winner_output['metrics']
    winner_report_df = winner_output['report_df']
    acceptance_cfg = dict(cfg.get('sequence', {}).get('acceptance_thresholds', {}) or {})
    acceptance_check = _build_acceptance_check(winner_metrics, winner_report_df, acceptance_cfg)
    winner_pred_path = pred_dir / 'holdout_predictions_winner.csv'
    shutil.copy2(winner_output['predictions_file'], winner_pred_path)
    winner_confusion_path = figures_dir / 'winner_confusion.png'
    selected_confusion = figures_dir / f'{winner_name}_confusion.png'
    if selected_confusion.exists():
        shutil.copy2(selected_confusion, winner_confusion_path)
    temporal_trace_path = figures_dir / 'winner_temporal_trace.png'
    _plot_temporal_detection_trace(winner_pred_path, temporal_trace_path)
    initial_context_path = figures_dir / 'winner_initial_context.png'
    _plot_initial_context_trace(winner_pred_path, initial_context_path)

    feature_importance_plot_path = figures_dir / 'winner_feature_importance.png'
    winner_checkpoint = winner_output['checkpoint']
    winner_model = winner_checkpoint['model']
    importance_df = _compute_permutation_importance(
        winner_model,
        X_test_scaled,
        y_test,
        feature_columns,
        selection_metric=selection_metric,
        batch_size=batch_size,
        device=device,
        random_state=int(cfg.get('training', {}).get('random_state', 42)),
        n_repeats=int(cfg.get('sequence', {}).get('training', {}).get('permutation_importance_repeats', 1)),
    )
    _plot_feature_importance(importance_df, feature_importance_plot_path, top_n=20)

    metrics_summary = {
        'pipeline_type': 'sequence_lstm_gru',
        'selection_metric': selection_metric,
        'best_model_name': winner_name,
        'best_model_metric': float(winner_metrics.get(selection_metric, winner_metrics.get('f1_macro', 0.0))),
        'feature_columns': feature_columns,
        'n_windows_test': int(len(y_test)),
        'holdout_metrics': {
            name: output['metrics'] for name, output in model_outputs.items()
        },
        'acceptance_thresholds': acceptance_cfg,
        'acceptance_check': acceptance_check,
        'comparison': comparison_df.to_dict(orient='records'),
        'model_files': {name: str(path) for name, path in model_paths.items()},
        'winner_predictions_file': str(winner_pred_path),
        'winner_confusion_file': str(winner_confusion_path),
        'winner_temporal_trace_file': str(temporal_trace_path),
        'winner_initial_context_file': str(initial_context_path),
        'winner_feature_importance_plot': str(feature_importance_plot_path),
    }
    with open(metrics_dir / 'metrics_summary.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    markdown_lines = [
        '# Reporte de evaluacion del modelo secuencial',
        '',
        '## Resumen de metricas',
        f'- Metrica de seleccion: {selection_metric}',
        f'- Numero de ventanas de hold-out: {len(y_test)}',
        '',
        '## Que significa cada metrica',
        '- Accuracy: porcentaje total de aciertos.',
        '- Balanced accuracy: media del recall por clase; ayuda cuando hay desbalance.',
        '- Precision macro: precision media entre clases, tratando todas por igual.',
        '- Recall macro: capacidad media de recuperar cada clase.',
        '- F1 macro: equilibrio entre precision y recall; es la metrica principal del pipeline.',
        '- Log loss: calidad de las probabilidades; penaliza predicciones seguras pero equivocadas.',
        '',
        '## Comparativa LSTM vs GRU',
    ]
    for row in comparison_rows:
        markdown_lines.extend(
            [
                f"- {row['model_name'].upper()}: {selection_metric}={row['selection_value']:.4f}, accuracy={row['accuracy']:.4f}, balanced_accuracy={row['balanced_accuracy']:.4f}, precision_macro={row['precision_macro']:.4f}, recall_macro={row['recall_macro']:.4f}, log_loss={row['log_loss']:.4f}",
            ]
        )
    markdown_lines.extend(
        [
            '',
            '## Modelo ganador',
            f'- {winner_name.upper()}',
            f'- Metrica principal ({selection_metric}): {float(winner_metrics.get(selection_metric, winner_metrics.get("f1_macro", 0.0))):.4f}',
            '',
            '## Criterios de aceptacion',
        '',
        ]
    )
    if acceptance_check['checks']:
        markdown_lines.append(f"- Resultado global: {'cumplido' if acceptance_check['overall_passed'] else 'no cumplido'}")
        for check in acceptance_check['checks']:
            if str(check.get('metric', '')).startswith('moho_critico.'):
                continue
            actual_value = 'n/d' if check['actual'] is None else f"{float(check['actual']):.4f}"
            markdown_lines.append(
                f"- {check['metric']} {check['operator']} {float(check['required']):.4f}: actual={actual_value} -> {'OK' if check['passed'] else 'NO OK'}"
            )
    else:
        markdown_lines.append('- No se configuraron umbrales minimos de aceptacion.')
    markdown_lines.extend(
        [
            '',
            '## Lectura rapida',
            'El modelo ganador es el que mejor equilibra acierto global, estabilidad por clase y calidad probabilistica sobre el hold-out.',
        ]
    )
    (metrics_dir / 'reporte_modelo.md').write_text('\n'.join(markdown_lines), encoding='utf-8')

    return {
        'comparison': comparison_df,
        'metrics_summary': metrics_summary,
        'best_model_name': winner_name,
        'best_model_metric': float(winner_metrics.get(selection_metric, winner_metrics.get('f1_macro', 0.0))),
        'predictions': {name: output['predictions_file'] for name, output in model_outputs.items()} | {'winner': str(winner_pred_path)},
    }


def evaluate_pipeline(df: pd.DataFrame, cfg: dict[str, Any], *, quick: bool = False) -> dict[str, Any]:
    del quick
    artifacts_dir = Path(cfg['paths']['models_artifacts_dir'])
    meta_path = artifacts_dir / 'model_bundle_metadata.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'No existe metadata de bundle: {meta_path}. Ejecuta train antes de evaluate.')

    with open(meta_path, 'r', encoding='utf-8') as f:
        bundle = json.load(f)

    runtime_cfg = merge_runtime_config_with_bundle(cfg, bundle)
    seq_df, feature_columns = build_sequence_feature_frame(df, runtime_cfg)
    _, feature_columns = align_feature_columns(
        seq_df.loc[:, feature_columns],
        bundle.get('feature_columns'),
    )
    split_ids_raw = bundle.get('split_sample_ids', {})
    split_ids = {
        'train_sample_ids': np.asarray(split_ids_raw.get('train_sample_ids', [])),
        'validation_sample_ids': np.asarray(split_ids_raw.get('validation_sample_ids', [])),
        'test_sample_ids': np.asarray(split_ids_raw.get('test_sample_ids', [])),
    }
    test_payload = build_sliding_windows(
        seq_df,
        runtime_cfg,
        feature_columns=feature_columns,
        sample_ids=split_ids['test_sample_ids'],
    )

    scaler = load_pickle(artifacts_dir / bundle['scaler_file'])
    eval_cfg = runtime_cfg
    model_paths = {
        'lstm': artifacts_dir / bundle['model_files']['lstm'],
        'gru': artifacts_dir / bundle['model_files']['gru'],
        'winner': artifacts_dir / bundle['model_files']['winner'],
    }
    requested_device = str(runtime_cfg['training'].get('device', 'cpu')).lower().strip()
    device = 'cuda' if requested_device in {'cuda', 'gpu'} and torch.cuda.is_available() else 'cpu'
    return evaluate_with_sequence_models(
        eval_cfg,
        seq_df=seq_df,
        feature_columns=feature_columns,
        split_ids=split_ids,
        scaler=scaler,
        model_paths=model_paths,
        X_test=test_payload['X_seq'],
        y_test=test_payload['y_seq'],
        test_meta=test_payload['window_meta'],
        selection_metric=str(bundle.get('selection_metric', runtime_cfg.get('sequence', {}).get('selection_metric', 'f1_macro'))),
        device=device,
        batch_size=int(runtime_cfg.get('sequence', {}).get('training', {}).get('eval_batch_size', 128)),
    )
