from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_processing import generate_and_export_synthetic_dataset, load_csv_dataset, prepare_processed_datasets
from src.evaluation import evaluate_pipeline
from src.get_stats import build_column_info
from src.predict.predictor import predict_with_bundle
from src.training.trainer import train_pipeline
from src.utils import get_logger, load_config


LOGGER = get_logger('src.main')


def data_processing(
    config_path: str = 'config/config.yaml',
    input_csv: str | None = None,
    synthetic: bool = False,
) -> dict[str, str]:
    cfg = load_config(config_path)

    generated_payload: dict[str, str] = {}
    if synthetic:
        if input_csv is not None:
            raise ValueError('No combines --synthetic con --input-csv.')
        LOGGER.info('Data processing | synthetic=true | generando dataset raw.')
        generated_payload = generate_and_export_synthetic_dataset(cfg)
        raw_path = Path(generated_payload['raw_dataset'])
    else:
        raw_path = Path(input_csv) if input_csv else Path(cfg['paths']['raw_dataset'])

    LOGGER.info('Data processing | input=%s', raw_path)
    df_raw = load_csv_dataset(raw_path)
    out = prepare_processed_datasets(df_raw, cfg)
    out.update(generated_payload)
    LOGGER.info('Data processing completado | processed=%s', out['processed_dataset'])
    return out


def train(
    config_path: str = 'config/config.yaml',
    dataset_path: str | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    processed_path = Path(dataset_path) if dataset_path else Path(cfg['paths']['processed_dataset'])

    if not processed_path.exists():
        raise FileNotFoundError(
            f'No existe dataset procesado: {processed_path}. Ejecuta scripts/data_processing.py primero.'
        )

    LOGGER.info('Training | dataset=%s | quick=%s', processed_path, quick)
    df = load_csv_dataset(processed_path)
    payload = train_pipeline(df, cfg, quick=quick)
    LOGGER.info('Training completado | metrics=%s', Path(cfg['paths']['models_metrics_dir']) / 'metrics_summary.json')
    return payload


def predict(
    config_path: str = 'config/config.yaml',
    input_csv: str | None = None,
) -> dict[str, str]:
    cfg = load_config(config_path)
    in_path = Path(input_csv) if input_csv else Path(cfg['paths']['processed_dataset'])

    if not in_path.exists():
        raise FileNotFoundError(
            f'No existe input_csv para inferencia: {in_path}. Ejecuta scripts/data_processing.py o pasa --input-csv.'
        )

    LOGGER.info('Predict | dataset=%s', in_path)
    df = load_csv_dataset(in_path)
    out = predict_with_bundle(df, cfg)
    LOGGER.info('Predict completado | sequence=%s', out['sequence_predictions'])
    return out


def evaluate(
    config_path: str = 'config/config.yaml',
    dataset_path: str | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    processed_path = Path(dataset_path) if dataset_path else Path(cfg['paths']['processed_dataset'])

    if not processed_path.exists():
        raise FileNotFoundError(
            f'No existe dataset procesado para evaluar: {processed_path}. Ejecuta scripts/data_processing.py primero.'
        )

    LOGGER.info('Evaluate | dataset=%s | quick=%s', processed_path, quick)
    df = load_csv_dataset(processed_path)
    payload = evaluate_pipeline(df, cfg, quick=quick)
    LOGGER.info('Evaluate completado | metrics=%s', Path(cfg['paths']['models_metrics_dir']) / 'metrics_summary.json')
    return payload


def get_stats(
    config_path: str = 'config/config.yaml',
    dataset_path: str | None = None,
) -> dict[str, str]:
    cfg = load_config(config_path)
    p = Path(dataset_path) if dataset_path else Path(cfg['paths']['processed_dataset'])
    if not p.exists():
        p = Path(cfg['paths']['raw_dataset'])
    if not p.exists():
        raise FileNotFoundError('No hay dataset para get_stats (ni processed ni raw).')

    LOGGER.info('Get stats | dataset=%s', p)
    df = load_csv_dataset(p)
    if cfg['data']['timestamp_column'] in df.columns:
        df[cfg['data']['timestamp_column']] = pd.to_datetime(df[cfg['data']['timestamp_column']], errors='coerce')

    out = build_column_info(df, cfg)
    stats_path = Path(cfg['paths']['stats_dataset'])
    LOGGER.info('Get stats completado | output=%s', stats_path)
    return {'stats_path': str(stats_path), 'n_columns': str(out.shape[0])}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Entry point principal de entrenamiento/inferencia/evaluacion/stats.')
    parser.add_argument(
        'command',
        choices=['train', 'evaluate', 'predict', 'data_processing', 'get_stats'],
        help='Comando a ejecutar.',
    )
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--dataset-path', default=None, help='Ruta dataset para train/evaluate/get_stats.')
    parser.add_argument('--input-csv', default=None, help='Ruta dataset para predict/data_processing.')
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Genera dataset sintetico (raw + metadata + summary) antes de data_processing.',
    )
    parser.add_argument('--quick', action='store_true', help='Modo rapido para train/evaluate.')
    return parser


def cli(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == 'train':
        train(config_path=args.config, dataset_path=args.dataset_path, quick=bool(args.quick))
        return
    if args.command == 'evaluate':
        evaluate(config_path=args.config, dataset_path=args.dataset_path, quick=bool(args.quick))
        return
    if args.command == 'predict':
        predict(config_path=args.config, input_csv=args.input_csv)
        return
    if args.command == 'data_processing':
        data_processing(config_path=args.config, input_csv=args.input_csv, synthetic=bool(args.synthetic))
        return
    if args.command == 'get_stats':
        get_stats(config_path=args.config, dataset_path=args.dataset_path)
        return

    raise ValueError(f'Comando no soportado: {args.command}')


if __name__ == '__main__':
    cli()
