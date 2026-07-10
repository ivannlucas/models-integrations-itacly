"""
DATAGIA — Main entry point.

Provides the four top-level pipeline functions required by the project
guidelines:

    - data_processing()  : raw CSV → processed CSV + splits
    - train()            : processed data → trained model + metrics
    - predict()          : model + input data → predictions CSV
    - optimize()         : scenarios → GA optimal setpoints CSV
    - get_stats()        : dataset → column info + baseline KPIs

Each function can run stand-alone or be invoked via the corresponding
script in ``scripts/``.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.utils.reproducibility import set_seed, get_device
from src.utils.paths import (
    ensure_dirs,
    PROCESSED_DATASET_PATH,
    PREDICTIONS_DIR,
    METRICS_DIR,
    BASELINE_METRICS_PATH,
    TRAIN_METRICS_PATH,
    EVAL_RT_REPORT_PATH,
)
from src.utils.constants import FEATURES, TARGETS, T_SAFETY

logger = get_logger(__name__)


# ============================================================================
# 1. DATA PROCESSING
# ============================================================================

def data_processing() -> None:
    """
    Generate raw data (if needed), filter production records and create splits.

    Pipeline:
        1. Run the pasteurization simulator → ``data/raw/``
        2. Filter CIP cleaning records → ``data/processed/``
        3. Temporal split by quartiles → ``data/splits/``

    This function is **not** called inside train() or predict().
    It exists solely for reproducibility: allows recreating all data
    artifacts from scratch.
    """
    ensure_dirs()
    set_seed(1)
    logger.info("=== DATA PROCESSING ===")

    # Step 1 — Simulate
    from src.data_processing.simulator import generar_dataset_pasteurizacion

    logger.info("Step 1/3: Generating synthetic dataset with simulator...")
    df_raw = generar_dataset_pasteurizacion(save=True)
    logger.info(f"  Raw dataset: {len(df_raw)} records")

    # Step 2 — Filter
    from src.data_processing.preprocessing import (
        filter_production,
        save_processed_data,
        temporal_split_by_quartiles,
        normalize_data,
        save_splits,
    )

    logger.info("Step 2/3: Filtering production records (removing CIP)...")
    df_prod = filter_production(df_raw)
    save_processed_data(df_prod)
    logger.info(f"  Production dataset: {len(df_prod)} records")

    # Step 3 — Split & save
    logger.info("Step 3/3: Temporal split by quartiles (70/15/15)...")
    splits = temporal_split_by_quartiles(df_prod)
    save_splits(splits)
    logger.info(
        f"  Train: {splits['X_train'].shape[0]} | "
        f"Val: {splits['X_val'].shape[0]} | "
        f"Test: {splits['X_test'].shape[0]}"
    )

    logger.info("Data processing complete.")


# ============================================================================
# 2. TRAIN
# ============================================================================

def train() -> None:
    """
    Train the MLP model on processed data and save all artifacts.

    Pipeline:
        1. Load processed data and create splits.
        2. Normalize with MinMaxScaler (fit on train only).
        3. Train DynamicMLP with early stopping.
        4. Evaluate on test set (RMSE, MAE, R²).
        5. Save model/scalers/config and persist verified metrics from saved artifacts.

    Inputs:
        - ``data/processed/final_data_sim.csv``
        - Configuration from ``src.utils.constants``

    Outputs:
        - ``models/artifacts/mlp_predictor.pt``
        - ``models/artifacts/model_config.json``
        - ``models/artifacts/scaler_X.pkl``, ``scaler_y.pkl``
        - ``models/metrics/train_metrics.json``
        - ``data/splits/train.csv``, ``val.csv``, ``test.csv``
    """
    ensure_dirs()
    set_seed(1)
    device = get_device()
    logger.info("=== TRAIN ===")
    logger.info(f"Device: {device}")

    # 1. Load & split
    from src.data_processing.preprocessing import (
        load_processed_data,
        filter_production,
        temporal_split_by_quartiles,
        normalize_data,
        save_splits,
    )

    logger.info("Step 1/5: Loading processed data and splitting...")
    df = load_processed_data()
    df_prod = filter_production(df)
    splits = temporal_split_by_quartiles(df_prod)
    save_splits(splits)

    # 2. Normalize
    logger.info("Step 2/5: Normalizing (MinMaxScaler on train)...")
    scaled, scaler_X, scaler_y = normalize_data(splits)

    # 3. Build & train — hyperparameters loaded from model_config.json (set by
    #    the tuning notebook). Falls back to a safe default if the file does
    #    not yet exist.
    import json as _json
    from src.training.model import DynamicMLP
    from src.training.trainer import train_model as _train_model
    from src.utils.paths import MODEL_CONFIG_PATH

    if MODEL_CONFIG_PATH.exists():
        with open(MODEL_CONFIG_PATH, "r") as _f:
            _cfg = _json.load(_f)
        best_params = {
            "num_layers": _cfg["num_layers"],
            "neurons":    _cfg["neurons"],
            "lr":         _cfg["lr"],
            "activation": _cfg["activation"],
        }
        train_batch_size = 128
        train_patience = 15
        logger.info(
            f"Step 3/5: Training DynamicMLP with tuned params from model_config.json: "
            f"{best_params}"
        )
    else:
        # Fallback aligned with config defaults (config/config.yaml training):
        # 2-layer MLP, 128 neurons, ReLU, batch_size=128, patience=15.
        best_params = {"num_layers": 2, "neurons": 128, "lr": 0.0005, "activation": "ReLU"}
        train_batch_size = 128
        train_patience = 15
        logger.warning(
            "Step 3/5: model_config.json not found — using default fallback params "
            "from the project configuration."
        )
        logger.info(f"  Fallback params: {best_params}")
        logger.info(
            f"  Fallback training settings: batch_size={train_batch_size} | "
            f"patience={train_patience}"
        )

    model = DynamicMLP(
        input_size=len(FEATURES),
        output_size=len(TARGETS),
        num_layers=best_params["num_layers"],
        neurons=best_params["neurons"],
        activation=best_params["activation"],
    ).to(device)
    model, info = _train_model(
        model,
        scaled["X_train"], scaled["y_train"],
        scaled["X_val"], scaled["y_val"],
        lr=best_params["lr"], epochs=300,
        batch_size=train_batch_size, patience=train_patience,
    )
    logger.info(
        f"  Epochs: {info['epochs_executed']} | "
        f"Best val MSE: {info['best_val_loss']:.6f}"
    )

    # 4. Evaluate on train, val and test
    from src.predict.inference import predict_batch
    from src.get_stats.metrics import compute_regression_metrics

    logger.info("Step 4/5: Evaluating on train, val and test sets...")
    
    y_pred_train = predict_batch(splits["X_train"], model, scaler_X, scaler_y)
    train_metrics = compute_regression_metrics(splits["y_train"], y_pred_train)
    
    y_pred_val = predict_batch(splits["X_val"], model, scaler_X, scaler_y)
    val_metrics = compute_regression_metrics(splits["y_val"], y_pred_val)

    y_pred = predict_batch(splits["X_test"], model, scaler_X, scaler_y)
    metrics = compute_regression_metrics(splits["y_test"], y_pred)

    logger.info("--- Test Set Metrics ---")
    for target_name, m in metrics.items():
        logger.info(
            f"  {target_name}: RMSE={m['RMSE']:.4f} | "
            f"MAE={m['MAE']:.4f} | R²={m['R2']:.6f}"
        )

    # 5. Save
    from src.training.artifacts import save_artifacts, load_artifacts

    logger.info("Step 5/5: Saving artifacts and verifying persisted metrics...")

    # Save production artifacts first.
    save_artifacts(model, scaler_X, scaler_y, best_params, metrics=None)

    # Re-load from disk and recompute test metrics so train_metrics.json always
    # matches the actual saved checkpoint and scalers.
    saved_model, saved_scaler_X, saved_scaler_y, _ = load_artifacts()
    
    y_pred_saved_train = predict_batch(splits["X_train"], saved_model, saved_scaler_X, saved_scaler_y)
    verified_train_metrics = compute_regression_metrics(splits["y_train"], y_pred_saved_train)
    
    y_pred_saved_val = predict_batch(splits["X_val"], saved_model, saved_scaler_X, saved_scaler_y)
    verified_val_metrics = compute_regression_metrics(splits["y_val"], y_pred_saved_val)
    
    y_pred_saved = predict_batch(
        splits["X_test"], saved_model, saved_scaler_X, saved_scaler_y
    )
    verified_metrics = compute_regression_metrics(splits["y_test"], y_pred_saved)

    final_metrics = {
        "train": verified_train_metrics,
        "val": verified_val_metrics,
        "test": verified_metrics
    }

    max_abs_delta = 0.0
    for target_name in metrics:
        for metric_name in ("RMSE", "MAE", "R2"):
            delta = abs(
                metrics[target_name][metric_name]
                - verified_metrics[target_name][metric_name]
            )
            max_abs_delta = max(max_abs_delta, delta)

    save_artifacts(model, scaler_X, scaler_y, best_params, metrics=final_metrics)

    logger.info(f"  Verified metrics saved to {TRAIN_METRICS_PATH}")
    logger.info(f"  Max abs delta vs in-memory eval: {max_abs_delta:.3e}")
    logger.info("Training complete. Artifacts saved to models/artifacts/")


# ============================================================================
# 3. PREDICT
# ============================================================================

def predict(input_path: str = None, output_path: str = None) -> None:
    """
    Run inference on a dataset and save predictions.

    Parameters
    ----------
    input_path : str, optional
        Path to input CSV (must have FEATURES columns).
        Default: ``data/splits/test.csv``.
    output_path : str, optional
        Path where predictions CSV will be saved.
        Default: ``data/predictions/predictions.csv``.

    Outputs:
        - CSV file with columns: FEATURES + E_consumo_pred + T_out_pred
    """
    ensure_dirs()
    logger.info("=== PREDICT ===")

    # Load artifacts
    from src.training.artifacts import load_artifacts
    from src.predict.inference import predict_batch, save_predictions
    from src.utils.paths import TEST_SPLIT_PATH

    model, scaler_X, scaler_y, config = load_artifacts()
    logger.info("Model and scalers loaded.")

    # Load input data
    in_path = input_path or str(TEST_SPLIT_PATH)
    df_input = pd.read_csv(in_path)
    logger.info(f"Input data: {in_path} ({len(df_input)} rows)")

    X_raw = df_input[FEATURES].values
    y_pred = predict_batch(X_raw, model, scaler_X, scaler_y)

    # Build output DataFrame
    df_pred = df_input.copy()
    df_pred["E_consumo_pred"] = y_pred[:, 0]
    df_pred["T_out_pred"] = y_pred[:, 1]

    # Save
    out_path = output_path or str(PREDICTIONS_DIR / "predictions.csv")
    saved = save_predictions(df_pred, out_path=out_path)
    logger.info(f"Predictions saved to {saved}")


def _build_backtesting_report(
    df_opt: pd.DataFrame,
    elapsed_s: float,
    ga_config: dict,
) -> dict:
    """Build the KPI report for real-time backtesting runs."""
    n_rows = int(len(df_opt))
    pct_factibles = float(df_opt["IA_factible"].mean() * 100.0) if n_rows else 0.0
    total_s = float(elapsed_s)
    mean_ms = float(elapsed_s / max(n_rows, 1) * 1000.0)

    hist_e_mean = float(df_opt["HIST_E_consumo"].mean())
    ia_e_mean = float(df_opt["IA_E_consumo"].mean())
    ahorro_medio = hist_e_mean - ia_e_mean
    # Mean of the per-row percentage (matches ga_evaluation.ipynb), NOT the
    # percentage of the means — these differ because HIST_E_consumo varies
    # row to row and is not linearly related to its own mean.
    ahorro_pct = float(df_opt["Ahorro_pct"].mean()) if "Ahorro_pct" in df_opt else (
        (ahorro_medio / hist_e_mean * 100.0) if hist_e_mean else 0.0
    )
    ahorro_total = float((df_opt["HIST_E_consumo"] - df_opt["IA_E_consumo"]).sum())

    hist_flow = df_opt["HIST_F_flow"].replace(0, np.nan)
    ia_flow = df_opt["IA_F_flow"].replace(0, np.nan)
    hist_ef = float((df_opt["HIST_E_consumo"] / hist_flow).mean())
    ia_ef = float((df_opt["IA_E_consumo"] / ia_flow).mean())
    if np.isnan(hist_ef):
        hist_ef = 0.0
    if np.isnan(ia_ef):
        ia_ef = 0.0
    mejora_abs = hist_ef - ia_ef
    mejora_pct = (mejora_abs / hist_ef * 100.0) if hist_ef else 0.0

    hist_flow_mean = float(df_opt["HIST_F_flow"].mean())
    ia_flow_mean = float(df_opt["IA_F_flow"].mean())
    delta_flow = ia_flow_mean - hist_flow_mean
    delta_flow_pct = (delta_flow / hist_flow_mean * 100.0) if hist_flow_mean else 0.0

    hist_tout = df_opt["HIST_T_out"]
    ia_tout = df_opt["IA_T_out"]
    compliance_hist = float((hist_tout >= T_SAFETY).mean() * 100.0) if n_rows else 0.0
    compliance_ia = float((ia_tout >= T_SAFETY).mean() * 100.0) if n_rows else 0.0

    hist_tserv = float(df_opt["HIST_T_servicio"].mean())
    ia_tserv = float(df_opt["IA_T_servicio"].mean())
    delta_tserv = ia_tserv - hist_tserv

    methodology = (
        "GA uni-objetivo (pop={pop}, n_gen={n_gen}, "
        "cxpb={cxpb}, mutpb={mutpb}) per scenario"
    ).format(
        pop=ga_config["pop_size"],
        n_gen=ga_config["n_gen"],
        cxpb=ga_config["cxpb"],
        mutpb=ga_config["mutpb"],
    )

    return {
        "descripcion": "Backtesting en tiempo real: GA ejecutado sobre cada instancia de test",
        "metodologia": methodology,
        "datos": {
            "n_instancias_test": n_rows,
            "pct_factibles": round(pct_factibles, 1),
            "tiempo_total_s": round(total_s, 1),
            "tiempo_medio_ms": round(mean_ms, 1),
        },
        "kpi_energia": {
            "E_consumo_hist_medio_kW": round(hist_e_mean, 2),
            "E_consumo_ia_medio_kW": round(ia_e_mean, 2),
            "ahorro_medio_kW": round(ahorro_medio, 2),
            "ahorro_medio_pct": round(ahorro_pct, 2),
            "ahorro_total_kW": round(ahorro_total, 1),
        },
        "kpi_eficiencia_especifica": {
            "E_F_hist_medio": round(hist_ef, 6),
            "E_F_ia_medio": round(ia_ef, 6),
            "mejora_absoluta": round(mejora_abs, 6),
            "mejora_pct": round(mejora_pct, 2),
        },
        "kpi_produccion": {
            "F_flow_hist_medio_Lh": round(hist_flow_mean, 1),
            "F_flow_ia_medio_Lh": round(ia_flow_mean, 1),
            "delta_F_flow_medio": round(delta_flow, 1),
            "delta_F_flow_pct": round(delta_flow_pct, 2),
        },
        "kpi_seguridad": {
            "cumplimiento_hist_pct": round(compliance_hist, 1),
            "cumplimiento_ia_pct": round(compliance_ia, 1),
            "T_out_ia_min": round(float(ia_tout.min()), 2) if n_rows else 0.0,
            "T_out_ia_medio": round(float(ia_tout.mean()), 2) if n_rows else 0.0,
        },
        "kpi_T_servicio": {
            "T_serv_hist_medio": round(hist_tserv, 2),
            "T_serv_ia_medio": round(ia_tserv, 2),
            "delta_T_servicio": round(delta_tserv, 2),
        },
    }


# ============================================================================
# 4. OPTIMIZE
# ============================================================================

def optimize(
    input_path: str = None,
    output_path: str = None,
    pop_size: int = None,
    n_gen: int = None,
    cxpb: float = None,
    mutpb: float = None,
    seed: int = 1,
) -> None:
    """
    Run single-objective GA optimization for each scenario row in an input CSV.

    Parameters
    ----------
    input_path : str, optional
        CSV containing at least: T_in_leche, Delta_P, t_ciclo.
        Default: ``data/splits/test.csv``.
    output_path : str, optional
        Destination CSV path.
        Default: ``data/predictions/evaluation_rt_hist_vs_ia.csv``.
    pop_size, n_gen, cxpb, mutpb : optional
        Optional overrides for GA configuration.
    seed : int
        Base seed used to make each scenario deterministic (seed + row_index).

    Outputs
    -------
    CSV with recommended setpoints and predicted outcomes per scenario.
    If historical columns are present, includes comparison KPIs.
    """
    ensure_dirs()
    set_seed(seed)
    logger.info("=== OPTIMIZE ===")

    from src.predict.inference import predict_with_model
    from src.training.artifacts import load_artifacts
    from src.predict.optimization import (
        setup_ga_toolbox,
        run_ga_single,
    )
    from src.utils.paths import TEST_SPLIT_PATH, EVAL_RT_CSV_PATH

    # Load model artifacts
    model, scaler_X, scaler_y, _ = load_artifacts()
    logger.info("Model and scalers loaded.")

    # Load scenario input
    in_path = input_path or str(TEST_SPLIT_PATH)
    df_input = pd.read_csv(in_path)
    logger.info(f"Input scenarios: {in_path} ({len(df_input)} rows)")

    required_cols = ["T_in_leche", "Delta_P", "t_ciclo"]
    missing = [c for c in required_cols if c not in df_input.columns]
    if missing:
        raise ValueError(
            "Input CSV must contain columns: "
            f"{required_cols}. Missing: {missing}"
        )

    # Build GA toolbox and optionally override config
    toolbox, ga_config = setup_ga_toolbox()
    if pop_size is not None:
        ga_config["pop_size"] = int(pop_size)
    if n_gen is not None:
        ga_config["n_gen"] = int(n_gen)
    if cxpb is not None:
        ga_config["cxpb"] = float(cxpb)
    if mutpb is not None:
        ga_config["mutpb"] = float(mutpb)

    logger.info(
        "GA config: "
        f"pop={ga_config['pop_size']} | "
        f"n_gen={ga_config['n_gen']} | cxpb={ga_config['cxpb']} | mutpb={ga_config['mutpb']}"
    )

    hist_cols = ["F_flow", "T_servicio", "E_consumo", "T_out_leche"]
    has_hist = all(c in df_input.columns for c in hist_cols)

    resultados = []
    n_rows = len(df_input)
    start_time = time.perf_counter()
    for idx, row in df_input.iterrows():
        t_in = float(row["T_in_leche"])
        dp = float(row["Delta_P"])
        t_ciclo = float(row["t_ciclo"])

        _, hof = run_ga_single(
            toolbox,
            T_in_leche=t_in,
            t_ciclo=t_ciclo,
            Delta_P=dp,
            model=model,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            ga_config=ga_config,
            seed=seed + idx,
        )

        best_ind = hof[0]
        F_flow_val, T_servicio_val = best_ind
        E_real, T_out_real = predict_with_model(
            F_flow_val, T_servicio_val, t_in, t_ciclo, dp,
            model=model, scaler_X=scaler_X, scaler_y=scaler_y,
        )
        e_real = float(E_real)
        t_out_real = float(T_out_real)
        f_flow_real = float(F_flow_val)
        specific_consumption = e_real / max(f_flow_real, 1.0)
        feasible = t_out_real >= 72.3

        rec = {
            "T_in_leche": t_in,
            "Delta_P": dp,
            "t_ciclo": t_ciclo,
            "IA_F_flow": round(f_flow_real, 2),
            "IA_T_servicio": round(float(T_servicio_val), 2),
            "IA_E_consumo": round(e_real, 4),
            "IA_T_out": round(t_out_real, 2),
            "IA_consumo_especifico": round(specific_consumption, 6),
            "IA_factible": feasible,
            "fitness_final": round(float(best_ind.fitness.values[0]), 6),
        }

        if has_hist:
            rec.update({
                "HIST_F_flow": float(row["F_flow"]),
                "HIST_T_servicio": float(row["T_servicio"]),
                "HIST_E_consumo": float(row["E_consumo"]),
                "HIST_T_out": float(row["T_out_leche"]),
            })

        resultados.append(rec)

        if (idx + 1) % 500 == 0:
            logger.info(f"  Optimized {idx + 1}/{n_rows} scenarios...")

    elapsed_s = time.perf_counter() - start_time
    df_opt = pd.DataFrame(resultados)

    if has_hist and not df_opt.empty:
        hist_flow = df_opt["HIST_F_flow"].replace(0, np.nan)
        ia_flow = df_opt["IA_F_flow"].replace(0, np.nan)
        df_opt["Ahorro_kW"] = df_opt["HIST_E_consumo"] - df_opt["IA_E_consumo"]
        df_opt["Ahorro_pct"] = (df_opt["Ahorro_kW"] / df_opt["HIST_E_consumo"]) * 100.0
        df_opt["HIST_Eficiencia"] = df_opt["HIST_E_consumo"] / hist_flow
        df_opt["IA_Eficiencia"] = df_opt["IA_E_consumo"] / ia_flow
        df_opt["Mejora_Eficiencia"] = df_opt["HIST_Eficiencia"] - df_opt["IA_Eficiencia"]
        df_opt["Delta_F_flow"] = df_opt["IA_F_flow"] - df_opt["HIST_F_flow"]
        df_opt["Delta_T_servicio"] = df_opt["IA_T_servicio"] - df_opt["HIST_T_servicio"]

    out_path = output_path or str(EVAL_RT_CSV_PATH)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df_opt.to_csv(out_path, index=False)

    feasible_pct = float(df_opt["IA_factible"].mean() * 100.0) if not df_opt.empty else 0.0
    logger.info(
        f"Optimization complete: {len(df_opt)} scenarios | "
        f"feasible={feasible_pct:.1f}%"
    )
    logger.info(f"Optimization results saved to {out_path}")

    if has_hist and not df_opt.empty:
        report = _build_backtesting_report(df_opt, elapsed_s, ga_config)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        with open(EVAL_RT_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Backtesting report saved to {EVAL_RT_REPORT_PATH}")
    else:
        logger.info("Backtesting report not generated (missing historical columns).")


# ============================================================================
# 5. GET_STATS
# ============================================================================

def get_stats() -> None:
    """
    Compute and display dataset statistics and baseline KPIs.

    Outputs:
        - Column info printed to stdout.
        - Baseline KPIs saved to ``models/metrics/baseline_metrics.json``.
    """
    ensure_dirs()
    logger.info("=== GET STATS ===")

    # Column info
    from src.get_stats.column_info import print_column_info

    logger.info("Dataset column inventory:")
    print_column_info()

    # Baseline KPIs
    from src.data_processing.preprocessing import load_processed_data, filter_production
    from src.get_stats.baseline import compute_baseline_kpis

    logger.info("\nComputing baseline KPIs (historical PID operation)...")
    df = load_processed_data()
    df_prod = filter_production(df)
    kpis = compute_baseline_kpis(df_prod)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_METRICS_PATH, "w") as f:
        json.dump(kpis, f, indent=4)

    for k, v in kpis.items():
        logger.info(f"  {k}: {v}")

    logger.info(f"Baseline KPIs saved to {BASELINE_METRICS_PATH}")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Command-line interface for DATAGIA pipeline."""
    parser = argparse.ArgumentParser(
        description="DATAGIA — Energy optimization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.main data_processing\n"
            "  python -m src.main train\n"
            "  python -m src.main predict --input data/splits/test.csv\n"
            "  python -m src.main optimize --input data/splits/test.csv\n"
            "  python -m src.main get_stats\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage to run")

    # data_processing
    subparsers.add_parser("data_processing", help="Generate raw data, filter and split")

    # train
    subparsers.add_parser("train", help="Train the MLP model")

    # predict
    p_pred = subparsers.add_parser("predict", help="Run inference on a dataset")
    p_pred.add_argument("--input", type=str, default=None, help="Input CSV path")
    p_pred.add_argument("--output", type=str, default=None, help="Output CSV path")

    # optimize
    p_opt = subparsers.add_parser(
        "optimize",
        help="Run GA optimization per scenario from an input CSV",
    )
    p_opt.add_argument("--input", type=str, default=None, help="Input CSV path")
    p_opt.add_argument("--output", type=str, default=None, help="Output CSV path")
    p_opt.add_argument("--pop-size", type=int, default=None, help="GA population size")
    p_opt.add_argument("--n-gen", type=int, default=None, help="GA generations")
    p_opt.add_argument("--cxpb", type=float, default=None, help="GA crossover probability")
    p_opt.add_argument("--mutpb", type=float, default=None, help="GA mutation probability")
    p_opt.add_argument("--seed", type=int, default=1, help="Base random seed")

    # get_stats
    subparsers.add_parser("get_stats", help="Compute dataset stats and baseline KPIs")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "data_processing":
        data_processing()
    elif args.command == "train":
        train()
    elif args.command == "predict":
        predict(input_path=args.input, output_path=args.output)
    elif args.command == "optimize":
        optimize(
            input_path=args.input,
            output_path=args.output,
            pop_size=args.pop_size,
            n_gen=args.n_gen,
            cxpb=args.cxpb,
            mutpb=args.mutpb,
            seed=args.seed,
        )
    elif args.command == "get_stats":
        get_stats()


if __name__ == "__main__":
    main()
