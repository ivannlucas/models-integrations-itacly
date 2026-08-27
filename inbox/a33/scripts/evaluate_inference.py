"""CLI script to evaluate constrained inference against baseline strategy."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.utils.utils import configure_logging, get_project_root, load_dataset


class EvaluationPathsConfig(BaseModel):
    """Validated paths for evaluation execution.

    Attributes:
        project_root: Absolute path to project root.
        input_path: Relative path to inference CSV file.
        report_path: Relative path to output JSON report.
        causal_params_path: Relative path to causal parameter JSON file.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    project_root: Path
    input_path: str = Field(min_length=1)
    report_path: str = Field(min_length=1)
    causal_params_path: str = Field(min_length=1)


class EvaluationResult(BaseModel):
    """Serialized evaluation output schema.

    Attributes:
        rows: Number of rows evaluated.
        distribution_original_pct_normalized: Baseline strategy distribution (%).
        distribution_ai_pct_normalized: AI-assigned strategy distribution (%).
        baseline_total_emissions_kg: Sum of baseline emissions under the same simulator used for AI.
        ai_estimated_total_emissions_kg: Sum of counterfactual AI emissions.
        total_emissions_reduction_pct: Estimated total reduction percentage.
        baseline_mean_emissions_kg: Mean baseline emissions under the same simulator used for AI.
        ai_estimated_mean_emissions_kg: Mean AI counterfactual emissions per row.
        mean_emissions_reduction_pct: Estimated mean reduction percentage.
        baseline_observed_total_emissions_kg: Optional observed baseline sum from dataset column.
        baseline_observed_mean_emissions_kg: Optional observed baseline mean from dataset column.
        baseline_simulation_alignment_mae_kg: Absolute error between observed and simulated baseline.
        reduction_vs_observed_baseline_pct: Secondary reduction against observed baseline (diagnostic).
        stochastic_runs: Number of Monte Carlo simulations executed.
        stochastic_random_state: Random seed used for reproducible Monte Carlo.
        stochastic_total_reduction_pct_mean: Mean total reduction (%) across Monte Carlo runs.
        stochastic_total_reduction_pct_std: Standard deviation of total reduction (%).
        stochastic_total_reduction_pct_p05: 5th percentile of total reduction (%).
        stochastic_total_reduction_pct_p50: Median total reduction (%).
        stochastic_total_reduction_pct_p95: 95th percentile of total reduction (%).
    """

    rows: int
    # True when the input carries a baseline strategy column (`reuse_strategy`);
    # reduction-vs-baseline metrics are only computed when it is present.
    has_baseline_strategy: bool = True
    # True when the input carries observed baseline emissions (`co2_emissions_kg`).
    has_observed_baseline: bool = True
    notes: list[str] = Field(default_factory=list)
    distribution_original_pct_normalized: dict[str, float] = Field(default_factory=dict)
    distribution_ai_pct_normalized: dict[str, float]
    ai_estimated_total_emissions_kg: float
    ai_estimated_mean_emissions_kg: float
    baseline_total_emissions_kg: float | None = None
    total_emissions_reduction_pct: float | None = None
    baseline_mean_emissions_kg: float | None = None
    mean_emissions_reduction_pct: float | None = None
    baseline_observed_total_emissions_kg: float | None = None
    baseline_observed_mean_emissions_kg: float | None = None
    baseline_simulation_alignment_mae_kg: float | None = None
    reduction_vs_observed_baseline_pct: float | None = None
    stochastic_runs: int | None = None
    stochastic_random_state: int | None = None
    stochastic_total_reduction_pct_mean: float | None = None
    stochastic_total_reduction_pct_std: float | None = None
    stochastic_total_reduction_pct_p05: float | None = None
    stochastic_total_reduction_pct_p50: float | None = None
    stochastic_total_reduction_pct_p95: float | None = None


DEFAULT_STRATEGY_TEMPERATURE_MEAN_C: dict[str, float] = {
    "animal_feed": 60.0,
    "composting": 60.0,
    "biochar": 450.0,
    "biomass_combustion": 900.0,
}

DEFAULT_STRATEGY_TEMPERATURE_STD_C: dict[str, float] = {
    "animal_feed": 10.0,
    "composting": 8.0,
    "biochar": 100.0,
    "biomass_combustion": 150.0,
}


def _normalize_text(value: str) -> str:
    """Normalize text to ASCII lowercase for robust comparisons.

    Args:
        value: Raw input text.

    Returns:
        str: Normalized comparable text.
    """

    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return text.strip().lower()


def _canonical_strategy(raw_strategy: str) -> str:
    """Map strategy labels to canonical identifiers.

    Args:
        raw_strategy: Strategy label possibly containing accents.

    Returns:
        str: Canonical strategy identifier.

    Raises:
        ValueError: If label is unknown.
    """

    strategy_map = {
        "combustion biomasa": "biomass_combustion",
        "biomass combustion": "biomass_combustion",
        "alimentacion animal": "animal_feed",
        "animal feed": "animal_feed",
        "compostaje": "composting",
        "composting": "composting",
        "biochar": "biochar",
    }
    normalized = _normalize_text(raw_strategy)
    if normalized not in strategy_map:
        raise ValueError(f"Unknown strategy label: {raw_strategy}")
    return strategy_map[normalized]


def _load_temperature_distributions(
    causal_params_abs: Path,
    strict: bool,
    logger,
) -> tuple[dict[str, float], dict[str, float]]:
    """Load strategy-level temperature mean/std from causal parameter JSON.

    Args:
        causal_params_abs: Absolute path to data generation parameter file.

    Returns:
        tuple[dict[str, float], dict[str, float]]: Canonical mean and std maps.
    """

    mean_by_strategy = dict(DEFAULT_STRATEGY_TEMPERATURE_MEAN_C)
    std_by_strategy = dict(DEFAULT_STRATEGY_TEMPERATURE_STD_C)

    if not causal_params_abs.exists():
        message = (
            "Causal parameter file not found for evaluation temperature distributions: "
            f"{causal_params_abs}"
        )
        if strict:
            raise FileNotFoundError(message)
        logger.warning("%s. Falling back to default strategy temperature distributions.", message)
        return mean_by_strategy, std_by_strategy

    try:
        payload = json.loads(causal_params_abs.read_text(encoding="utf-8"))
    except Exception as exc:
        message = (
            "Unable to parse causal parameter file for evaluation temperature distributions: "
            f"{causal_params_abs}"
        )
        if strict:
            raise ValueError(message) from exc
        logger.warning("%s. Falling back to default strategy temperature distributions.", message)
        return mean_by_strategy, std_by_strategy

    temperature_by_strategy = payload.get("temperature_by_strategy")
    if not isinstance(temperature_by_strategy, dict):
        message = (
            "Invalid causal parameter schema: key 'temperature_by_strategy' is missing or not an object."
        )
        if strict:
            raise ValueError(message)
        logger.warning("%s Falling back to default strategy temperature distributions.", message)
        return mean_by_strategy, std_by_strategy

    loaded_mean: dict[str, float] = {}
    loaded_std: dict[str, float] = {}
    for strategy_label, distribution in temperature_by_strategy.items():
        if not isinstance(distribution, dict):
            if strict:
                raise ValueError(
                    "Invalid temperature distribution entry. "
                    f"Strategy='{strategy_label}' must map to an object with mean/std."
                )
            continue
        if "mean" not in distribution or "std" not in distribution:
            if strict:
                raise ValueError(
                    "Invalid temperature distribution entry. "
                    f"Strategy='{strategy_label}' must include mean and std."
                )
            continue
        try:
            canonical_key = _canonical_strategy(str(strategy_label))
            mean_value = float(distribution["mean"])
            std_value = float(distribution["std"])
        except Exception as exc:
            if strict:
                raise ValueError(
                    "Invalid strategy label or numeric values in temperature distribution. "
                    f"Strategy='{strategy_label}'."
                ) from exc
            continue

        if std_value <= 0:
            if strict:
                raise ValueError(
                    "Invalid temperature distribution. "
                    f"Strategy='{strategy_label}' has non-positive std={std_value}."
                )
            continue

        loaded_mean[canonical_key] = mean_value
        loaded_std[canonical_key] = std_value

    required_strategies = set(DEFAULT_STRATEGY_TEMPERATURE_MEAN_C.keys())
    loaded_strategies = set(loaded_mean.keys())
    missing_strategies = sorted(required_strategies.difference(loaded_strategies))

    if missing_strategies and strict:
        raise ValueError(
            "Causal parameter file is incomplete. Missing strategy temperature distributions for "
            f"{missing_strategies}."
        )

    if missing_strategies:
        logger.warning(
            "Causal parameter file missing strategy distributions for %s. "
            "Using defaults for missing strategies.",
            missing_strategies,
        )

    if loaded_mean:
        mean_by_strategy.update(loaded_mean)
    if loaded_std:
        std_by_strategy.update(loaded_std)

    logger.info(
        "Loaded strategy temperature distributions from %s (strict=%s).",
        causal_params_abs,
        strict,
    )

    return mean_by_strategy, std_by_strategy


def _calculate_emissions(
    generated_volume_tons: float,
    moisture_pct: float,
    canonical_strategy: str,
    subproduct_type: str,
    strategy_temperature_mean_c: dict[str, float],
) -> float:
    """Calculate emissions using strategy-derived thermodynamic conditions.

    Args:
        generated_volume_tons: Batch volume.
        moisture_pct: Moisture percentage.
        canonical_strategy: Canonical strategy label.
        subproduct_type: Residue type label.
        strategy_temperature_mean_c: Strategy-level temperature means.

    Returns:
        float: Estimated CO2 emissions.
    """

    temperature_c = strategy_temperature_mean_c.get(canonical_strategy, 180.0)
    emission_base = (temperature_c * 0.5) * generated_volume_tons

    if canonical_strategy == "biomass_combustion":
        moisture_penalty = (moisture_pct**1.5) * 2
        return emission_base + moisture_penalty - 50
    if canonical_strategy == "animal_feed":
        if subproduct_type in {"Husk", "Straw", "Silo dust"} or moisture_pct > 18:
            return emission_base * 1.8
        return emission_base * 0.4
    if canonical_strategy == "biochar":
        if moisture_pct < 10:
            return emission_base * 0.2
        return emission_base * 1.2
    if canonical_strategy == "composting":
        return emission_base * 0.8 + (generated_volume_tons * 5)

    return emission_base


def _calculate_emissions_vectorized(
    generated_volume_tons: np.ndarray,
    moisture_pct: np.ndarray,
    temperature_c: np.ndarray,
    canonical_strategy: np.ndarray,
    subproduct_type: np.ndarray,
) -> np.ndarray:
    """Vectorized emissions simulator matching the scalar business rules."""

    emission_base = (temperature_c * 0.5) * generated_volume_tons
    emissions = emission_base.copy()

    is_biomass = canonical_strategy == "biomass_combustion"
    if np.any(is_biomass):
        emissions[is_biomass] = (
            emission_base[is_biomass]
            + ((moisture_pct[is_biomass] ** 1.5) * 2.0)
            - 50.0
        )

    is_animal = canonical_strategy == "animal_feed"
    if np.any(is_animal):
        is_feed_penalty = (
            np.isin(subproduct_type[is_animal], np.array(["Husk", "Straw", "Silo dust"]))
            | (moisture_pct[is_animal] > 18.0)
        )
        animal_values = emission_base[is_animal] * 0.4
        animal_values[is_feed_penalty] = emission_base[is_animal][is_feed_penalty] * 1.8
        emissions[is_animal] = animal_values

    is_biochar = canonical_strategy == "biochar"
    if np.any(is_biochar):
        biochar_values = emission_base[is_biochar] * 1.2
        low_moisture = moisture_pct[is_biochar] < 10.0
        biochar_values[low_moisture] = emission_base[is_biochar][low_moisture] * 0.2
        emissions[is_biochar] = biochar_values

    is_composting = canonical_strategy == "composting"
    if np.any(is_composting):
        emissions[is_composting] = (
            emission_base[is_composting] * 0.8 + (generated_volume_tons[is_composting] * 5.0)
        )

    return emissions


def _evaluate_dataframe(
    df: pd.DataFrame,
    stochastic_runs: int,
    stochastic_random_state: int,
    strategy_temperature_mean_c: dict[str, float],
    strategy_temperature_std_c: dict[str, float],
) -> EvaluationResult:
    """Evaluate baseline vs AI assignments from an inference dataframe.

    Args:
        df: Inference dataframe with baseline and AI strategy columns.
        stochastic_runs: Number of Monte Carlo runs for stochastic temperature analysis.
        stochastic_random_state: Random seed for reproducibility.
        strategy_temperature_mean_c: Strategy-level temperature means.
        strategy_temperature_std_c: Strategy-level temperature standard deviations.

    Returns:
        EvaluationResult: Computed KPI report. When the input lacks baseline
        columns (`reuse_strategy` / `co2_emissions_kg`), reduction-vs-baseline
        metrics are omitted (set to null) and only AI-side metrics are reported,
        so the tool remains usable on arbitrary user CSVs.

    Raises:
        ValueError: If the always-required inference columns are missing.
    """

    # These columns are produced for every inference run (input features + the
    # assigned strategy), so they are always required.
    always_required = {
        "subproduct_type",
        "generated_volume_tons",
        "moisture_pct",
        "ai_assigned_strategy",
    }
    missing_required = always_required.difference(df.columns)
    if missing_required:
        raise ValueError(
            "The evaluation input is missing columns that every inference output "
            f"must contain: {sorted(missing_required)}. Provide the CSV produced by "
            "`scripts.run_inference` (it always includes 'ai_assigned_strategy' plus "
            "the input features 'subproduct_type', 'generated_volume_tons' and "
            "'moisture_pct')."
        )

    # Baseline comparison is OPTIONAL: it requires a pre-existing strategy
    # column. A raw operational CSV (only the 4 inference inputs) has no baseline
    # to compare against, so we degrade gracefully to AI-only metrics.
    has_baseline_strategy = "reuse_strategy" in df.columns
    has_observed_baseline = "co2_emissions_kg" in df.columns
    notes: list[str] = []

    ai_norm = df["ai_assigned_strategy"].map(lambda value: _canonical_strategy(str(value)))
    ai_dist = ai_norm.value_counts(normalize=True).mul(100).round(2).to_dict()

    generated_volume = df["generated_volume_tons"].astype(float).to_numpy()
    moisture = df["moisture_pct"].astype(float).to_numpy()
    subproduct = df["subproduct_type"].astype(str).to_numpy()
    ai_strategy = ai_norm.astype(str).to_numpy()

    ai_temperature_det = np.array(
        [strategy_temperature_mean_c.get(strategy, 180.0) for strategy in ai_strategy],
        dtype=float,
    )
    ai_estimated = _calculate_emissions_vectorized(
        generated_volume_tons=generated_volume,
        moisture_pct=moisture,
        temperature_c=ai_temperature_det,
        canonical_strategy=ai_strategy,
        subproduct_type=subproduct,
    )
    ai_total = float(ai_estimated.sum())
    ai_mean = float(ai_estimated.mean())

    if not has_baseline_strategy:
        # AI-only report: no baseline strategy to compare against.
        notes.append(
            "Input has no 'reuse_strategy' column: reduction-vs-baseline metrics are "
            "omitted. Reported values describe only the AI-assigned distribution and its "
            "estimated emissions. To obtain a reduction percentage, evaluate an inference "
            "output whose source dataset includes a baseline 'reuse_strategy' column "
            "(e.g. the held-out test split)."
        )
        if not has_observed_baseline:
            notes.append(
                "Input has no 'co2_emissions_kg' column: observed-baseline diagnostics are "
                "also omitted."
            )
        return EvaluationResult(
            rows=int(len(df)),
            has_baseline_strategy=False,
            has_observed_baseline=has_observed_baseline,
            notes=notes,
            distribution_original_pct_normalized={},
            distribution_ai_pct_normalized={k: float(v) for k, v in ai_dist.items()},
            ai_estimated_total_emissions_kg=round(ai_total, 2),
            ai_estimated_mean_emissions_kg=round(ai_mean, 4),
        )

    baseline_norm = df["reuse_strategy"].map(lambda value: _canonical_strategy(str(value)))
    baseline_dist = baseline_norm.value_counts(normalize=True).mul(100).round(2).to_dict()
    baseline_strategy = baseline_norm.astype(str).to_numpy()

    baseline_temperature_det = np.array(
        [strategy_temperature_mean_c.get(strategy, 180.0) for strategy in baseline_strategy],
        dtype=float,
    )
    baseline_estimated = _calculate_emissions_vectorized(
        generated_volume_tons=generated_volume,
        moisture_pct=moisture,
        temperature_c=baseline_temperature_det,
        canonical_strategy=baseline_strategy,
        subproduct_type=subproduct,
    )

    baseline_total = float(baseline_estimated.sum())
    baseline_mean = float(baseline_estimated.mean())

    if has_observed_baseline:
        observed_baseline = df["co2_emissions_kg"].clip(lower=0.0)
        baseline_observed_total = float(observed_baseline.sum())
        baseline_observed_mean = float(observed_baseline.mean())
        baseline_alignment_mae = float((observed_baseline - baseline_estimated).abs().mean())
        reduction_vs_observed_baseline_pct = round(
            ((baseline_observed_total - ai_total) / baseline_observed_total) * 100
            if baseline_observed_total
            else 0.0,
            2,
        )
    else:
        baseline_observed_total = None
        baseline_observed_mean = None
        baseline_alignment_mae = None
        reduction_vs_observed_baseline_pct = None
        notes.append(
            "Input has no 'co2_emissions_kg' column: the observed-baseline diagnostic "
            "(reduction_vs_observed_baseline_pct) is omitted; the homogeneous "
            "simulated-vs-simulated reduction is still reported."
        )

    reduction_total_pct = ((baseline_total - ai_total) / baseline_total) * 100 if baseline_total else 0.0
    reduction_mean_pct = ((baseline_mean - ai_mean) / baseline_mean) * 100 if baseline_mean else 0.0

    rng = np.random.default_rng(stochastic_random_state)
    baseline_temp_mean = np.array(
        [strategy_temperature_mean_c.get(strategy, 180.0) for strategy in baseline_strategy],
        dtype=float,
    )
    ai_temp_mean = np.array(
        [strategy_temperature_mean_c.get(strategy, 180.0) for strategy in ai_strategy],
        dtype=float,
    )
    baseline_temp_std = np.array(
        [strategy_temperature_std_c.get(strategy, 0.0) for strategy in baseline_strategy],
        dtype=float,
    )
    ai_temp_std = np.array(
        [strategy_temperature_std_c.get(strategy, 0.0) for strategy in ai_strategy],
        dtype=float,
    )

    stochastic_reductions: list[float] = []
    for _ in range(stochastic_runs):
        # Common random shocks per row improve baseline-vs-AI comparability in each run.
        z = rng.standard_normal(len(df))
        baseline_temp = np.clip(baseline_temp_mean + (baseline_temp_std * z), a_min=0.0, a_max=None)
        ai_temp = np.clip(ai_temp_mean + (ai_temp_std * z), a_min=0.0, a_max=None)

        baseline_stochastic = _calculate_emissions_vectorized(
            generated_volume_tons=generated_volume,
            moisture_pct=moisture,
            temperature_c=baseline_temp,
            canonical_strategy=baseline_strategy,
            subproduct_type=subproduct,
        )
        ai_stochastic = _calculate_emissions_vectorized(
            generated_volume_tons=generated_volume,
            moisture_pct=moisture,
            temperature_c=ai_temp,
            canonical_strategy=ai_strategy,
            subproduct_type=subproduct,
        )

        stochastic_baseline_total = float(np.sum(baseline_stochastic))
        stochastic_ai_total = float(np.sum(ai_stochastic))
        run_reduction = (
            ((stochastic_baseline_total - stochastic_ai_total) / stochastic_baseline_total) * 100
            if stochastic_baseline_total
            else 0.0
        )
        stochastic_reductions.append(run_reduction)

    stochastic_reduction_array = np.array(stochastic_reductions, dtype=float)

    return EvaluationResult(
        rows=int(len(df)),
        has_baseline_strategy=True,
        has_observed_baseline=has_observed_baseline,
        notes=notes,
        distribution_original_pct_normalized={k: float(v) for k, v in baseline_dist.items()},
        distribution_ai_pct_normalized={k: float(v) for k, v in ai_dist.items()},
        baseline_total_emissions_kg=round(baseline_total, 2),
        ai_estimated_total_emissions_kg=round(ai_total, 2),
        total_emissions_reduction_pct=round(reduction_total_pct, 2),
        baseline_mean_emissions_kg=round(baseline_mean, 4),
        ai_estimated_mean_emissions_kg=round(ai_mean, 4),
        mean_emissions_reduction_pct=round(reduction_mean_pct, 2),
        baseline_observed_total_emissions_kg=(
            round(baseline_observed_total, 2) if baseline_observed_total is not None else None
        ),
        baseline_observed_mean_emissions_kg=(
            round(baseline_observed_mean, 4) if baseline_observed_mean is not None else None
        ),
        baseline_simulation_alignment_mae_kg=(
            round(baseline_alignment_mae, 4) if baseline_alignment_mae is not None else None
        ),
        reduction_vs_observed_baseline_pct=reduction_vs_observed_baseline_pct,
        stochastic_runs=int(stochastic_runs),
        stochastic_random_state=int(stochastic_random_state),
        stochastic_total_reduction_pct_mean=round(float(np.mean(stochastic_reduction_array)), 2),
        stochastic_total_reduction_pct_std=round(float(np.std(stochastic_reduction_array)), 4),
        stochastic_total_reduction_pct_p05=round(float(np.quantile(stochastic_reduction_array, 0.05)), 2),
        stochastic_total_reduction_pct_p50=round(float(np.quantile(stochastic_reduction_array, 0.50)), 2),
        stochastic_total_reduction_pct_p95=round(float(np.quantile(stochastic_reduction_array, 0.95)), 2),
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed argument values.
    """

    parser = argparse.ArgumentParser(
        description="Evaluate inference output against baseline emissions in one command.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=get_project_root(),
        help="Project root path.",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/predictions/inference_with_constraints.csv",
        help="Inference CSV path relative to project root.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="models/metrics/inference_evaluation_report.json",
        help="Output report JSON path relative to project root.",
    )
    parser.add_argument(
        "--causal-params-path",
        type=str,
        default="config/data_generation_params.json",
        help="Causal data-generation parameter JSON used to source temperature distributions.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--stochastic-runs",
        type=int,
        default=200,
        help="Monte Carlo runs for stochastic strategy-temperature evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for Monte Carlo reproducibility.",
    )
    parser.add_argument(
        "--allow-causal-params-fallback",
        action="store_true",
        help=(
            "Allow fallback to default temperature distributions when causal parameter file is "
            "missing or invalid. By default, evaluation fails fast."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run evaluation workflow and persist a JSON report."""

    args = parse_args()
    logger = configure_logging(args.log_level)

    try:
        path_config = EvaluationPathsConfig(
            project_root=args.project_root.resolve(),
            input_path=args.input_path,
            report_path=args.report_path,
            causal_params_path=args.causal_params_path,
        )

        input_abs = path_config.project_root / path_config.input_path
        report_abs = path_config.project_root / path_config.report_path
        causal_params_abs = path_config.project_root / path_config.causal_params_path

        strategy_temperature_mean_c, strategy_temperature_std_c = _load_temperature_distributions(
            causal_params_abs=causal_params_abs,
            strict=not bool(args.allow_causal_params_fallback),
            logger=logger,
        )

        df = load_dataset(input_abs, logger)
        result = _evaluate_dataframe(
            df,
            stochastic_runs=max(1, int(args.stochastic_runs)),
            stochastic_random_state=int(args.random_state),
            strategy_temperature_mean_c=strategy_temperature_mean_c,
            strategy_temperature_std_c=strategy_temperature_std_c,
        )

        report_abs.parent.mkdir(parents=True, exist_ok=True)
        report_payload: dict[str, Any] = result.model_dump()
        report_abs.write_text(json.dumps(report_payload, ensure_ascii=True, indent=2), encoding="utf-8")

        logger.info("Evaluation report saved to %s", report_abs)
        for note in result.notes:
            logger.warning(note)
        if result.has_baseline_strategy:
            logger.info("Total emissions reduction (%%): %s", result.total_emissions_reduction_pct)
            logger.info("Mean emissions reduction (%%): %s", result.mean_emissions_reduction_pct)
            logger.info(
                "Stochastic total reduction (%%) mean=%s, p05=%s, p95=%s, runs=%s",
                result.stochastic_total_reduction_pct_mean,
                result.stochastic_total_reduction_pct_p05,
                result.stochastic_total_reduction_pct_p95,
                result.stochastic_runs,
            )
        else:
            logger.info(
                "AI-only report (no baseline column). AI estimated total emissions (kg): %s",
                result.ai_estimated_total_emissions_kg,
            )
    except ValidationError:
        logger.exception("Invalid evaluation configuration.")
        raise
    except Exception:
        logger.exception("Inference evaluation failed.")
        raise


if __name__ == "__main__":
    main()
