# DATAGIA - CO2 Reuse-Strategy Optimization

This repository contains an end-to-end, reproducible pipeline to optimize cereal byproduct reuse strategies under plant capacity constraints, minimizing simulated CO2 emissions.

The **deployed decision engine is an exact optimizer**: each operational block
(capacities reset every `lots_per_day` lots) is solved to proven optimality as a
mixed-integer linear program (MILP). Neuroevolution (NEAT) is retained as a
**learned-policy benchmark** and is bounded above by the exact optimum. See
`scripts/run_baselines.py` and `models/metrics/baseline_comparison.json` for the
full comparison ladder (exact vs greedy vs NEAT vs fitness-optimized linear vs
supervised logistic regression vs random).

## What This Project Does

- Generates synthetic process data from causal assumptions plus CTGAN.
- Applies a deterministic temperature reassignment by strategy after CTGAN sampling.
- Splits the generated dataset into train and test partitions.
- Runs the **exact MILP optimizer** (primary, self-contained: no trained artifact) on the held-out test split under capacity constraints.
- Trains a NEAT policy on the training split only, as a benchmark.
- Produces deterministic and stochastic (Monte Carlo) evaluation metrics against the baseline strategy.
- Uses temperature only inside the emissions simulator, not as a decision input.

## Methodological Scope (Important)

- This repository models a synthetic decision environment, not a historical operational dataset.
- Causal parameters come from configurable assumptions (`config/data_generation_params.json`) and are not a post-hoc fit against observed plant records.
- CTGAN output is post-processed for strategy-temperature coherence, so the final dataset combines generative sampling with deterministic rules.
- Emissions and reduction metrics are simulator-based estimates (`src/training/evolution.py`, `src/predict/inference.py`), not direct measurements from real process sensors.
- Process temperature is treated as a strategy-derived variable for carbon footprint estimation, not as an exogenous ML feature at decision time.
- The baseline strategy is synthetic (generated inside the same environment), so AI-vs-baseline results reflect improvement against an artificial reference.

## Repository Structure

- `config/`: NEAT and pipeline configuration files.
- `data/`: generated datasets, train/test splits, and predictions.
- `models/`: trained artifacts and metrics reports.
- `scripts/`: CLI entrypoints for each pipeline step.
- `src/`: reusable core modules, organized into subpackages.
- `notebooks/`: exploratory and historical notebooks.

Current main subpackages under `src/`:

- `src/data_processing/`: synthetic data generation and preprocessing.
- `src/predict/`: constrained inference logic.
- `src/training/`: NEAT evolution and fitness evaluation.
- `src/utils/`: shared helpers such as logging and loading.

## Complete Repository Structure

Estructura del repositorio versionada en git (código y artefactos de reproducibilidad). No se versionan: `.venv/`, `__pycache__/`, los ficheros de contexto `*.docx` (auditor/entregable/checklist/registro) ni los CSV de prueba manual.

```text
.
|-- config
|   |-- config-feedforward-linear.txt
|   |-- config-feedforward.txt
|   |-- data_generation_params.json
|   `-- pipeline_config.json
|-- data
|   |-- EDA
|   |   |-- co2_boxplot.png
|   |   |-- distribucion_categoricas.png
|   |   |-- distribucion_numericas.png
|   |   `-- matriz_correlacion.png
|   |-- predictions
|   |   |-- capacity_sensitivity
|   |   |   |-- inference_base.csv
|   |   |   |-- inference_cap_animal_feed_100.csv
|   |   |   |-- inference_cap_animal_feed_110.csv
|   |   |   |-- inference_cap_animal_feed_120.csv
|   |   |   |-- inference_cap_animal_feed_80.csv
|   |   |   |-- inference_cap_animal_feed_90.csv
|   |   |   |-- inference_cap_biochar_15.csv
|   |   |   |-- inference_cap_biochar_30.csv
|   |   |   |-- inference_cap_biochar_45.csv
|   |   |   |-- inference_cap_biochar_60.csv
|   |   |   |-- inference_cap_biochar_75.csv
|   |   |   |-- inference_cap_composting_100.csv
|   |   |   |-- inference_cap_composting_120.csv
|   |   |   |-- inference_cap_composting_140.csv
|   |   |   |-- inference_cap_composting_160.csv
|   |   |   |-- inference_cap_composting_180.csv
|   |   |   |-- inference_conservador.csv
|   |   |   `-- inference_expansivo.csv
|   |   |-- tuning_hyperparameters
|   |   |   |-- inference_base_line.csv
|   |   |   |-- inference_deeper_evolution.csv
|   |   |   |-- inference_fast_small_sample.csv
|   |   |   `-- inference_larger_sample.csv
|   |   `-- inference_with_constraints.csv
|   |-- processed
|   |   `-- dataset_optimization_cereal_co2.csv
|   |-- raw
|   |   `-- dataset_optimization_cereal_co2_seed.csv
|   `-- split
|       |-- dataset_optimization_cereal_co2_scaler.joblib
|       |-- dataset_optimization_cereal_co2_scaler_metadata.json
|       |-- dataset_optimization_cereal_co2_test_raw.csv
|       |-- dataset_optimization_cereal_co2_test_scaled.csv
|       |-- dataset_optimization_cereal_co2_train_raw.csv
|       `-- dataset_optimization_cereal_co2_train_scaled.csv
|-- models
|   |-- artifacts
|   |   |-- tuning_hyperparameters
|   |   |   |-- winner_base_line.pkl
|   |   |   |-- winner_deeper_evolution.pkl
|   |   |   |-- winner_fast_small_sample.pkl
|   |   |   `-- winner_larger_sample.pkl
|   |   |-- winner_genome.pkl
|   |   `-- winner_genome.pkl.metadata.json
|   `-- metrics
|       |-- capacity_sensitivity
|       |   |-- inference_base_report.json
|       |   |-- inference_cap_animal_feed_100_report.json
|       |   |-- inference_cap_animal_feed_110_report.json
|       |   |-- inference_cap_animal_feed_120_report.json
|       |   |-- inference_cap_animal_feed_80_report.json
|       |   |-- inference_cap_animal_feed_90_report.json
|       |   |-- inference_cap_biochar_15_report.json
|       |   |-- inference_cap_biochar_30_report.json
|       |   |-- inference_cap_biochar_45_report.json
|       |   |-- inference_cap_biochar_60_report.json
|       |   |-- inference_cap_biochar_75_report.json
|       |   |-- inference_cap_composting_100_report.json
|       |   |-- inference_cap_composting_120_report.json
|       |   |-- inference_cap_composting_140_report.json
|       |   |-- inference_cap_composting_160_report.json
|       |   |-- inference_cap_composting_180_report.json
|       |   |-- inference_conservador_report.json
|       |   `-- inference_expansivo_report.json
|       |-- tuning_hyperparameters
|       |   |-- inference_base_line.json
|       |   |-- inference_deeper_evolution.json
|       |   |-- inference_fast_small_sample.json
|       |   `-- inference_larger_sample.json
|       |-- baseline_comparison.json
|       |-- inference_evaluation_report.json
|       `-- training_fitness_history.json
|-- notebooks
|   |-- EDA
|   |   `-- EDA.ipynb
|   |-- training
|   |   `-- 2_neat.ipynb
|   |-- tuning_hyperparameters
|   |   |-- capacity_sensitivity.ipynb
|   |   `-- neuroevolution_hyperparameter_search.ipynb
|   |-- 0_data_gen.ipynb
|   |-- 1_preprocessing.ipynb
|   `-- 3_inference.ipynb
|-- scripts
|   |-- evaluate_inference.py
|   |-- run_baselines.py
|   |-- run_data_generation.py
|   |-- run_inference.py
|   |-- run_optimization.py
|   |-- run_pipeline.py
|   `-- run_preprocessing.py
|-- src
|   |-- data_processing
|   |   |-- data_generation.py
|   |   `-- preprocessing.py
|   |-- predict
|   |   |-- exact_optimizer.py
|   |   `-- inference.py
|   |-- training
|   |   `-- evolution.py
|   |-- utils
|   |   |-- artifacts.py
|   |   `-- utils.py
|   |-- __init__.py
|   |-- config.py
|   `-- model.py
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Environment

- Python `3.10+`
- Virtual environment at `.venv`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Single-Command Pipeline

Run the full workflow:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py --project-root . --config-path config/pipeline_config.json --log-level INFO
```

Main outputs:

- `data/raw/dataset_optimization_cereal_co2_seed.csv`
- `data/processed/dataset_optimization_cereal_co2.csv`
- `data/split/*`
- `models/artifacts/winner_genome.pkl`
- `data/predictions/inference_with_constraints.csv`
- `models/metrics/inference_evaluation_report.json`

Pipeline flow:

1. Generate synthetic data.
2. Split into train and test sets.
3. Train NEAT on the train split.
4. Run constrained inference on the test split.
5. Evaluate AI vs baseline on the test output.

Evaluation behavior in this command:

- The pipeline evaluation stage runs Monte Carlo by default with `stochastic_runs=200` and `random_state=42` unless overridden in `config/pipeline_config.json` under `evaluation`.
- Temperature distributions for Monte Carlo are loaded from `config/data_generation_params.json`.
- Loading is fail-fast by default. If the causal parameter file is missing/invalid/incomplete, evaluation stops with an explicit error unless `evaluation.allow_causal_params_fallback=true` is configured.

Recommended `evaluation` block in `config/pipeline_config.json`:

```json
"evaluation": {
	"input_path": "data/predictions/inference_with_constraints.csv",
	"report_path": "models/metrics/inference_evaluation_report.json",
	"stochastic_runs": 200,
	"random_state": 42,
	"allow_causal_params_fallback": false
}
```

## Execution Profiles

- `scripts.run_pipeline` + `config/pipeline_config.json` (recommended for reproducibility)
	- Training `sample_size`: `500`
	- Training `generations`: `50`
- `scripts.run_optimization` standalone defaults are aligned to the same values (`500` / `50`).
- The NEAT search runs the full number of generations (`no_fitness_termination = True`) so the
  final architecture reflects the problem structure, not premature termination.

You can still override either value with CLI flags when running sensitivity studies.

## Step-by-Step Commands

1. Generate data

```powershell
.\.venv\Scripts\python.exe scripts/run_data_generation.py --project-root . --causal-params-path config/data_generation_params.json --output-path data/processed/dataset_optimization_cereal_co2.csv --seed-output-path data/raw/dataset_optimization_cereal_co2_seed.csv --n-real-samples 1500 --ctgan-epochs 100 --n-synthetic-samples 50000 --random-state 42 --log-level INFO
```

2. Preprocess / split

```powershell
.\.venv\Scripts\python.exe scripts/run_preprocessing.py --project-root . --input-path data/processed/dataset_optimization_cereal_co2.csv --output-dir data/split --split-prefix dataset_optimization_cereal_co2 --test-size 0.2 --random-state 42 --log-level INFO
```

3. Train optimization model

```powershell
.\.venv\Scripts\python.exe scripts/run_optimization.py --project-root . --dataset-path data/split/dataset_optimization_cereal_co2_train_scaled.csv --neat-config-path config/config-feedforward.txt --sample-size 500 --generations 50 --winner-output models/artifacts/winner_genome.pkl --log-level INFO
```

Optionally, run the benchmark ladder (deployed exact optimizer vs NEAT vs
fitness-optimized linear vs supervised/greedy/random baselines):

```powershell
.\.venv\Scripts\python.exe scripts/run_baselines.py --project-root . --sample-size 500 --generations 50 --random-state 42 --lots-per-day 15 --report-path models/metrics/baseline_comparison.json --log-level INFO
```



4. Run constrained inference on test

The default `--selector exact` runs the deployed MILP optimizer (self-contained;
needs no trained artifact). Use `--selector neat` to run the neuroevolution
benchmark instead.

```powershell
.\.venv\Scripts\python.exe scripts/run_inference.py --project-root . --dataset-path data/split/dataset_optimization_cereal_co2_test_raw.csv --output-path data/predictions/inference_with_constraints.csv --selector exact --lots-per-day 15 --cap-animal-feed 90 --cap-composting 140 --cap-biochar 45 --cap-biomass-combustion 10000 --log-level INFO
```

The output CSV always contains the input columns plus `ai_assigned_strategy`,
`ai_assignment_source` and `ai_is_fallback`. It runs on **any** CSV with the four
required columns (`generated_volume_tons`, `moisture_pct`, `subproduct_type`,
`season`), regardless of filename.

5. Evaluate impact (deterministic + Monte Carlo)

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_inference.py --project-root . --input-path data/predictions/inference_with_constraints.csv --report-path models/metrics/inference_evaluation_report.json --causal-params-path config/data_generation_params.json --stochastic-runs 200 --random-state 42 --log-level INFO
```

Reduction-vs-baseline metrics require a baseline strategy column (`reuse_strategy`)
and, for the observed-baseline diagnostic, `co2_emissions_kg`. When these are
absent (e.g. a raw operational CSV with only the four inference inputs), the
report **degrades gracefully** to AI-only metrics (assigned distribution and
estimated emissions) with an explanatory `notes` field, instead of failing. The
report's `has_baseline_strategy` / `has_observed_baseline` flags state which
metrics were computed.

Optional fallback mode (only when intentionally needed):

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_inference.py --project-root . --input-path data/predictions/inference_with_constraints.csv --report-path models/metrics/inference_evaluation_report.json --causal-params-path config/data_generation_params.json --stochastic-runs 200 --random-state 42 --allow-causal-params-fallback --log-level INFO
```

The default mode is strict fail-fast and is recommended for auditability.

## Reproducibility

- The pipeline supports a single top-level seed in `config/pipeline_config.json` (`random_state`).
- Generation, preprocessing, training, and inference inherit that seed unless explicitly overridden.
- Synthetic post-processing restores temperature distributions by selected strategy after CTGAN sampling.
- The train split is used for NEAT optimization, and the test split is used for inference and evaluation.
- Training writes a sidecar metadata file next to the winner model (`*.pkl.metadata.json`) and inference validates this metadata before execution.
- Preprocessing writes both the binary scaler and a human-readable metadata JSON with physical min/max ranges.
- Full pipeline execution writes `models/metrics/pipeline_run_manifest.json` with config snapshot and artifact checksums.
- Evaluation writes deterministic and Monte Carlo metrics to `models/metrics/inference_evaluation_report.json`, including `stochastic_runs`, `stochastic_random_state`, and p05/p50/p95 reduction percentiles.
- Evaluation loads temperature distributions from `config/data_generation_params.json` and defaults to strict fail-fast behavior.
- The base validated inference regime uses capacities of 90 t for Animal feed, 140 t for Composting, 45 t for Biochar, and 10000 t for Biomass combustion.

## Artifact Metadata Validation (artifacts.py)

The module `src/utils/artifacts.py` centralizes metadata management for model artifacts to avoid silent incompatibilities between training and inference.

Training phase (`scripts/run_optimization.py`):

- Builds a metadata payload with dataset path, NEAT config path, sample size, generations, seed, strategy list, input columns, and policy design flags.
- Writes a sidecar JSON next to the model artifact (`winner_genome.pkl.metadata.json`).

Inference phase (`scripts/run_inference.py --selector neat` via `src/predict/inference.py`):

- Validates that the sidecar metadata exists before loading the model.
- Validates required keys and metadata schema version (1 or 2).
- **Hard checks** (block execution): the loaded genome's SHA-256 must match the
  metadata, the input-feature order and strategy order must match, and the policy
  must not use process temperature as an input.
- **Soft checks** (warn only, for portability): the training-dataset presence/hash
  and the NEAT-config hash are provenance signals. A missing training file or a
  cosmetic config edit no longer blocks inference, because inference does not read
  the training dataset (physical ranges are embedded in the metadata) and paths are
  stored **relative to the project root**.

Practical outcome:

- Inference is portable across machines and independent of the training host's paths.
- The deployed `exact` selector needs no artifact at all, so it is unaffected by metadata.
- Incompatible artifacts (wrong genome, wrong feature/strategy layout, temperature
  leakage) are still rejected early with an explicit error.

### Self-contained inference (any input CSV name)

Schema v2 metadata embeds the training scaler's physical min/max ranges
(`physical_ranges`). Inference reads them from the model metadata, so it accepts
**any** input CSV that contains the four required columns
(`generated_volume_tons`, `moisture_pct`, `subproduct_type`, `season`),
regardless of the filename. The legacy filename-based scaler resolution remains
only as a fallback; an explicit `--scaler-path` can also be supplied for legacy
(schema v1) artifacts that do not embed ranges.

### Baselines and training curves

- `scripts/run_baselines.py` builds the full comparison ladder under the same
  capacity layer and emissions simulator, writing `models/metrics/baseline_comparison.json`:
  - `exact_optimum` — per-block MILP, the deployed model (provably optimal).
  - `oracle` — greedy lowest-emission feasible per lot (a lower bound on the optimum).
  - `neat` — the evolved neuroevolution policy (benchmark).
  - `linear_fitness` — a fixed linear policy trained with the SAME fitness
    function as NEAT (`config/config-feedforward-linear.txt`); isolates the
    contribution of topology augmentation.
  - `logistic_reg` — supervised imitation of the greedy optimum (label learner).
  - `random` — uniform random scores (naive lower bound).
- Training writes `models/metrics/training_fitness_history.json` with best/mean/std
  fitness and the best genome's structural size per generation.

## Data Transformation Traceability

The preprocessing step applies normalization and encoding that must be reversed for interpretation:

### Understanding the Scaler Metadata

After preprocessing, the file `scaler_metadata.json` contains explicit information for denormalization:

```json
{
  "scaled_columns": ["generated_volume_tons", "moisture_pct", "process_temperature_c"],
  "non_scaled_columns": ["subproduct_type_*", "season_*"],
  "note": "One-hot encoded categorical columns are NOT scaled. Only scaled_columns are min-max normalized to [0, 1].",
  "ranges": {
    "generated_volume_tons": {
      "data_min": 0.1,
      "data_max": 61.69,
      "scale": 0.01624,
      "denormalization_formula": "value_physical = value_normalized / 0.01624 + 0.1"
    }
  },
  "usage": {
    "inverse": "Use scaler.inverse_transform() or apply denormalization_formula manually to recover physical units."
  }
}
```

### Denormalizing Predictions

To recover physical units from normalized predictions:

```python
import joblib
from src.utils.utils import load_scaler_metadata, denormalize_columns

# Load scaler and metadata
scaler = joblib.load('data/split/dataset_optimization_cereal_co2_scaler.joblib')
metadata = load_scaler_metadata('data/split/dataset_optimization_cereal_co2_scaler_metadata.json')

# Load scaled predictions
import pandas as pd
df_scaled = pd.read_csv('data/predictions/inference_with_constraints.csv')

# Denormalize physical columns
df_physical = denormalize_columns(df_scaled, scaler, metadata['scaled_columns'])
```

The inverse transformation applies the formula:
$$\text{value}_{\text{physical}} = \frac{\text{value}_{\text{normalized}}}{\text{scale}} + \text{data}_{\text{min}}$$

## Notes

- Domain dataset column names remain compatible with historical data files.
- Inference now defaults to the held-out test split: `data/split/dataset_optimization_cereal_co2_test_raw.csv`.
- Notebooks are exploratory/historical; scripts and `config/pipeline_config.json` are the source of truth for reproducible runs.

