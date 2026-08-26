"""Project configuration for CU21."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Garantiza que PROJECT_ROOT esté en sys.path cuando config se importa
# directamente (p.ej. python config/config.py). No afecta a ejecuciones
# con python -m, donde el entorno ya está configurado correctamente.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODEL_ARTIFACTS_DIR: Path = PROJECT_ROOT / "models" / "artifacts"
MODEL_METRICS_DIR: Path = PROJECT_ROOT / "models" / "metrics"
PREDICTIONS_DIR: Path = PROJECT_ROOT / "data" / "predictions"
RANDOM_SEED: int = 42

# Umbrales estratégicos para inferencia (calibrados sobre datos históricos)
PROB_BULL = 0.65
PROB_BEAR = 0.35
RET_BULL = 0.015
RET_BEAR = -0.015
SPREAD_MIN = 0.01
CONFIDENCE_HIGH_UP = 0.60
CONFIDENCE_HIGH_DOWN = 0.40
TIMING_DELTA = 0.01
BENCHMARK_BREAK_DELTA = 0.01

# Desfase administrativo de publicacion oficial del MAPA (meses).
# Fuente: protocolo de difusion IPPA/INDPAG. Revisable si cambia el calendario.
MAPA_ADMIN_LAG: int = 3
