"""Generate ml28 test payloads from the AI team's demo CSVs + manifest golden cases (no invented values)."""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(sys.argv[1])
OUT = ROOT / "outputs/a28/datos_prueba"
OUT.mkdir(parents=True, exist_ok=True)
DEMO = ROOT / "inbox/a28/codigo/a28-cnp-carnico-neuroevolutivas-materia/data/demo"

golden = yaml.safe_load((ROOT / "inbox/a28/manifest.yaml").read_text())["golden_cases"]
expected = []
for case in golden:
    (OUT / f"predict_inline_{case['id']}.json").write_text(
        json.dumps({"mode": "inline", **case["input"]}, indent=2))
    expected.append({"case_id": case["id"], **case["input"], **{f"esperado_{k}": v for k, v in case["expected"].items()}})
pd.DataFrame(expected).to_csv(OUT / "expected_predictions.csv", index=False)

for src in sorted(DEMO.glob("*.csv")):
    shutil.copy(src, OUT / f"predict_batch_{src.name}")
print(len(golden), "inline cases;", len(list(DEMO.glob('*.csv'))), "batch CSVs")
