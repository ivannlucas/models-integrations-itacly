# Datos de prueba — ml47-dairy-dnsl-pasteurization-fault-detection

Extraídos de la entrega del equipo de IA (`inbox/a47/codigo/data/`, `golden_cases` del manifest), sin
valores inventados. Regenerar: `.venv/bin/python outputs/a47/datos_prueba/generar_datos_prueba.py "$PWD"`.

Datos del banco hidráulico UCI (laboratorio, no planta real). **Todos los ficheros están ya en dominio
planta (TS1 ≈ 65 °C):** se usan con el plugin en su modo por defecto `APPLY_DIGITAL_TWIN=false`. Si el
servidor arranca con `APPLY_DIGITAL_TWIN=true`, la temperatura se desplazaría dos veces.

| Fichero | Endpoint | Contenido |
|---|---|---|
| `predict_inline_cycle_NNN.json` | `POST /models/ml47-dairy-dnsl-pasteurization-fault-detection/predict` | 15 golden cases: 1 ciclo (601 lecturas a 10 Hz de los 7 sensores) cada uno |
| `predict_batch_golden.csv` | `POST .../predict` con `{"mode": "batch", "data_path": "<ruta>"}` | Los 15 ciclos en un CSV (`Cycle_ID`, `Time_Segundos`, 7 sensores) |
| `expected_predictions.csv` | — | Estado esperado por componente (0 sano / 1 warning / 2 crítico) + etiqueta real |
| `train_100ciclos.csv` | `POST .../train` con `{"data_path": "<ruta>", "mlflow_run_id": "<run>"}` | 100 ciclos del split de train con las 4 etiquetas `Target_*` |

El modelo usa 600 pasos por ciclo; los ciclos traen 601 y el plugin avisa de que descarta el último (esperado).

## Resultados comprobados en local (2026-11-04)

- Inline y batch: 15/15 ciclos con los 4 componentes idénticos al manifest. `cycle_211` incluye el
  error real del modelo documentado (Acumulador predice WARNING con etiqueta SANO).
- Train (`mode` por defecto `fine_tune`): ~4 s, 70/15/15 ciclos, exact_match 1.0 en los 15 de test.
  `mode: "full"` reentrena la CNN desde cero y con 100 ciclos da un modelo peor que el servido.
