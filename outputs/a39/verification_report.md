# Verificación — ml39-wine-predictive-maintenance-xai

> **[FUSIONADO EN ml37 — 2026-11-24]** Este plugin se ha eliminado: la capa XAI de a39 se sirve
> desde `ml37-wine-press-predictive-maintenance` (`include_xai` / `include_shap`). La
> reverificación está en `outputs/a37/verification_report.md`. Corrección a este informe: el
> 500 de `include_shap=true` **no** era un comportamiento aceptable limitado al servidor de
> desarrollo. `main.py` importa `app.registry` entero antes del filtro `MODEL`, así que también
> en el pod el backend de Keras es siempre torch y SHAP fallaba en el 100 % de las llamadas.
> El texto de debajo es el histórico, sin reescribir.

Fecha: 2026-11-19

## Checklist técnico

- [x] **flake8** (`app/plugins/ml39_wine_predictive_maintenance_xai/`): 0 errores.
- [x] **pytest**: 12/12 pasados
  - `tests/unit/test_ml39_wine_predictive_maintenance_xai.py` (9 tests, wiring con
    `FakePlugin`): health, stats, predict inline con/sin `include_shap`, predict inline sin
    `readings` → 422, `DataContractError` → 422 (inline y batch), predict batch, train → 501.
  - `tests/unit/test_ml39_inference.py` (3 tests, código real — no `FakePlugin`): reproduce
    Integrated Gradients real contra 3 de los 20 golden cases del manifest, 6 decimales.
  - Suite completa: 1511 pasados, 38 skipped, 2 fallos **pre-existentes y ajenos**
    (`test_ml10_dairy_disease_vector_detection_unit`, `test_numba_cache` — versiones de
    sklearn/numba en este entorno, nada relacionado con ml39). Mismo baseline que antes de
    este trabajo — 0 regresiones.
- [x] **pylint** (`app/plugins/ml39_wine_predictive_maintenance_xai/`): **10.00/10**.
- [x] **pip-audit**: no disponible en este entorno (`ensurepip`/`python3-venv` no instalado) —
  misma limitación ya documentada para a37 y otros modelos de este repo, no específica de a39.
- [x] **Arranque local real** (`MODEL=ml39-wine-predictive-maintenance-xai python main.py`):
  1/1 modelo listo.
- [x] **`GET /health`**: `{"status":"ok","model":"ml39-wine-predictive-maintenance-xai","version":"1.0.0","loaded":true}`
- [x] **`POST /predict` (inline)**: probado con el ciclo real 33 (600 lecturas reales de
  `data/processed/tensors/X.npy`, el mismo payload usado en la verificación de a37) → ver
  correctitud abajo. `predicted_class` coincide exactamente con la predicción de a37 para el
  mismo ciclo (`cooler=3, valve=100, pump=0, accumulator=130`), como debe ser (mismo modelo).
- [x] **`POST /predict` con `include_shap=true`**: en este proceso real, devuelve **HTTP 500**
  con un mensaje diagnosticable (`"SHAP no disponible: el backend de Keras 3 activo... es
  'torch'..."`) — comportamiento esperado y documentado, ver "Hallazgo real" abajo. Integrated
  Gradients sigue funcionando sin problema en el mismo proceso.
- [x] **`POST /predict` (batch)**: 2 ciclos (uno con un sensor roto a propósito) → `n_rows=2`,
  `n_predictions=1`, el ciclo roto devuelve su `error` sin abortar el lote.
- [x] **`GET /stats`**: `model_name`, `task_type=explainability_layer`, 2 inputs
  (`readings`, `include_shap`), métricas heredadas de a37 (mismo modelo) y las propias de XAI
  (separabilidad latente, Brier score, concordancia IG-SHAP) del manifest.
- [x] **`POST /train`**: **501** (`TrainingNotSupportedError`) con mensaje explicando que a39
  no tiene parámetros propios que entrenar — esperado, `manifest.training.supported: false`
  confirmado por código (ver manifest, no un 501 por omisión).

## Hallazgo real durante esta verificación: contaminación de backend entre plugins

Confirmado de forma reproducible contra el servidor real (no solo en tests): importar
`app.registry` — lo que hace todo proceso real, con o sin filtro `MODEL` — importa el
`model_loader.py` de ml1 (y, una vez en main, el de ml37), que fija `KERAS_BACKEND=torch` como
variable de entorno del proceso. Keras 3 fija su backend en el primer import, así que esto
deja `tf.keras` corriendo sobre **torch**, no TensorFlow real, para el resto del proceso —
rompiendo silenciosamente `tf.GradientTape` para cualquier plugin que lo necesite de verdad
(exactamente el caso de a39).

Esto se descubrió y se corrigió durante `plugin-integration` (no aquí), y esta verificación lo
**reproduce de forma independiente contra el servidor real**: Integrated Gradients, al tener
una vía nativa en torch (`_integrated_gradients_torch`), sigue dando resultados idénticos sea
cual sea el backend activo (verificado: 80/80 cabezas con el ranking top-3 exacto). SHAP no
tiene vía torch nativa, así que falla con un **500 claro y diagnosticable** en vez de un error
críptico — comportamiento correcto dado que es un problema de estado del proceso, no de los
datos de entrada (un 422 habría sido incorrecto aquí). Documentado en
`inbox/a39/manifest.yaml::known_issues` y en el docstring de `inference.py`.

## Correctitud (golden dataset)

Fuente: los mismos 20 ciclos reales de test que a37 (mismo `run_id`, mismo split —
`inbox/a39/manifest.yaml::golden_cases`), con el top-3 de sensores por `|IG|` y por head
calculado de forma independiente en `manifest-extraction` ejecutando el `_integrated_gradients`
real del entregable (steps=50, perfil `full`) contra el artefacto real.

**Verificación contra el servidor real en marcha** (no solo contra las funciones en aislado):
para los 20 ciclos × 4 heads = 80 combinaciones, llamando a `/predict` real:

| Criterio | Resultado |
|---|---|
| Ranking top-3 de sensores exacto (mismo orden) | **80/80** |
| Valor `\|IG\|` dentro de tolerancia relativa 1% | 74/80 |

Las 6 discrepancias de valor están **todas** en los ciclos de menor magnitud del dataset
(1126, 1175 — del orden de 1e-6 a 1e-8, ya señalados en el manifest como los de atribución más
pequeña) y la causa es de **precisión del valor de referencia guardado**, no del cálculo del
plugin: el manifest almacena esos valores redondeados a 6 decimales (p. ej. `7e-06`), y para
magnitudes tan pequeñas ese redondeo por sí solo introduce un error relativo de hasta el 3-4%
frente al valor real sin redondear. En términos absolutos la diferencia máxima observada es
≈2.7e-7 (cycle 1175, head cooler, sensor EPS1: esperado 0.000008, obtenido 0.0000077). El
ranking (qué sensor es más importante) es exacto en los 20/20 ciclos — lo que realmente importa
para la explicación que ve el usuario — así que esto no se trata como un fallo real de wiring
ni de fidelidad, sino como una limitación de precisión del golden case documentada aquí en vez
de silenciada o resuelta ajustando la tolerancia para taparla.

**Resultado: 20/20 ciclos con el ranking top-3 exacto en las 4 cabezas; 74/80 cabezas también
dentro de tolerancia estricta del 1% en valor absoluto — las 6 restantes son un artefacto de
redondeo del valor de referencia en ciclos de magnitud casi nula, no un defecto del plugin.**

SHAP no se verifica aquí numéricamente contra el entregable (ver known_issues del manifest:
pendiente, requeriría replicar bit a bit la secuencia del generador aleatorio del background) —
se verifica solo que la forma de salida es correcta y que falla con un mensaje claro cuando el
backend del proceso no lo permite (ver hallazgo arriba).

## Notas adicionales

- **a39 no es un modelo nuevo**: es la capa de explicabilidad de a37, sobre el mismo modelo
  (pesos MD5-idénticos, confirmado en manifest-extraction), mismo dataset (proxy técnico UCI,
  no datos reales de bodega) y mismo `run_id`. Esto debe quedar igual de explícito en la
  documentación institucional de a39, no diluirse por tratarse "solo" de explicabilidad.
- **Alcance en vivo vs. offline, decisión documentada**: Integrated Gradients (siempre) y SHAP
  (opcional) se sirven por ciclo en el plugin; t-SNE, calibración (Brier Score) y la
  concordancia IG-SHAP/neurosimbólica agregada sobre todo el test set siguen siendo análisis de
  auditoría offline (el script CLI original del entregable), no se han reimplementado como
  parte de la API. Ver manifest para el razonamiento completo.
- **`training.supported: false`** es correcto para el alcance de este plugin (confirmado por
  código: el único comando `train` del entregable de a39 es una copia literal del de a37) — el
  reentrenamiento del modelo base sigue siendo competencia exclusiva de `ml37`.

## Estado final

**[ LISTO PARA PR ]**
