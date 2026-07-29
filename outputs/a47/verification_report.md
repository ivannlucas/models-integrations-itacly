# Verificación — a47 (m47-dnsl-fallas-maquinaria-pasteurizado)

**Fecha:** 2026-07-16
**Plugin:** `app/plugins/m47_dnsl_fallas_maquinaria_pasteurizado/`
**Manifest:** `inbox/a47/manifest.yaml` (15 golden_cases)
**Entorno:** WSL2, Python 3.10.12, torch 2.1.2+cu121 (device=cuda), scikit-learn 1.7.0

---

## Checklist técnico (Parte A)

- [x] **flake8**: 1 hallazgo menor — `mlflow_utils.py:78 F841 local variable 'last_error' is assigned to but never used`. Preexistente, no afecta a la correctitud. **No se corrigió** para respetar el alcance "no tocar app/plugins/" de esta tanda; queda anotado para revisión humana.
- [x] **pytest** (`tests/unit/test_a47_...py -p no:flask`): **4/4 passed** (health, stats, predict_inline, predict_batch). ⚠️ Estos tests usan el `FakePlugin` del conftest: validan el **wiring HTTP + esquema Pydantic + mapeo de excepciones**, NO el modelo real.
- [x] **pylint** (`app/plugins/m47_.../`): **8.51/10**. Solo estilo (nombres `X_*` que replican el código original de forma intencionada, docstrings ausentes, `l_sup/l_pump/l_cool` sin usar). Sin issues de correctitud.
- [~] **pip-audit** (`requirements.txt`): 5 CVEs en 3 paquetes — `torch 2.11.0` (CVE-2025-3000), `setuptools 81.0.0` (PYSEC-2026-3447), `keras 3.12.3` (×3). **Preexistentes y a nivel de repo** (no introducidas por a47; `keras` ni siquiera lo usa este plugin). Flag para revisión transversal de dependencias.
- [x] **Carga + inferencia del plugin REAL**: artefactos copiados a `artifacts/a47_dnsl_fallas_maquinaria_pasteurizado/`; el plugin carga (`load()`) y ejecuta `predict_inline` y `predict_batch` correctamente sobre datos reales del test split.
  - Modo **inline**: verificado (ver Parte B).
  - Modo **batch**: smoke test con 2 ciclos (20, 210) → 2 predicciones correctas y coherentes con el golden.
- [~] **Arranque HTTP local (`main.py`) + curl**: **NO ejecutado**. El `main.py` completo no arranca en este WSL (dependencias pesadas de otros plugins ausentes). En su lugar se ejercitó el plugin por su ruta de código real (use-case → plugin), que es la lógica de producción sin la capa HTTP. Recomendado repetir health/predict/stats vía HTTP en un entorno con todas las dependencias.
- [~] **/train**: `manifest.training.supported = true` y el endpoint está registrado en `app/registry.py` (train_request/response cableados). **No se ejecutó un entrenamiento end-to-end** en local (proceso pesado y `train()` requiere `mlflow_run_id` + servidor MLflow para subir artefactos). Verificación del entrenamiento real pendiente de entorno con MLflow.

> Parte A en verde salvo los puntos marcados `[~]`, que son limitaciones del entorno local (no fallos del plugin) o hallazgos menores/preexistentes.

---

## Correctitud contra golden dataset (Parte B)

Fuente: `data/splits/test_split.csv` (331 ciclos, 601 filas/ciclo). Cada golden case = 1 ciclo completo (7 sensores × ~600 pasos @10Hz). `expected` = predicción real del modelo capturada en la extracción del manifest.
Verificación ejecutada contra el **plugin real** (`M47DnsFallMaquinariaPasteurizadoPlugin.predict_inline`), reconstruyendo la ventana de cada ciclo y comparando las 4 clases de componente.

**Tolerancia:** coincidencia **exacta** de clase (modelo de clasificación; `metrics_reported` no aporta MAE, por lo que la tolerancia numérica del 5 % por defecto no aplica — se exige match exacto de las 4 etiquetas). Orden de estados: `[Enfriador_Fouling, Valvula_Switch, Bomba_Leakage, Acumulador_Gas]`.

| Caso | Ciclo | Esperado | Obtenido | ¿OK? |
|---|---|---|---|---|
| cycle_020 | 20  | [2,0,0,0] | [2,0,0,0] | ✅ |
| cycle_023 | 23  | [2,0,0,0] | [2,0,0,0] | ✅ |
| cycle_210 | 210 | [2,0,2,0] | [2,0,2,0] | ✅ |
| cycle_211 | 211 | [2,2,2,1] | [2,2,2,1] | ✅ |
| cycle_219 | 219 | [2,2,2,0] | [2,2,2,0] | ✅ |
| cycle_231 | 231 | [2,1,2,0] | [2,1,2,0] | ✅ |
| cycle_233 | 233 | [2,1,2,0] | [2,1,2,0] | ✅ |
| cycle_251 | 251 | [2,0,1,0] | [2,0,1,0] | ✅ |
| cycle_254 | 254 | [2,2,1,0] | [2,2,1,0] | ✅ |
| cycle_256 | 256 | [2,2,1,0] | [2,2,1,0] | ✅ |
| cycle_270 | 270 | [2,1,1,0] | [2,1,1,0] | ✅ |
| cycle_275 | 275 | [2,1,1,0] | [2,1,1,0] | ✅ |
| cycle_282 | 282 | [2,0,1,0] | [2,0,1,0] | ✅ |
| cycle_294 | 294 | [2,2,0,0] | [2,2,0,0] | ✅ |
| cycle_303 | 303 | [2,1,0,0] | [2,1,0,0] | ✅ |

**Resultado: 15/15 casos reproducidos exactamente.** (Detalle numérico con confianzas en `outputs/a47/golden_results.json`.)

Nota sobre el caso 211: el `expected` es `[2,2,2,1]` mientras la etiqueta verdadera del ciclo es `[2,2,2,0]` — es un **error real del modelo** (falsa alarma de clase contigua en el acumulador, Sano→Warning), conservado a propósito. El plugin lo reproduce exactamente, que es justo lo que la verificación de reproducción debe comprobar. Coherente con el 98,79 % de Exact Match declarado.

---

## Estado final

**REQUIERE REVISIÓN HUMANA** — correctitud verificada en verde (15/15 golden), con los siguientes puntos para el revisor:

1. **Correctitud**: ✅ el plugin reproduce fielmente el pipeline original (inline y batch), 15/15 golden exactos.
2. **flake8 F841** en `mlflow_utils.py:78`: limpieza trivial pendiente (no se tocó por alcance).
3. **HTTP end-to-end y `/train`** no ejercitados en local (limitación del entorno WSL / falta MLflow). Repetir en entorno completo antes del PR.
4. **pip-audit**: CVEs preexistentes de dependencias compartidas (torch/setuptools/keras) — decisión transversal, no específica de a47.
5. **BLOQUEANTE para producción (no para PR)**: métricas obtenidas sobre banco de laboratorio UCI (aceite, no leche). El modelo **requiere reentrenamiento/calibración con datos reales de planta** antes de cualquier despliegue. Advertido en manifest y en ambas fichas.

_Este skill no abre PR ni hace merge — el gate humano es el último paso._
