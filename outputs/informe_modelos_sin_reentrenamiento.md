# Informe de modelos sin reentrenamiento aplicable

**Repositorio:** `inference-pan-model` (DatagIA) · **Rama:** `fix/model-21-integration`
**Fecha:** 6 de octubre de 2026

---

## 1. Conclusión

De los 25 plugins registrados, **22 tienen reentrenamiento real operativo** (`POST /train`).
Exigen `mlflow_run_id` y nunca sobrescriben el artefacto base.

En **3 el reentrenamiento no aplica**: su motor desplegado no tiene parámetros aprendidos que
ajustar con datos. Es una propiedad del modelo entregado, no un hueco de la integración:

| Plugin | Motor desplegado | Por qué no hay nada que reentrenar |
|---|---|---|
| **ml28** · `ml28-meat-neuroevolutionary-raw-materials-prediction` | Motor de reglas deterministas (`platform_run`) | Su comportamiento lo fijan constantes de configuración, no pesos |
| **ml31** · `ml31-cereals-residue-optimizer` | Optimizador de Programación Lineal (PuLP + CBC) | Resuelve un problema exacto en cada petición; no hay modelo predictivo |
| **ml33** · `ml33-cereals-reuse-strategy-optimizer` | Optimizador MILP exacto (`scipy.optimize.milp`, HiGHS) | Resuelve una asignación exacta en cada petición; no hay modelo predictivo |

En los tres casos:

- `POST /train` responde **HTTP 501** (`TrainingNotSupportedError`) con un mensaje que explica
  el motivo. No hay `train_request_type` en `app/registry.py`.
- El plugin incluye `mlflow_utils.py`, porque la regla del repo es "sin excepción".
- `training.supported: false` en el manifest, con el motivo **re-verificado el 2 de octubre de
  2026 contra el código real entregado** en `inbox/aNN/codigo/`, no solo contra la memoria.
- La correctitud está verificada en verde (ver §3).

> **Los nombres no reflejan lo que se sirve.** Los tres modelos llevan "neuroevolutivo" en su
> nombre de carpeta o en la memoria. Esto se debe a versiones anteriores o a comparativas
> experimentales, no al motor que se sirve hoy. El nombre del paquete se mantuvo tal cual se
> pidió, pero la ficha técnica de cada modelo deja visible esta discrepancia.

---

## 2. Por qué se revisó con cuidado

Durante esta auditoría se encontraron **cinco modelos marcados como "no entrenables" que en
realidad sí lo eran**: ml2, ml4, ml7, ml17 y ml23. Su manifest o su plugin afirmaban que no se
había entregado código de entrenamiento, pero `inbox/aNN/codigo/` lo contenía. A los cinco se
les habilitó `/train`.

El caso de **ml23** se detectó al preparar estos informes. Su manifest decía *"sin
procedimiento de reentrenamiento entregado"*, pero el código incluía `scripts/train.py` y
`config.yaml` con `training_enabled: true`. Una vez portado, su `/train` reproduce a 4+
decimales las métricas del artefacto servido. Por eso **no figura en este informe**.

Para ml28, ml31 y ml33 se buscaron expresamente scripts y artefactos de entrenamiento en
`inbox/aNN/codigo/`, para descartar el mismo error. Lo que se encontró se explica en cada caso.

---

## 3. Detalle por modelo

### 3.1 ml28 — Aprovisionamiento de materia prima cárnica (CU28, NEUROCARN-OPT)

**Qué hace.** Es un sistema de apoyo a la decisión de compra. Para cada fila (materia prima,
stock, demanda prevista, plazo de entrega…) calcula tres cosas:
1. **Disparo de compra:** una probabilidad sigmoide sobre el déficit frente al stock de
   seguridad, más un *flag* que indica si el stock proyectado queda por debajo.
2. **Cantidad recomendada:** solo se calcula si se dispara la compra.
3. **Métricas agregadas del lote:** pedidos generados y reducción de excedente.

**Qué se sirve.** La función `src/cli/platform_run.py::run_platform_pipeline` del código
entregado, vendorizada en el plugin. Su comportamiento lo definen las constantes de
`config/platform_config.yaml`:
- `purchase_trigger_gap_sigmoid_scale = 5.0`
- `baseline_safety_stock_factor = 1.25`
- `max_stockout_increase_pct = 5.0`

**Por qué no aplica el reentrenamiento.**
- No hay pesos: cambiar el comportamiento del motor significa cambiar constantes de
  configuración, no ajustar un modelo con datos.
- Sí existe un pipeline de ML en el repo (`train_upstream_predictor.py`,
  `train_purchase_trigger.py`, `train_quantity_optimizer.py`, ruta `mixed_context`), con
  `LinearRegression`/`Ridge` y una comparativa neuroevolutiva. Tiene dos problemas:
  - **Entrena artefactos que el motor servido no carga.** El propio `DELIVERY_README_CU28.md`
    dice: *"La neuroevolución es una comparativa experimental offline; no es el modelo
    promovido"*.
  - **Sus columnas objetivo son sintéticas** (`synthetic_procurement_need`,
    `purchase_trigger_label`, `quantity_optimizer_target_tons`). Se generan con un ETL propio
    (contexto INE/MAPA más una capa sintética de planta), y `input_contract.md` excluye
    expresamente esas columnas del contrato de inferencia. Un usuario no podría aportarlas en
    un CSV sin replicar ese ETL completo.

**Verificación.** 5/5 golden cases reproducidos exactamente por HTTP. El batch completo del CSV
de demostración (20 filas) coincide con la ejecución directa del código original
(`triggered_orders=12`, `aggregate_excess_reduction_pct=20.959`).

**Qué haría falta para habilitarlo.** Una decisión de producto:
- **Opción 1:** promover uno de los modelos entrenados (el baseline lineal o la neuroevolución)
  a motor de producción.
- **Opción 2:** exponer el ajuste de las constantes de `platform_config.yaml` como
  "calibración", que no es un reentrenamiento ML.

Ninguna de las dos se ha pedido ni está respaldada por la entrega actual.

---

### 3.2 ml31 — Reducción de residuos vegetales en cereal (REOLSEC v2.0)

**Qué hace.** A partir de la asignación histórica de superficie por cultivo (secano/regadío),
resuelve un problema de **Programación Lineal**. El LP reasigna la superficie para minimizar el
residuo vegetal (o maximizar el beneficio). Hay 20 variables de decisión (2 por cultivo) y
restricciones duras (memoria, Tabla 9).

**Qué se sirve.** El solver LP (PuLP 3.3.2 con CBC embebido) junto con datos de referencia en
texto plano:
- `crop_economics.json`: precios y costes MAPA 2023/24.
- Ficheros de escenario.

No hay artefactos serializados (`.pkl`/`.pt`). Es determinista: la misma entrada produce
siempre la misma salida.

**Por qué no aplica el reentrenamiento.**
- La memoria lo dice textualmente (§4.9): *"No hay modelo que entrenar, ni artefacto
  serializado que cargar, ni semilla aleatoria"*.
- El residuo se **calcula** con una fórmula agronómica; no se **predice**.
- La versión 1.x sí usaba un MLP *surrogate* con un algoritmo genético. La v2.0 entregada lo
  retiró por completo (rediseño del 8 de junio de 2026). Releído el `README.md` del código real
  (2 de octubre de 2026): el LP va de principio a fin y no queda rastro de neuroevolución.
- No hay ningún fichero de entrenamiento en `inbox/a31/codigo/`.
- Lo que el usuario puede ajustar son los parámetros de escenario, que ya se pasan como
  entrada en cada `/predict`.

**Verificación.** 12/12 golden cases dentro de la tolerancia del 0,5%. Durante la auditoría se
detectó que las fichas institucionales describían el antiguo modelo ANN+GA. Se regeneraron para
describir el LP v2.0 que realmente se sirve.

**Qué haría falta para habilitarlo.** Nada que tenga sentido. Para actualizar el modelo hay que
actualizar `crop_economics.json` con nuevos precios y costes oficiales: es mantenimiento de
datos de referencia, no reentrenamiento.

---

### 3.3 ml33 — Estrategia de reutilización de subproductos cerealistas (CO₂)

**Qué hace.** Asigna cada lote de subproducto a una estrategia de reutilización. Lo hace
mediante **optimización exacta MILP** sobre bloques de `lots_per_day` lotes, con restricciones
de capacidad de planta, para minimizar las emisiones de CO₂ simuladas. Si el MILP fuera
infactible, entra una guarda defensiva (`capacity_fallback`); con la capacidad por defecto no
ocurre nunca.

**Qué se sirve.** `ExactEmissionsOptimizer` (`src/predict/exact_optimizer.py`): la función
determinista `strategy_emissions()` más `scipy.optimize.milp` (HiGHS). No carga pesos, ni
*scaler*, ni genoma.

**Por qué no aplica el reentrenamiento.**
- Es un solver exacto: no hay pesos que aprender ni variable objetivo que ajustar.
- **Sí existe código de entrenamiento NEAT** en el repo, pero pertenece a un *benchmark*
  retenido, no al modelo productivo:
  - `src/training/evolution.py`
  - `scripts/run_optimization.py`
  - el genoma `models/artifacts/winner_genome.pkl`
  - `models/metrics/training_fitness_history.json`

  El `README.md` del código entregado lo dice literalmente: *"The deployed decision engine is
  an exact optimizer… Neuroevolution (NEAT) is retained as a learned-policy benchmark and is
  bounded above by the exact optimum"*. `baseline_comparison.json` lo confirma: NEAT nunca
  supera al MILP. El plugin no expone el selector NEAT ni carga el genoma.
- **La memoria v2.0 lo confirma de forma explícita** (entregada después de redactar la primera
  versión de este informe, 21/07/2026). Su §10.3 dice: *"El modelo desplegado (optimizador
  exacto) no se reentrena: recalcula el óptimo en cada ejecución. Lo que requiere mantenimiento
  son sus dos entradas de conocimiento —la calibración del simulador de emisiones y los valores
  de capacidad/horizonte operativo—; el reentrenamiento, en su caso, solo afecta al benchmark
  NEAT"*.

**Verificación.** Correctitud exacta, con **0 discrepancias**, frente al golden dataset del
manifest y frente a las **10.000 filas** del split de test original.

**Qué haría falta para habilitarlo.** Solo tendría sentido si se decidiera servir la política
NEAT en lugar del MILP. Sería un retroceso, porque el MILP es el óptimo exacto y NEAT está
acotado por él.

---

## 4. Revisión posterior

Si en el futuro cambia la entrega de alguno de estos tres modelos (nueva versión del código,
cambio de motor desplegado o nueva memoria), hay que repetir la comprobación del §2. Consiste
en buscar en `inbox/aNN/codigo/` scripts y configuraciones de entrenamiento, y comprobar si
entrenan **el motor que realmente se sirve**. Existir código de entrenamiento no basta por sí
solo (ml28 y ml33 lo tienen), ni tampoco basta que el manifest diga que no existe (ml23 lo
decía y era falso).
