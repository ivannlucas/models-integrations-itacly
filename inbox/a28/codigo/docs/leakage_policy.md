# Leakage Policy

## Principle

La ruta oficial mixed_context separa claramente:

- senales upstream;
- trigger de compra;
- optimizacion de cantidad;
- simulacion/evaluacion de politica.

## Variables prohibidas como input upstream

- `purchase_trigger_label`
- `order_quantity_tons`
- `quantity_optimizer_recommendation_tons`
- `quantity_optimizer_target_tons`
- `excess_tons`
- `stockout_tons`
- `purchase_trigger_flag`
- `purchase_trigger_proba`

## Variables prohibidas como input del trigger

- cualquier salida del optimizador;
- cualquier metrica de simulacion ya calculada;
- cualquier reconstruccion directa del target de decision final.

## Variables permitidas al quantity optimizer

- la prediccion del trigger producida para la fila (`purchase_trigger_flag`);
- `purchase_trigger_proba` cuando exista;
- variables operativas disponibles antes de la decision;
- contexto del escenario.

`purchase_trigger_label` es una etiqueta supervisada disponible solo durante
entrenamiento y evaluacion. Esta prohibida como input de inferencia, incluido
el `quantity_optimizer`; durante inferencia debe usarse exclusivamente la
salida predicha del trigger.

## Split policy

- `validation` se usa para seleccion y calibracion.
- `test` se reserva para evaluacion final.
- No se ajustan hiperparametros usando resultados de `test`.
