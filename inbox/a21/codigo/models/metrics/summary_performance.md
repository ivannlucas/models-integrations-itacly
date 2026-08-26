# Resumen de Performance - DATAGIA

Generado UTC: 2026-07-24T09:53:22.189506+00:00

## Regresion (Train vs Test)

| Horizon | MAE Train | MAE Test | Gap MAE | Pearson Train | Pearson Test | DA Train | DA Test |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1 | 0.039027 | 0.050759 | 0.011732 | 0.6401 | 0.3785 | 0.6769 | 0.5591 |
| H2 | 0.046776 | 0.067436 | 0.020661 | 0.7092 | 0.2653 | 0.7159 | 0.5656 |
| H3 | 0.026498 | 0.101161 | 0.074663 | 0.9412 | 0.2093 | 0.8631 | 0.6250 |

## Clasificacion (Train vs Test)

Nota: en clasificacion, DA coincide numericamente con Accuracy (DA = Accuracy).

| Horizon | Acc Train | Acc Test | Gap Acc | AUC Train | AUC Test | Gap AUC | DA (=Acc) Train | DA (=Acc) Test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H1 | 0.7228 | 0.6172 | -0.1056 | 0.8073 | 0.7116 | -0.0957 | 0.7228 | 0.6172 |
| H2 | 0.7059 | 0.6612 | -0.0447 | 0.7958 | 0.7441 | -0.0517 | 0.7059 | 0.6612 |
| H3 | 0.7079 | 0.6261 | -0.0818 | 0.8123 | 0.7523 | -0.0600 | 0.7079 | 0.6261 |

## Hallazgos de Robustez

- Gap bajo entre Train/Test sugiere buena generalizacion.
- Gap alto (sobre todo en AUC o MAE) sugiere posible sobreajuste.
- Revisar la pendiente de residuos en 2022 para detectar sensibilidad a shocks.