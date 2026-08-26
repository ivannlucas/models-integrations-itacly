# Comparativa rendimiento tuning XGBoost (Caso 21)

## Resultados (H1)

| Modelo | Pearson | DA | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| Baseline reciente | 0.6209 | 0.6086 | - | - |
| Tuning v2 | 0.7480 | 0.6152 | 4.3159 | 6.1778 |

## Delta vs Baseline

- Pearson delta: +0.1271
- DA delta: +0.0066

## Analisis de importancia (Top 15)

- Top 15: precio_vecinos_media_lag1, precio_provincial_lag_1, wheat_intl_eur_ma3, idx_energia_lag_1, corn_intl_eur_ma3, prepag2_torta de girasol_lag_2, eur_usd_lag_2, prepag1_urea 46_lag_2, eur_usd_lag_1, precio_nacional_base_ma3, prepag2_torta de girasol_lag_1, idx_bienes_inversion_lag_2, idx_piensos_lag_3, idx_piensos_lag_2, idx_semillas_lag_1

- Urea (prepag1_urea) aparece en ranking #8.
- Trigo internacional (wheat_intl) aparece en ranking #3.

## Conclusion de produccion

- No Apto para Produccion (criterio: DA > 0.62 y Pearson > 0.40).