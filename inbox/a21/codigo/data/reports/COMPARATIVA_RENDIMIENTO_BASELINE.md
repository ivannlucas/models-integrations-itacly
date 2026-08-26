# Comparativa rendimiento baseline XGBoost (Caso 21)

## Resultados (Pearson y Directional Accuracy)

| Horizonte | Pearson previo | Pearson actual | Delta | DA previo | DA actual | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H1 | 0.1685 | 0.6209 | +0.4524 | 0.5691 | 0.6086 | +0.0395 |

## Analisis de impacto

- H1: Pearson improved; DA improved.

## Top features y nuevas variables

- H1: Top10 = precio_vecinos_media_lag1, precio_provincial_lag_1, wheat_intl_eur_ma3, precio_nacional_base_ma3, prepag2_torta de girasol_lag_1, prepag2_torta de girasol_lag_2, idx_fertilizantes_lag_2, idx_energia_lag_1, eur_usd_lag_2, prepag1_urea 46_lag_2. New variables in Top10: wheat_intl_eur_ma3, prepag2_torta de girasol_lag_1, prepag2_torta de girasol_lag_2, idx_fertilizantes_lag_2, eur_usd_lag_2, prepag1_urea 46_lag_2.