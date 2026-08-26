# TABLA_MAESTRA_DATAGIA

Resumen de campeones por horizonte (H1-H3).
Regresion: Pearson/MAE/RMSE/DA. Clasificacion: Accuracy/DA/AUC.
Apto: DA > 0.60.

## Campeones (regresion)
|   horizon | task   | model   |   Pearson |     MAE |    RMSE |       DA | Apto   | Top5                                                                                                                        |
|----------:|:-------|:--------|----------:|--------:|--------:|---------:|:-------|:----------------------------------------------------------------------------------------------------------------------------|
|         1 | reg    | XGB     |  0.648067 | 6.1289  | 7.59284 | 0.602541 | True   | prepag2_torta de girasol_lag_1, wheat_intl_eur_ma3, wheat_intl_eur_lag_1, prepag2_torta de girasol_lag_2, corn_intl_eur_ma3 |
|         2 | reg    | RF      |  0.719786 | 5.92753 | 7.23708 | 0.634236 | True   | prepag2_torta de girasol_lag_1, wheat_intl_eur_ma3, corn_intl_eur_ma3, prepag2_torta de girasol_lag_2, wheat_intl_eur_lag_1 |
|         3 | reg    | RF      |  0.737465 | 6.48721 | 7.75386 | 0.633229 | True   | prepag2_torta de girasol_lag_1, wheat_intl_eur_ma3, corn_intl_eur_ma3, prepag1_dap_lag_2, wheat_intl_eur_lag_1              |

## Campeones (clasificacion)
|   horizon | task   | model   |   Accuracy |       DA |      AUC | Apto   | Top5                                                                                               |
|----------:|:-------|:--------|-----------:|---------:|---------:|:-------|:---------------------------------------------------------------------------------------------------|
|         1 | clf    | RF      |   0.654567 | 0.654567 | 0.840145 | True   | eur_usd_lag_2, prepag1_urea 46_lag_2, wheat_intl_eur_ma3, prepag1_dap_lag_2, prepag1_urea 46_lag_1 |
|         2 | clf    | RF      |   0.724138 | 0.724138 | 0.803268 | True   | prepag1_urea 46_lag_2, eur_usd_lag_2, prepag1_dap_lag_2, eur_usd_lag_1, prepag1_dap_lag_1          |
|         3 | clf    | XGB     |   0.644283 | 0.644283 | 0.51628  | True   | prepag1_urea 46_lag_2, fase_siembra, fase_crecimiento, eur_usd_lag_2, eur_usd_lag_1                |

## Tabla unificada
|   horizon | task   | model   |    Pearson |       MAE |      RMSE |       DA | Apto   | Top5                                                                                                                        |   Accuracy |        AUC |
|----------:|:-------|:--------|-----------:|----------:|----------:|---------:|:-------|:----------------------------------------------------------------------------------------------------------------------------|-----------:|-----------:|
|         1 | clf    | RF      | nan        | nan       | nan       | 0.654567 | True   | eur_usd_lag_2, prepag1_urea 46_lag_2, wheat_intl_eur_ma3, prepag1_dap_lag_2, prepag1_urea 46_lag_1                          |   0.654567 |   0.840145 |
|         1 | reg    | XGB     |   0.648067 |   6.1289  |   7.59284 | 0.602541 | True   | prepag2_torta de girasol_lag_1, wheat_intl_eur_ma3, wheat_intl_eur_lag_1, prepag2_torta de girasol_lag_2, corn_intl_eur_ma3 | nan        | nan        |
|         2 | clf    | RF      | nan        | nan       | nan       | 0.724138 | True   | prepag1_urea 46_lag_2, eur_usd_lag_2, prepag1_dap_lag_2, eur_usd_lag_1, prepag1_dap_lag_1                                   |   0.724138 |   0.803268 |
|         2 | reg    | RF      |   0.719786 |   5.92753 |   7.23708 | 0.634236 | True   | prepag2_torta de girasol_lag_1, wheat_intl_eur_ma3, corn_intl_eur_ma3, prepag2_torta de girasol_lag_2, wheat_intl_eur_lag_1 | nan        | nan        |
|         3 | clf    | XGB     | nan        | nan       | nan       | 0.644283 | True   | prepag1_urea 46_lag_2, fase_siembra, fase_crecimiento, eur_usd_lag_2, eur_usd_lag_1                                         |   0.644283 |   0.51628  |
|         3 | reg    | RF      |   0.737465 |   6.48721 |   7.75386 | 0.633229 | True   | prepag2_torta de girasol_lag_1, wheat_intl_eur_ma3, corn_intl_eur_ma3, prepag1_dap_lag_2, wheat_intl_eur_lag_1              | nan        | nan        |
