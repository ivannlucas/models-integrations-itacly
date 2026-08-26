# SISTEMA_INFERENCIA_DATAGIA_V1

## Resumen
Evaluacion de regresion por retornos y sistema hibrido (clasificacion calibrada + regresor de retornos).
Regla de inferencia: prob>0.65 y |retorno|>1%.

## Benchmark de retornos (vs retorno cero)
|   horizon | model   |   MAE_model |   RMSE_model |   MAE_baseline |   RMSE_baseline | beats_baseline   |
|----------:|:--------|------------:|-------------:|---------------:|----------------:|:-----------------|
|         1 | RF      |   0.0467301 |    0.0668391 |      0.0548817 |       0.0712449 | True             |
|         2 | RF      |   0.0597804 |    0.0891014 |      0.0736032 |       0.0960456 | True             |
|         3 | RF      |   0.0736517 |    0.11033   |      0.0923698 |       0.118706  | True             |

## Desempeno hibrido (DA en subset con senal)
|   horizon |   signals |   DA_hybrid |   DA_persist |   DA_solo_sube |   DA_solo_baja |
|----------:|----------:|------------:|-------------:|---------------:|---------------:|
|         1 |         0 |         nan |  nan         |      nan       |    nan         |
|         2 |         4 |           1 |    0         |        1       |      0         |
|         3 |        92 |           1 |    0.0108696 |        0.98913 |      0.0108696 |

## Top 5 variables que explican retornos
|   horizon | model   | Top5                                                                                                           |
|----------:|:--------|:---------------------------------------------------------------------------------------------------------------|
|         1 | RF      | eur_usd_lag_2, prepag1_urea 46_lag_1, wheat_intl_eur_ma3, prepag1_dap_lag_2, prepag1_dap_lag_1                 |
|         2 | RF      | prepag1_dap_lag_2, prepag1_dap_lag_1, idx_piensos_lag_1, prepag1_urea 46_lag_2, prepag2_torta de girasol_lag_2 |
|         3 | RF      | prepag1_dap_lag_2, prepag1_urea 46_lag_2, prepag2_torta de girasol_lag_2, prepag1_dap_lag_1, eur_usd_lag_1     |
