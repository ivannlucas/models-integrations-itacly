# CERTIFICACION_DATAGIA_PROD

## Resumen
Certificacion de 6 modelos campeones con calibracion y benchmarks ingenuos.

## Stress test (sin torta de girasol)
Pearson base (H1 RF): 0.7305953898128889
Pearson stress: 0.7331924777299058
Caida relativa: -0.003554755413502031

## Benchmark regresion (vs retorno cero)
|   horizon | model   |   MAE_model |   RMSE_model |   DA_model |   MAE_naive |   RMSE_naive |   DA_naive |
|----------:|:--------|------------:|-------------:|-----------:|------------:|-------------:|-----------:|
|         1 | RF      |     5.11586 |      6.85937 |   0.581367 |     1.6335  |      2.16515 |          0 |
|         2 | RF      |     5.73031 |      7.41756 |   0.589901 |     2.18248 |      2.88436 |          0 |
|         3 | RF      |     6.59585 |      8.19957 |   0.61442  |     2.73253 |      3.52939 |          0 |

## Benchmark clasificacion (vs baselines)
|   horizon | model   |   DA_model |   AUC_model |   DA_persist |   DA_solo_sube |   DA_solo_baja |
|----------:|:--------|-----------:|------------:|-------------:|---------------:|---------------:|
|         1 | RF      |   0.714459 |    0.85761  |     0.529946 |       0.470054 |       0.529946 |
|         2 | RF      |   0.738052 |    0.820027 |     0.559589 |       0.440411 |       0.559589 |
|         3 | RF      |   0.730188 |    0.870231 |     0.588022 |       0.411978 |       0.588022 |

## Estabilidad por cereal (regresion)
|   horizon | cereal   |   Pearson |       DA |
|----------:|:---------|----------:|---------:|
|         1 | cebada   |  0.821931 | 0.610939 |
|         1 | trigo    |  0.864161 | 0.539474 |
|         2 | cebada   |  0.79394  | 0.62395  |
|         2 | trigo    |  0.822175 | 0.541667 |
|         3 | cebada   |  0.754322 | 0.653476 |
|         3 | trigo    |  0.680513 | 0.559091 |

## Estabilidad por cereal (clasificacion)
|   horizon | cereal   |   Pearson |       DA |      AUC |
|----------:|:---------|----------:|---------:|---------:|
|         1 | cebada   |  0.624291 | 0.732714 | 0.869745 |
|         1 | trigo    |  0.606718 | 0.688596 | 0.853856 |
|         2 | cebada   |  0.543602 | 0.692466 | 0.815363 |
|         2 | trigo    |  0.591298 | 0.802632 | 0.831747 |
|         3 | cebada   |  0.627627 | 0.711042 | 0.87378  |
|         3 | trigo    |  0.629228 | 0.75731  | 0.867992 |

## Modelos que baten baselines
Regresion:
|   horizon | model   | beats_naive   |
|----------:|:--------|:--------------|
|         1 | RF      | False         |
|         2 | RF      | False         |
|         3 | RF      | False         |

Clasificacion:
|   horizon | model   | beats_naive   |
|----------:|:--------|:--------------|
|         1 | RF      | True          |
|         2 | RF      | True          |
|         3 | RF      | True          |
