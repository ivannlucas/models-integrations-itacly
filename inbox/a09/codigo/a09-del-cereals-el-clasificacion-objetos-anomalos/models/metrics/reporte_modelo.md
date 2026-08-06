# Reporte de evaluacion del modelo secuencial

## Resumen de metricas
- Metrica de seleccion: f1_macro
- Numero de ventanas de hold-out: 2220

## Que significa cada metrica
- Accuracy: porcentaje total de aciertos.
- Balanced accuracy: media del recall por clase; ayuda cuando hay desbalance.
- Precision macro: precision media entre clases, tratando todas por igual.
- Recall macro: capacidad media de recuperar cada clase.
- F1 macro: equilibrio entre precision y recall; es la metrica principal del pipeline.
- Log loss: calidad de las probabilidades; penaliza predicciones seguras pero equivocadas.

## Comparativa LSTM vs GRU
- LSTM: f1_macro=0.9328, accuracy=0.9302, balanced_accuracy=0.9346, precision_macro=0.9313, recall_macro=0.9346, log_loss=0.2097
- GRU: f1_macro=0.9436, accuracy=0.9414, balanced_accuracy=0.9452, precision_macro=0.9422, recall_macro=0.9452, log_loss=0.1830

## Modelo ganador
- GRU
- Metrica principal (f1_macro): 0.9436

## Criterios de aceptacion

- Resultado global: cumplido
- f1_macro >= 0.9000: actual=0.9436 -> OK
- recall_macro >= 0.9000: actual=0.9452 -> OK
- accuracy >= 0.9000: actual=0.9414 -> OK
- log_loss <= 0.2000: actual=0.1830 -> OK

## Lectura rapida
El modelo ganador es el que mejor equilibra acierto global, estabilidad por clase y calidad probabilistica sobre el hold-out.