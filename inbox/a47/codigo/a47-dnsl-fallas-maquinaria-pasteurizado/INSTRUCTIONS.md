En este archivo se documenta la guía de uso paso a paso del sistema.

## 🚀 Guía de uso

El pipeline se encuentra orquestado por `src/main.py`, y convenientemente envuelto en los scripts dentro de `scripts/`. 

Para ejecutar las distintas fases, colóquese en la raíz del proyecto clonado y siga los siguientes pasos:

0. **Creacion del entorno virtual y configuración del entorno**: Tras situarse en la carpeta raíz del proyecto.
   Creación del entorno virtual:
   ```bash
   python -m venv venv
   ``` 
   Activar entorno (Windows):
   ```bash
   venv\Scripts\activate
   ```
   Activar entorno (Linux/Mac):
   ```bash
   source venv/bin/activate
   ```
   Instalar librerías:
   ```bash
   pip install -r requirements.txt
   ```
   > **Nota:** En la instalacion de torch en el archivo `requirements.txt`, se utiliza la url `https://download.pytorch.org/whl/cu130` porque el entorno se ha configurado originalmente para Python 3.13 y CUDA 13.0. Si da error en otro dispositivo, se debe visitar [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) para buscar el comando exacto para el sistema en cuestion. En caso de no tener tarjeta gráfica compatible, puedes simplemente ejecutar `pip install torch torchvision` (sin especificar `--index-url` y eliminandolo del archivo `requirements.txt` previamente), aunque esto podría implicar tener que entrenar el modelo utilizando únicamente la CPU.

1. **Descarga de datos y colocacion correcta de estos en la estructura** (Se deben descargar los datos del dataset [UCI Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) y colocar la carpeta resultante (con nombre `Dataset_Hydraulic`) en `data/raw/` para que el resultado final sea el siguiente):
   ```bash
   ├── data/
   │   ├── raw/
   |  ...    └── Dataset_Hydraulic/   # Esta carpeta debe contener los archivos .txt como:
   |                                  # CE, CP, description, documentation, EPS1, FS1, profile...
   ...
   ```

2. **Preprocesamiento de datos** (Lee los archivos .txt desde `data/raw/Dataset_Hydraulic` y genera un dataset base resampleado a 10Hz en `data/processed/hydraulic_10hz_raw.csv`):
   ```bash
   python -m src.main data_processing
   ```

3. **Entrenamiento** (Aplica Data Augmentation y entrena el modelo. Guarda componentes en `models/artifacts/`):
   ```bash
   python -m src.main train
   ```

   > **💡 Modo Fallback (Entrenamiento sin datos raw):** Si no se ha descargado el dataset original y no existe `data/processed/hydraulic_10hz_raw.csv`, pero la carpeta `data/splits/` contiene los archivos preprocesados (`train_split.csv`, `val_split.csv`, `test_split.csv`), el sistema lo detectará automáticamente. El pipeline omitirá las costosas etapas de preprocesamiento, *Feature Engineering* y Gemelo Digital (que ya vienen aplicadas en los *splits*) y entrenará el modelo directamente. Esta función es ideal para auditorías o compartición del modelo sin mover gigabytes de datos en crudo.

   *(Opcional)* Puedes calcular y guardar métricas de rendimiento final sobre el conjunto de Test en un `.csv` (Matriz de confusión completa, F1-Score, Recall, Precision, etc.) usando el argumento `--metrics`:
   ```bash
   python -m src.main train --metrics
   ```

4. **(Opcional) Extracción de Estadísticas del dataset**: Generará un reporte estadístico exportado a data/processed/estadisticas.csv, una descripción de cada una de las variables del dataset en data/processed/feature_descriptions.csv y una breve descripción del modelo y su uso por consola.
   ```bash
   python -m src.main get_stats
   ```

5. **Inferencia** (Genera predicciones a partir de los datos procesados):
   > Esta fase espera por defecto el archivo de datos crudos data/raw/hydraulic_raw.csv. El preprocesamiento (resampleo a 10Hz, feature engineering, etc.) se realiza internamente durante la inferencia. El modelo evaluará el estado de salud de los 4 componentes principales.
   > **Por defecto, la inferencia se realiza sobre las temperaturas reales** de los sensores, sin aplicar el desplazamiento térmico del Gemelo Digital.

   *   **Opción 1: Ejecución estándar** (Usa rutas por defecto y temperaturas reales de los sensores, es la opción que se debe aplicar sobre los datos obtenidos de la maquinaria real):
       ```bash
       python -m src.main predict
       ```
   
   *   **Opción 2: Especificar archivos de entrada/salida**:
       ```bash
       # Útil para procesar archivos nuevos o renombrados
       python -m src.main predict --input data/raw/nombre_entrada.csv --output data/predictions/nombre_salida.csv
       ```

   *   **Opción 3: Predecir un ciclo específico** (Filtra los datos por `Cycle_ID` antes de la inferencia):
       ```bash
       # Ejemplo para el ciclo 22
       python -m src.main predict --cycle 22
       ```

   *   **Opción 4: Modo Laboratorio / Dataset UCI** (Aplica el desplazamiento térmico del Gemelo Digital al baseline de 65ºC. Es la opción que se debe usar sobre los datos que provienen del dataset de laboratorio UCI, donde las temperaturas son de ~30ºC):
       ```bash
       # Usar cuando los sensores TS1/TS2 provengan del dataset de laboratorio UCI para simular el comportamiento de la máquina en condiciones reales
       python -m src.main predict --apply_digital_twin
       ```

   *   **NOTA:** La opción 1 es la que se debe aplicar sobre los datos obtenidos de la maquinaria real, para datos experimentales (dataset UCI), se debe usar la opción 4.

6. **Calibración del modelo a datos reales de planta (Fine-Tuning)**:
   > Antes de desplegar el modelo en producción, se recomienda calibrarlo con datos reales del fluido de planta (leche). El modelo preentrenado con aceite hidráulico (laboratorio UCI) tiene el backbone CNN congelado. Sólo se re-entrenan las 4 cabezas de clasificación (capas lineales finales), lo que requiere únicamente unos pocos cientos de ciclos reales para ajustar los umbrales de presión, densidad y viscosidad del nuevo fluido.
   >
   > Los CSV de entrada deben tener las mismas columnas de sensores que los datos de entrenamiento, más las columnas de etiquetas (`Target_Fouling`, `Target_Valvula`, `Target_Bomba`, `Target_Acumulador`).

   Ejecución estándar (leche entera, parámetros físicos por defecto):
   ```bash
   python scripts/fine_tune.py \
       --train_input data/planta/ciclos_train.csv \
       --val_input   data/planta/ciclos_val.csv
   ```

   Con parámetros físicos personalizados del fluido real:
   ```bash
   python scripts/fine_tune.py \
       --train_input  data/planta/ciclos_train.csv \
       --val_input    data/planta/ciclos_val.csv \
       --fluid_density 1.03 \
       --fluid_cp      3.93 \
       --epochs        50   \
       --patience      7
   ```

   El modelo calibrado se guarda en `models/artifacts/neurosymbolic_cnn_finetuned.pth` **sin sobreescribir** el modelo base. Se genera también un reporte de calibración en `models/artifacts/finetuning_report.txt`.

   > **Parámetros físicos de referencia:**
   > | Fluido          | Densidad (kg/L) | Cp (kJ/kg·K) |
   > |-----------------|-----------------|---------------|
   > | Leche entera    | ~1.03           | ~3.93         |
   > | Aceite (ref.)   | ~0.87           | ~1.88         |