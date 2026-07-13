Modelo de DNSL para la predicción de fallas inminentes en equipos de refrigeración y aireado de embutidos
Sector: Cárnico
Fecha: 19/03/2026
Índice
# Lista de Tablas
# Lista de Figuras
# Listado de Abreviaturas
#### Tabla 1 Listado de abreviaturas
# 1  Introducción
La anticipación de anomalías y fallas en los sistemas de refrigeración y aireado es de vital importancia para la sostenibilidad económica y la seguridad alimentaria en el sector cárnico. En un entorno industrial cada vez más dependiente de la precisión térmica y el control higroscópico, disponer de herramientas que permitan identificar con antelación desviaciones en el comportamiento de los equipos es clave para evitar mermas críticas de producto, optimizar el consumo energético y reducir los costes de mantenimiento correctivo.
En el contexto del secado y conservación de productos cárnicos, particularmente en procesos de maduración donde el control de la humedad y la temperatura tiene un peso determinante en la calidad organoléptica [1], las fallas en el sistema de aireado pueden provocar efectos irreversibles como el encostramiento (case hardening) o el crecimiento microbiano no deseado. La capacidad de identificar de manera temprana señales de ineficiencia en el compresor o saturación en los evaporadores aporta una ventaja diferencial en la planificación de paradas técnicas y en la garantía de la cadena de frío, mejorando la resiliencia operativa de las plantas de procesamiento [2].
En cuanto a los sistemas de refrigeración, su eficiencia operativa está intrínsecamente ligada a la estabilidad del ciclo de compresión de vapor, donde cualquier desviación en la relación de presiones o en las temperaturas de saturación impacta directamente en la capacidad de extracción de calor del evaporador [3]. La degradación del Coeficiente de Rendimiento (COP) debido a ineficiencias en el compresor o a la formación de escarcha en el intercambiador no solo incrementa el gasto energético, sino que altera las condiciones psicrométricas del aire, comprometiendo el control del punto de rocío necesario para un secado homogéneo [4]. Por tanto, la integración de principios termodinámicos en el diagnóstico permite distinguir entre fluctuaciones operativas normales y fallas que amenazan la integridad de la cadena de frío y la seguridad del producto [5].
En este proyecto se desarrolla un modelo predictivo neurosimbólico orientado a diagnosticar fallos en sistemas de refrigeración y aireado. El sistema no solo se apoya en algoritmos de aprendizaje Random Forest (RF) para detectar patrones en los datos de sensores, sino que integra una capa de razonamiento neurosimbólico basada en leyes termodinámicas y de psicrometría [6]. Este enfoque permite validar las predicciones de la IA (Inteligencia Artificial) frente a restricciones físicas reales [7]. Con ello, se busca dotar al sector cárnico de una herramienta de apoyo a la decisión que facilite una gestión proactiva del mantenimiento, contribuya a una planificación más informada y, en última instancia, asegure la máxima calidad del producto final bajo criterios de eficiencia energética y seguridad.
## Estado del Arte y Contexto Técnico
El auge de la industria apoyada en inteligencia artificial ha consolidado el uso de técnicas de Machine Learning para el mantenimiento predictivo; sin embargo, los modelos puramente estadísticos a menudo carecen de la capacidad de razonamiento causal y coherencia física necesaria en procesos críticos. En el ámbito de la industrial, la arquitectura neurosimbólica permite realizar lo que se conoce como grounding lógico, donde el conocimiento experto actúa como una restricción funcional sobre la salida del modelo estadístico, garantizando que cualquier diagnóstico sea físicamente plausible [8, 9]. La integración de estos modelos de aprendizaje estadístico (capaces de gestionar datos ruidosos) con lógica simbólica (capaz de aportar interpretabilidad) define el paradigma de la IA Neurosimbólica actual [10, 11].
Específicamente en sistemas de refrigeración y secado, el uso de marcos neurosimbólicos no solo mejora la precisión, sino que aporta una capa de explicabilidad (Explainable AI) vital en industrias reguladas. Investigaciones recientes destacan que, mientras los modelos de ensamble como Random Forest destacan en la percepción de patrones temporales, la inclusión de un motor de inferencia simbólico permite aplicar "votos de estabilidad" y reglas de consistencia física que reducen drásticamente los falsos positivos [10, 12]. El estado del arte sugiere que marcos como Logic Tensor Networks (LTNs) y DeepProbLog son la frontera de esta integración [10]. Los LTNs traducen reglas lógicas estrictas a funciones diferenciables, permitiendo su integración en el propio entrenamiento y haciendo que el modelo aprenda la física del problema [9, 13, 14]. Otra estrategia plausible es la de utilizar una capa neurosimbólica en el postprocesado, esto permite implementar reglas físicas  que no quieres que sean suaves o probabilísticas como hard constraints (por ejemplo saltos discretos por fallos de sensor).
# 2  Objetivos
El objetivo principal es desarrollar un sistema de mantenimiento predictivo neurosimbólico (DNLS) para detectar fallos en equipos de refrigeración y aireado de embutido. El desarrollo del sistema se ha dividido en dos entornos diferenciados (refrigeración y aireado) debido a que la física subyacente en cada proceso es distinta. Mientras que el sistema de refrigeración se centra en el ciclo de compresión de vapor, el de aireado requiere una gestión mucho más exhaustiva del control de humedad y la integración de sistemas de calor, factores críticos para el proceso de curación de los productos.
El proyecto utiliza como modelos base Random Forest (RF) y Logic Tensor Networks (LTNs), evaluando su rendimiento a través de tres configuraciones comparativas: una pura, una neurosimbólica y una neurosimbólica con voto por ciclo. La integración neurosimbólica es incorporada como una capa de post-procesado simbólico que aplica reglas físicas rígidas tras la inferencia; dicha capa es independiente del modelo base y aplicable tanto a RF como a LTN, permitiendo un análisis exhaustivo del impacto de la lógica simbólica en cada arquitectura.
Para ambos sistemas (refrigeración y aireado), se analiza el impacto de la lógica física sobre el modelo. La selección de estas arquitecturas responde a la necesidad de diferenciar fallos térmicos complejos (como la distinción entre suciedad en condensador y presencia de incondensables). Se utilizan Hard Constraints en el post-procesado para asegurar coherencia física (ej. errores de sensor > 2.5 desviaciones) y Soft Constraints en el modelo LTN para penalizar comportamientos termodinámicamente imposibles durante el aprendizaje.
La capa de voto por run consiste en agregar las inferencias individuales de cada instante temporal enriquecidas con lags y deltas para emitir un veredicto único por cada ciclo operativo (run). Al contar con etiquetas globales por ciclo, este mecanismo de voto permite reconciliar las predicciones puntuales con el estado real del equipo, actuando como un filtro de ruido que garantiza que el diagnóstico final responda a la tendencia general del proceso y no a fluctuaciones transitorias de los sensores.
Los objetivos específicos de desarrollo  son los siguientes:
Ingeniería de características: Generar un dataset con variables de dominio físicas que actúen como la base de conocimiento. Esto se realizará de forma independiente para cada uno de los modelos (refrigeración y aireado).
Evaluación y comparación de las dos arquitecturas propuestas: entrenar y comparar un modelo LTN (lógica integrada en la pérdida, es decir, en el propio entrenamiento) frente a un modelo de machine learning con post-procesado simbólico + voto, determinando a la vez cuál ofrece mayor estabilidad diagnóstica frente al ruido de los sensores.
Validación basada en “runs”: Implementar validación cruzada por ciclos operativos (run_id) para garantizar que el sistema es capaz de generalizar a diferentes equipos de la planta sin sesgos de entrenamiento.
# 3 Dataset y Pipeline de Datos (ETL)
## 3.1 Descripción del Dataset
La arquitectura se entrena y valida utilizando dos fuentes de datos diferenciadas, seleccionadas por su capacidad para representar fenómenos físicos complejos en entornos controlados:
Sistemas de Refrigeración: Se utiliza el Simulated Refrigerator Fault Diagnosis Dataset. Este conjunto de datos proporciona series temporales  generadas a partir de un modelo físico basado en primeros principios. La simulación modela el comportamiento dinámico de los componentes clave (compresor, condensador, válvula de expansión y evaporador) bajo ecuaciones termodinámicas. Incluye 13 clases que abarcan desde la operación normal hasta fallos críticos como el ensuciamiento del condensador (Condenser Fouling), ineficiencia del compresor, presencia de incondensables y derivas de sensores (Sensor Drift).
Sistemas de Aireado: Ante la ausencia de datasets públicos específicos para el secado de productos cárnicos, se ha generado un dataset sintético basado en evidencia científica. Los parámetros y comportamientos de este dataset se han derivado de investigaciones fundamentales sobre la maduración del jamón y el secado de vegetales [1, 6, 7], permitiendo simular el "Delta Higroscópico" y la transferencia de masa necesaria para entrenar los modelos neurosimbólicos. La lógica completa de la generación de este dataset se puede encontrar en la ruta scripts/generate_dataset_aireado.py.
Los datos en ambos datasets están organizados en ejecuciones (runs), donde cada una presenta condiciones variables físicas.
El uso de datos simulados y sintéticos responde a la dificultad técnica de instrumentar equipos reales hasta su fallo en una planta de procesamiento o de encontrar datasets que hagan lo propio. Por una parte, esto permite disponer de un balance de clases controlado, incluyendo fallos raros pero críticos que difícilmente se capturarían en un entorno real en poco tiempo. Sin embargo, aunque la fidelidad física es alta, los modelos deben interpretarse como una base de conocimiento robusta que requiere una fase de fine-tuning al aplicarse a maquinaria específica de planta. Esta brecha entre simulación y realidad se mitiga en este proyecto mediante el uso de la lógica neurosimbólica, que aporta una capa de principios físicos universales (termodinámica) que no dependen de la fidelidad absoluta de la muestra estadística, sino de leyes inmutables.
## 3.2 Variables e Ingeniería de Características
Modelo de refrigeración:
Para el modelo de refrigeración estas son las características primarias utilizadas (para ver todas las características crudas utilizadas, ver la Sección 4.2):
run_id y time_min: Identificadores que permiten segmentar cada ciclo operativo y mantener la coherencia cronológica de la serie temporal.
P_dis_bar y P_suc_bar: Presiones de descarga (alta) y succión (baja). Definen el trabajo termodinámico del compresor.
T_cab_meas y T_set: Temperatura real medida en la cabina y el punto de consigna deseado.
T_evap_sat y T_cond_sat: Temperaturas de saturación en los intercambiadores, derivadas de las presiones del refrigerante.
Para dotar al modelo de capacidad de razonamiento neurosimbólico, se han generado indicadores basados en conocimiento de dominio mediante feature engineering:
P_ratio: Relación de presiones (P_dis / P_suc). Un incremento indica posible ineficiencia o fallos como COND_FOUL.
T_cond_approach: Diferencial entre la temperatura de condensación y la ambiente (T_cond_sat - T_amb). Es el indicador clave para detectar suciedad en el condensador.
Sensor_error: Diferencia entre la temperatura medida y la real (T_cab - T_cab_meas), fundamental para identificar las clases SENSOR_DRIFT.
EEI (Energy Efficiency Index): Relación entre el consumo eléctrico y el salto térmico (P_comp_W / (T_cab - T_evap_sat)). Mide el esfuerzo energético para extraer calor.
P_dis_volatility: Desviación estándar móvil de la presión de descarga; detecta inestabilidades térmicas asociadas a carga excesiva de refrigerante o gases incondensables.
Lags y Deltas (5, 15, 45, 100 min): Capturan la inercia térmica y las tendencias de cambio. Permiten distinguir entre fluctuaciones rápidas (fallos de ventilador) y derivas lentas (ensuciamiento progresivo).
La variable objetivo es fault_id, una etiqueta multiclase que identifica el estado operativo del equipo entre 13 categorías posibles:
NORMAL: Operación correcta bajo parámetros de diseño.
Fallos de Componentes: Ineficiencia de compresor (COMP_INEFFICIENCY), degradación o fallo de ventiladores (EVAP_FAN_DEG / EVAP_FAN_FAIL), problemas de carga de refrigerante (UNDERCHARGE_MILD / UNDERCHARGE_SEVERE / OVERCHARGE) o suciedad en el condensador (COND_FOUL_MILD / COND_FOUL_SEVERE).
Fallos de Control: Derivas en el sensor de temperatura (SENSOR_DRIFT_PLUS / MINUS).
Fallos Complejos: Casos con más de un fallo combinados (UNDERCHARGE_AND_COND_FOUL) y presencia de gases no condensables (NON_CONDENSABLES).
Esta arquitectura de variables permite que el modelo no solo aprenda de la estadística, sino que aplique restricciones lógicas (como la relación directa entre P_ratio y EEI para validar sus predicciones.
Modelo de aireado:
Las variables base principales provienen del dataset sintético derivado de la literatura y monitorizan la interacción entre el aire y el embutido (para ver todas las características crudas utilizadas, ver la Sección 4.2):
run_id y time_min: Identificadores de ciclo y tiempo de residencia en cámara.
Kg_embutido: Masa total de producto cargado; variable fundamental para determinar la inercia del sistema.
T_cab y RH_cab: Temperatura y humedad relativa dentro de la cámara de aireado.
N_fan_Hz: Frecuencia de los ventiladores, que define la velocidad del aire sobre la superficie del producto.
T_evap_sat y P_comp_W: Variables del ciclo de refrigeración asociadas a la extracción de humedad y calor.
Para capturar los fenómenos complejos descritos en la literatura, se han implementado las siguientes características mediante feature engineering:
Delta Higroscópico (RH_error): Diferencia respecto al 75% de humedad relativa, umbral crítico para evitar defectos de maduración [10].
Ratio Aire/Carga (Air_Flow_Ratio): Evalúa si el flujo de aire es proporcional a la carga de embutido, parámetro clave según los estudios de Imre [7].
Eficiencia de Evaporación (Evap_Eff_Index): Relación entre el enfriamiento sensible y la deshumidificación [5]; detecta si el sistema está extrayendo agua de forma eficiente.
Potencia Específica por Carga (Specific_Power_Load): Energía consumida por cada kilogramo de producto fresco, indicador de eficiencia operativa [5].
Riesgo de Encostramiento (Encostramiento_Risk): Relación crítica entre alta ventilación y baja humedad, utilizada para predecir el endurecimiento prematuro de la superficie (case hardening) [1, 6].
Déficit de Presión de Vapor (VPD): Calculado mediante la fórmula de Tetens para determinar la fuerza motriz real de la evaporación. El VPD indica la capacidad del aire para absorber agua del producto, permitiendo identificar si el ritmo de secado es óptimo o excesivo [3, 15].
Análisis de Tendencias (Lags y Deltas 10, 30, 60 min): Capturan si el secado se ha estancado o si la humedad desciende demasiado rápido, además de suavizar ciclos de ventilación periódica mediante medias móviles (RH_roll_mean_20).
La variable objetivo clasifica el estado de la cámara de aireado en cuatro categorías que vinculan la performance técnica con la calidad del producto:
NORMAL: Proceso de maduración equilibrado según la curva teórica.
ENCOSTRAMIENTO: Riesgo de pérdida de calidad por secado superficial excesivo (clase crítica para el valor del producto).
SATURACIÓN/HIELO: Fallo en la extracción de humedad que puede comprometer la seguridad alimentaria.
FALLO VENTILADOR: Anomalía mecánica que detiene la transferencia de masa.
Para dotar al dataset sintético de aireado de una mayor fidelidad operativa y mitigar la brecha entre la simulación y la realidad, se ha implementado una degradación progresiva de variables durante la generación de los datos. Esto permite que estas variables no sean estáticas, sino que repliquen el decaimiento gradual que experimentarían en un sistema real, introduciendo una deriva temporal (t_prog) en las variables críticas y asegurando que los patrones de fallos reflejen transiciones graduadas hacia los estados críticos, forzando así al modelo a aprender la inercia del sistema y evitando que las predicciones se basen únicamente en ruido aleatorio. La modelización de la degradación progresiva se ha realizado de la siguiente forma para cada tipo de falla:
Encostramiento: La caída de la humedad relativa (RH_cab) provoca el endurecimiento superficial del producto. Ante esta situación, el sistema de control (basado en una lógica de secado agresiva) interpreta erróneamente la falta de humedad del aire como una necesidad de mayor circulación y aumenta la frecuencia del ventilador (N_fan_Hz). Este incremento, lejos de solucionar el problema, acelera la evaporación superficial, cerrando aún más la estructura del embutido y consolidando el defecto.
N_fan_Hz = Base_ventilador + (t_prog * 15) + Ruido_Gaussiano
RH_cab = 75 - (t_prog * 20) + Ruido_Lote
Saturación/Hielo: En el fenómeno de saturación / hielo, el descenso progresivo de la temperatura de evaporación (T_evap_sat) indica una transferencia térmica ineficiente por acumulación de escarcha, mientras que el incremento en la humedad relativa (RH_cab) evidencia la incapacidad del evaporador para extraer el agua del ambiente.
T_evap_sat = (T_cab - 8) - (t_prog * 10) + Ruido_Gaussiano
RH_cab = 75 + (t_prog * 20) + Ruido_Lote
Fallo Ventilador: En esta falla, la reducción progresiva de la frecuencia del ventilador (N_fan_Hz) representa la pérdida de capacidad mecánica del componente, la cual se manifiesta mediante una disminución en el flujo de aire efectivo e incrementos de ruido aleatorio asociados a vibraciones anómalas del motor.
N_fan_Hz = (Base_ventilador * (1 - (t_prog * 0.8))) + Ruido_Gaussiano
Donde t_prog representa el progreso temporal normalizado [0, 1] dentro del lote (run).
## 3.3 Análisis exploratorio de datos (EDA) e ingeniería de características
El análisis exploratorio de los datasets de refrigeración y aireado permite validar la coherencia de las señales de los sensores y confirmar la coherencia de los enfoques neurosimbólicos. Los resultados principales se sintetizan a continuación (recogidos enteramente los notebooks de EDA, notebooks/EDA/EDA_refrigeracion.ipynb y notebooks/EDA/EDA_aireado.ipynb).
Dataset de Refrigeración:
##### Figura 1: Balance de clases en el dataset de refrigeración
Estructura y balance de clases: El dataset de refrigeración presenta un balance de clases perfecto entre sus 13 categorías (12 tipos de fallo y estado NORMAL), como se observa en la Figura 1. El conjunto de datos consta de un total de 1.872.000 muestras, estructuradas en 1.300 ciclos operativos de 1.440 time-steps minutales cada uno (equivalentes a 24 horas de operación continua por ciclo). Esta distribución equitativa garantiza que el modelo cuente con el mismo volumen de información para caracterizar cada diagnóstico, facilitando el aprendizaje de fallos complejos en el ciclo de compresión. El dataset no contiene valores nulos.
Correlaciones y relaciones físicas: El EDA revela que la señal predictiva no reside en un único sensor, sino en la interacción entre ellos. La Figura 2 muestra que correlaciones directas entre variables crudas (ej. P_dis_bar, T_cab…) y el target (fault_id) de falla son bajas (|Coeficiente de Pearson| < 0,30). Esto confirma que el problema requiere capturar interacciones complejas para alcanzar una capacidad discriminativa útil.
##### Figura 2: Correlación lineal de las features con el target en el dataset de refrigeración
Para superar esta limitación, se ha introducido ingeniería de características basadas en conocimiento de dominio, cuyo objetivo es explicitar las leyes de la termodinámica que rigen el sistema. Estas variables ya se han descrito en la Sección 3.2.
Como se puede observar en la Figura 3, la mayoría de las variables introducidas a partir de la ingeniería de características acaban siendo entre las más importantes a la hora de realizar una predicción en el modelo (esto es, modelo sin implementación de LTNs). Esto es un indicio de que la ingeniería de características logró aportar al modelo información necesaria para discriminar los fallos de manera eficiente.
##### Figura 3: Top 20 de características más predictivas en el modelo de Random Forest sin LTNs para la predicción de fallas en refrigeración.
Multicolinealidad de características: Se observa una correlación extremadamente alta entre variables como T_cond_sat y P_dis_bar (|Coef. de Pearson| = 0,99), lo cual muestra una redundancia en el fenómeno capturado debido a las relaciones lineales entre presiones y temperaturas de los gases. Después de la ingeniería de características, T_cond_sat, T_cab_meas y P_suc_bar serán eliminadas por redundancia. Esta limpieza no solo previene el overfitting a un posible overfitting a sensores específicos, sino que garantiza que la red neuronal se focalice en las features complementarias.
Variables como run_id, fault_id y time_min se excluyeron también del entrenamiento para evitar el riesgo de overfitting y data leakage. El run_id actúa como un identificador único de experimento sin valor predictivo físico, mientras que el time_min podría inducir correlaciones que no generalizan a nuevos ciclos de operación debido a su naturaleza temporal y cíclica.
Dataset de Aireado:
Estructura y balance de clases: Este dataset presenta una estructura altamente equilibrada, compuesta por un total de 30.000 muestras. Concretamente, las clases NORMAL y ENCOSTRAMIENTO representan un 25,33% cada una (7.600 muestras respectivamente), mientras que las clases SATURACIÓN/HIELO y FALLO VENTILADOR constituyen un 24,67% cada una (7.400 muestras respectivamente), tal como se observa en la Figura 4. Este nivel de paridad garantiza que el modelo no presente sesgos hacia una clase mayoritaria, permitiendo desarrollar una capacidad discriminativa uniforme y robusta ante cualquier tipo de incidencia. Asimismo, el dataset mantiene una integridad técnica rigurosa al estar estructurado en 300 runs con 100 time-steps cada uno, garantizando una serie temporal continua y libre de valores nulos.
##### Figura 4: Distribución de clases en el dataset de aireado
Correlaciones y relaciones físicas: El dataset de aireado por el contrario si que presenta presenta variables con una dependencia lineal mayor respecto al objetivo (fault_id). Destaca especialmente la humedad relativa de la cabina (RH_cab), que registra una correlación de Pearson superior a 0,60 (ver Figura 5), siendo de esta forma un buen predictor del estado del proceso. No obstante, basar el modelo únicamente en variables crudas limitaría su capacidad para capturar la naturaleza dinámica del secado. Por ello, se ha procedido a implementar nuevamente una ingeniería de características basada en el conocimiento de los procesos de curado [2, 6, 7], explicitando las leyes físicas mediante las siguientes métricas. Estas variables están ya descritas en la Sección 3.2.
##### Figura 5: Correlación lineal de las características con el target en el dataset de aireado
Como se puede observar en la Figura 6, la mayoría de las variables introducidas a partir de la ingeniería de características acaban situándose entre las más importantes a la hora de realizar una predicción en el modelo (esto es, el modelo sin implementación de LTNs), destacando especialmente la relevancia de las variables de retardo (lag features). El hecho de que varios predictores principales correspondan a métricas temporales, como la versión de Evap_Eff_Index con retardos de 10 muestras, es un indicio de que la ingeniería de características logró aportar al modelo la información necesaria para discriminar los fallos de manera eficiente al capturar la trayectoria y la inercia del sistema. Esta predominancia de variables diseñadas bajo conocimiento de dominio, como el RH_error y la media móvil RH_roll_mean_20, demuestra que el diagnóstico del proceso de aireado depende críticamente de la evolución histórica y la estabilidad de la eficiencia de evaporación, permitiendo al algoritmo distinguir patrones de degradación física que las variables crudas instantáneas no logran identificar por sí solas.
##### Figura 6: Top 10 de características más predictivas en el modelo de Random Forest sin LTNs para la predicción de fallas en aireado.
## 3.4  Pipeline de ETL
El pipeline para el modelo de refrigeración sería el presentado en la Tabla 2.
#### Tabla 2: Pasos en el pipeline de ETL.
Para el modelo de aireado, el pipeline sería el mismo, con la diferencia de que en este modelo se generan sus propias características debido a una física más centrada en la interacción entre el flujo de aire y la transferencia de masa del producto (psicrometría), priorizando la dinámica de deshumidificación y el riesgo de encostramiento superficial.
# 4 Especificaciones Técnicas
## 4.1 Arquitectura y Algoritmos
El sistema propuesto para ambos modelos se basa en una arquitectura de clasificación multiclase, diseñada para identificar y diagnosticar estados de falla específicos en sistemas de refrigeración y procesos de aireado. Para ello, se ha seleccionado el algoritmo Random Forest para el sistema de predicción, debido a su robustez en entornos industriales con datos tabulares y su capacidad para manejar relaciones no lineales complejas entre variables termodinámicas. Estos modelos se alimentan del conjunto de variables crudas e introducidas mediante ingeniería de características que capturan la eficiencia energética, diferenciales de control y dinámicas temporales del sistema.
Los Random Forest se han configurado mediante una búsqueda de hiperparámetros (RandomizedSearchCV) optimizada para maximizar el F1-Score pesado, asegurando un equilibrio entre precisión y sensibilidad (recall) en todas las categorías de fallo. Se ha implementado con un número de estimadores y una profundidad de árbol controlada para mitigar el riesgo de sobreajuste en experimentos específicos, validando su rendimiento mediante un esquema de GroupKFold basado en el identificador de run (run_id).
El sistema incorpora además una capa de lógica neurosimbólica en su etapa final. Esta capa actúa como un filtro supervisor que utiliza reglas físicas deterministas (como leyes de presión-temperatura y ratios de flujo de aire) para refinar las predicciones del Random Forest. Esta arquitectura híbrida garantiza que el sistema no solo aprenda patrones estadísticos del dataset histórico, sino que sus diagnósticos sean físicamente coherentes, reduciendo drásticamente los falsos positivos causados por el ruido o el desplazamiento de los sensores (sensor drift).
En otra versión de los modelos se han implementado, en vez de Random Forest con post procesamiento neurosimbólico, LTNs como una posible evolución hacia la involucración de razonamiento lógico. Esta aproximación permite integrar el conocimiento experto del dominio directamente en el proceso de aprendizaje de la red neuronal mediante la definición de predicados lógicos y restricciones de satisfacción. Las LTNs optimizan una función de pérdida que incluye un término de "satisfacibilidad", forzando al modelo a que sus predicciones no solo sean precisas, sino que cumplan con los axiomas físicos preestablecidos.
## 4.2 Entradas y salidas del sistema
Entradas
Tipo de Datos: Series temporales provenientes de sensores industriales (presiones, temperaturas, humedad y potencia).
Formato Físico: DataFrame de Pandas en memoria o archivos .csv estructurados por cada experimento o "run".
Estructura esperada para inferencia: Registro continuo de señales con una frecuencia de muestreo de alta resolución.
Variables crudas requeridas (modelo de refrigeración): T_amb (temperatura ambiente, en °C), T_set (temperatura de consigna configurada en el refrigerador, en °C), T_cab (temperatura interna de la cabina, en °C), T_evap_sat (temperatura de saturación en el evaporador, en °C), T_cond_sat (temperatura de saturación en el condensador, en °C), P_suc_bar (presión de succión del compresor, en bar), P_dis_bar (presión de descarga del compresor, en bar), N_comp_Hz (frecuencia de funcionamiento del compresor, en Hz), SH_K (recalentamiento del refrigerante a la salida del evaporador, en K), P_comp_W (consumo de potencia eléctrica del compresor, en vatios), Q_evap_W (capacidad de refrigeración en el evaporador, en vatios), COP (coeficiente de rendimiento, relación entre la capacidad de refrigeración y la potencia del compresor), frost_level (nivel estimado de acumulación de escarcha en el evaporador), T_cab_meas (temperatura interna de la cabina medida por el sensor, en °C), valve_open (porcentaje de apertura de la válvula de expansión), door_open (indicador de puerta abierta) y defrost_on (indicador de desescarche activo).
Variables crudas requeridas (modelo de aireado): T_amb (temperatura ambiente, en °C), T_set (temperatura de consigna configurada para el proceso de aireado, en °C), Kg_embutido (masa total del producto cargado en la cabina, en kg), N_fan_Hz (frecuencia de operación de los ventiladores de la cabina, en Hz), T_cab (temperatura del aire en el interior de la cabina, en °C), RH_cab (humedad relativa del aire en el interior de la cabina, en %), T_evap_sat (temperatura de saturación en el evaporador del sistema de tratamiento de aire, en °C) y P_comp_W (consumo de potencia eléctrica del compresor del sistema de aireado, en vatios).
Identificador de sesión (run_id) para el particionado y validación por grupos.
Salidas
Tipo de Salida: Diagnóstico de clasificación multiclase sobre el estado de salud del sistema (categorías de falla o estado normal).
Formato Físico: Archivos .csv o logs de diagnóstico que incluyen:
Índice temporal: Marca de tiempo absoluto o time_min relativo al inicio de la corrida.
Vector de Características: Todas las variables de ingeniería calculadas (lags, deltas, EEI, P_ratio, Encostramiento_Risk, etc.).
Probabilidades de Clase: Columnas con la probabilidad estimada para cada tipo de fallo (ej. prob_normal, prob_encostramiento, prob_fallo_ventilador).
Predicción Final: Etiqueta del fallo detectado tras pasar por el filtro de la capa de lógica neurosimbólica, garantizando coherencia física en el diagnóstico.
## 4.3 Requisitos de Hardware y Entorno
El desarrollo, entrenamiento y evaluación de los modelos se han realizado en un entorno de Python 3.12 sobre Windows 11, optimizado para el procesamiento de datos de sensores y la construcción de modelos de clasificación multiclase (Random Forest). Se recomienda tener una versión de Python 3.12 o superior, ya que garantiza la total paridad con el entorno de desarrollo original, evitando discrepancias en la serialización de los modelos (.pkl) y aprovechando las mejoras de rendimiento en el manejo de estructuras de datos masivas. No obstante, el sistema mantiene una compatibilidad hacia atrás hasta la versión 3.9, permitiendo el uso de operadores de tipado nativos y asegurando la estabilidad de las librerías de cálculo científico en entornos industriales menos actualizados. La gestión de dependencias mediante un entorno virtual y un archivo requirements.txt asegura la reproducibilidad total del pipeline, desde la ingeniería de características hasta la inferencia final.
Hardware
CPU: Procesador x86-64 (mínimo 4 vCPU recomendadas). Aunque el entrenamiento es eficiente, el cálculo de múltiples lags (10, 30, 60) y medias móviles sobre grandes volúmenes de datos de sensores se beneficia de la ejecución en paralelo.
Memoria RAM: Se recomiendan al menos 8 GB como mínimo operativo. Aunque el sistema de desarrollo cuenta con 64 GB de RAM, permitiendo una manipulación fluida de datasets extensos en memoria, el pipeline está optimizado para ejecutarse en equipos con recursos más limitados.
GPU: No requerida para el flujo de producción basado en Random Forest, para los modelos de LTNs se utilizó en fases experimentales, sin embargo, debido al poco tiempo de entrenamiento, se acabó utilizando CPU.
Almacenamiento: 1 GB para el código fuente, almacenamiento de modelos serializados (.pkl) y los registros históricos en formato CSV.
Tiempos de Ejecución
En la Tabla 3 se documentan los tiempos medios obtenidos con el hardware de referencia durante el proceso de desarrollo.
#### Tabla 3: Tiempos de entrenamiento e inferencia registrados.
El ecosistema de herramientas se divide según las necesidades de cada modelo, garantizando que tanto el aprendizaje estadístico como el basado en reglas (LTN) tengan el soporte adecuado:
Librerías de Procesamiento
pandas & numpy: Pilares para la manipulación de las series temporales de sensores y cálculos matriciales de ingeniería.
scikit-learn: Utilizada para el particionado avanzado (GroupKFold, StratifiedGroupKFold), el escalado de variables (StandardScaler) y el motor de clasificación Random Forest.
joblib: Para la serialización de modelos y escaladores, permitiendo su despliegue posterior.
scipy & math: Operaciones matemáticas para la definición de reglas físicas.
Librerías de Deep Learning y Lógica (LTN)
TensorFlow: Motor de ejecución para la rama experimental de redes neuronales.
LTN (Logic Tensor Networks): Framework específico para integrar el conocimiento experto y los predicados lógicos en el aprendizaje profundo.
Visualización y Análisis
matplotlib & seaborn: Generación de gráficos de importancia de variables, matrices de confusión y tendencias temporales.
plotly.express: Empleada en el modelo de refrigeración para análisis exploratorios interactivos.
# 5 Procesos de entrenamiento y validación
## 5.1 Descripción de los Procesos
Una vez generado el conjunto de datos procesado a partir de las señales de sensores y la ingeniería de características (ver Sección 3.2), se entrena un modelo de clasificación multiclase donde cada observación representa un estado temporal del sistema y la etiqueta indica la presencia de un fallo específico o el estado normal de operación. Sobre este dataset se ajustan modelos de Random Forest Classifier, elegidos por su robustez ante volúmenes de datos industriales, baja complejidad computacional en inferencia (ver Tabla 3 para tiempos de inferencia) y facilidad de explicabilidad mediante la importancia de variables.
Todas las características (lags de temperatura, presiones, ratios de eficiencia) son normalizadas mediante un escalador (StandardScaler) ajustado exclusivamente sobre los datos de entrenamiento para evitar el sesgo de información (data leakage). Aunque para Random Forests este escalado no es estrictamente necesario, ya que son invariantes a escala, siempre es una buena práctica y es necesario si en el futuro se quiere implementar otro modelo.
Las arquitecturas de Deep Learning (LTN) se han utilizado de forma exploratoria para validar la convergencia de reglas lógicas. En el entrenamiento de estas arquitecturas, el modelo de aireado alcanza una precisión cercana al 100% rápidamente (epoch número 40), mientras que el de Refrigeración requiere un proceso híbrido más extenso (600 épocas) para equilibrar la precisión estadística con la consistencia lógica.
## 5.2 Hiperparámetros globales
La configuración global de los modelos de producción es:
random_seed = 42. Semilla para garantizar la reproducibilidad de los experimentos.
n_jobs = -1. Uso de procesamiento paralelo para la búsqueda de parámetros.
## 5.3 Hiperparámetros para la optimización
### Sistema Aireado:
Número de árboles (n_estimators): 200
Profundidad máxima (max_depth): 20
Mínimo de muestras por hoja (min_samples_leaf): 4
Mínimo de muestras para división (min_samples_split): 5
Uso de Bootstrap: False
### Sistema Refrigeración:
Número de árboles (n_estimators): 200
Profundidad máxima (max_depth): 20
Mínimo de muestras por hoja (min_samples_leaf): 20
Máximo de características (max_features): 'sqrt'
Uso de Bootstrap: True
## 5.4 Estrategia de validación
La validación se basa en un esquema de Validación Cruzada por Grupos (GroupKFold / StratifiedGroupKFold). En lugar de una partición aleatoria simple, se utiliza el run_id como identificador de grupo para asegurar que todas las muestras de una misma corrida experimental permanezcan juntas en el mismo bloque (entrenamiento o validación).
Para el Aireado, se utilizan 5 folds con GroupKFold.
Para la Refrigeración, se emplean 5 folds con StratifiedGroupKFold debido a la mayor complejidad. El uso de StratifiedGroupKFold garantiza que cada partición mantenga esta distribución equitativa de clases mientras respeta la integridad de los run_id.
En ambos procesos de validación cruzada, se ha establecido el f1_macro como la métrica de puntuación (scoring), garantizando que todas las categorías de estado y fallo tengan el mismo peso en la evaluación final.
# 6 Métricas y rendimiento
Para fundamentar la arquitectura final, se evaluaron dos enfoques principales: Random Forest, por su eficacia con datos tabulares, y Logic Tensor Networks (LTN), para integrar conocimiento experto en el entrenamiento. En ambas familias se implementaron y compararon tres niveles de complejidad: una versión base con lógica convencional, una versión integrada con una capa Neurosimbólica (NS) para asegurar coherencia física, y una configuración avanzada que combina la capa NS con un sistema de voto por run para maximizar la estabilidad diagnóstica. Este análisis permite identificar el impacto directo de cada componente en la fiabilidad del sistema. Nos hemos basado en las métricas macro-avg de precisión, sensibilidad y F1, que permiten obtener una visión equilibrada del rendimiento del sistema, garantizando que el diagnóstico de fallos críticos y poco frecuentes tenga el mismo peso que el estado de funcionamiento normal. Al promediar los resultados de cada clase de forma independiente, esta métrica evita que el elevado volumen de datos del estado NORMAL enmascare posibles deficiencias en la detección de anomalías específicas.
## 6.1  Criterios de éxito
El éxito de un algoritmo de detección de fallos en refrigeración o sistemas similares de fallas instrumentales reside en una alta precisión, bajo tiempo de cómputo y reducida tasa de falsos positivos [16]. Una clasificación exacta agiliza el mantenimiento, mientras que la eficiencia computacional reduce el tiempo de respuesta y los costes de hardware, lo cual se evaluará positivamente a la hora de elegir el modelo final empleado para el despliegue. Asimismo, minimizar los falsos positivos aumenta la fiabilidad y reduce los gastos por avisos de servicio innecesarios. Bajo este marco, se establecieron los siguientes objetivos, alineados con los estándares de desempeño hallados en la literatura técnica [16]:
F1-Score (Macro) ≥ 0,85: Asegura un equilibrio entre precisión y sensibilidad, evitando que el modelo se sesgue hacia las clases mayoritarias.
Recall (Sensibilidad) ≥ 0,90: Criterio de seguridad mínimo que garantiza la detección de al menos 9 de cada 10 fallos críticos.
Precisión ≥ 0,85: Guardarraíl operativo para minimizar intervenciones de mantenimiento innecesarias.
## 6.2  Resultados Comparativos
#### Tabla 4: Comparativa de métricas entre los distintos modelos utilizados, tanto en el modelo de aireado como en el de refrigeración
6.3 Análisis de resultados y selección de modelos
Modelo de Aireado:
En condiciones nominales (sin perturbaciones), el sistema de aireado presenta un escenario de alta separabilidad geométrica en el espacio de fases. Tanto el modelo base Random Forest puro como la arquitectura de Logic Tensor Networks (LTN) alcanzan un F1-Macro perfecto de 1,00, confirmando que las firmas térmicas y de humedad de los fallos de Encostramiento, Saturación y Ventilador están óptimamente bien definidas.
Para someter a examen los límites de ruptura de las arquitecturas y simular entornos industriales hostiles, se integró en el orquestador principal un experimento de estrés (stress_test) parametrizable desde el archivo de configuración centralizado (config.yaml). En esta sección, el test se configuró con un nivel crítico de ruido del 30% (noise_level: 0.30) mediante la inyección de ruido gaussiano blanco sobre las variables escaladas (lo que representa introducir una dispersión equivalente a casi las tres cuartas partes de la varianza nominal del proceso), complementado con artefactos tipo spikes aleatorios (include_spikes: true) con una probabilidad de aparición del 1% (1 de cada 100 muestras).
Los resultados de este test (recogidos en la Tabla 4) justifican plenamente la selección del modelo final y demuestran la interacción dinámica de sus componentes:
Random Forest Puro con Ruido (F1 Score 0,84): El modelo base degrada su rendimiento estadístico al deformarse las fronteras de decisión por la varianza del ruido gaussiano.
Random Forest + Capa Neurosimbólica con Ruido (F1 Score 0,84): La inyección rígida de las reglas físicas a posteriori puede penalizar el rendimiento general en instantes de ruido pico, ya que los spikes empujan los datos a zonas prohibidas por la termodinámica, forzando falsos descartes y bajando la precisión, aun así, el sistema se mantiene estable.
Random Forest + NS + Voto por Run con Ruido (1,00): La ventana temporal del voto por run absorbe por completo la varianza estocástica del 30% de ruido gaussiano.
Por tanto, se selecciona la arquitectura de Random Forest + Capa Neurosimbólica + Voto por run para el despliegue. No solo iguala el F1-Macro de 1,00 de las LTN, sino que lo hace garantizando inmunidad absoluta ante fallos catastróficos de instrumentación en planta, con una simplicidad de despliegue y coste computacional drásticamente inferiores a la infraestructura de tensores.
Modelo de Refrigeración:
En comparación al modelo de aireado, la refrigeración presenta un reto mayor, especialmente en la diferenciación de las clases de COND_FOUL_SEVERE y NON_CONDENSABLES, las cuales presentan comportamientos muy parecidos en el sistema: ambas producen un aumento de la presión de descarga (P_dis_bar), y de la temperatura de saturación del condensador (T_cond_sat), a la vez que una bajada en el COP. Es la correcta diferenciación entre estas dos clases donde se centra la mayoría de nuestro esfuerzo neurosimbólico para este modelo.
A nivel de arquitectura de software, se mantiene una simetría total con el sistema anterior, permitiendo activar un experimento de estrés paramétrico independiente desde el config.yaml. No obstante, para la obtención de las métricas base del informe, el flag se configuró en desactivado (active: false), debido a que este dataset procede de un simulador físico dinámico que ya incorpora de forma nativa sus propios modelos estocásticos de ruido de medición y derivas térmicas. Aplicar ruido adicional en condiciones estándar habría provocado una sobre-distorsión artificial (double-noising).
Para validar el peor escenario operativo posible, se realizaron pruebas de estrés secundarias activando un nivel de ruido del 10% (noise_level: 0.10) y spikes analógicos (include_spikes: true). El análisis de los resultados de la Tabla 4 consolida las siguientes conclusiones:
Limitación de las LTN: Aunque las LTN son ligeramente más robustas inicialmente que un RF puro en entornos ruidosos (F1-Macro de 0,91 frente a 0,87), la optimización continua basada en tensores lógicos no consigue aislar por completo el solapamiento termodinámico de las clases más complejas.
Superioridad del Modelo Híbrido: La arquitectura combinada de Random Forest + Capa Neurosimbólica + Voto por run supera a todos los enfoques evaluados, alcanzando un F1-Macro de 0,94. Lo más destacable es que el modelo es tan robusto que su rendimiento se mantiene imperturbable en 0,95 tanto en condiciones normales como bajo el efecto del 10% de ruido del test de estrés.
La capacidad del filtro neurosimbólico para explotar sutiles gradientes de recuperación temporal e inyectar leyes físicas a posteriori corrige de forma quirúrgica las clasificaciones erróneas entre el ensuciamiento severo y los gases incondensables, elevando el Recall global y consolidando esta arquitectura como la solución óptima para su despliegue en producción.
6.4 Orquestación y Criterios de Parada
El ciclo de vida del modelo (procesamiento, entrenamiento, tuning y evaluación) se gestiona mediante un orquestador centralizado (src.main) que garantiza la reproducibilidad total del experimento. El usuario interactúa con el sistema mediante comandos atómicos, permitiendo ejecutar el pipeline completo o fases específicas. El sistema realiza de forma automática la ingesta, la ingeniería de features basada en física, el ajuste de hiperparámetros, la validación cruzada y la exportación de artefactos.
Estrategia de Validación y Criterios de Parada
A diferencia de los modelos estándar que detienen su aprendizaje por épocas, nuestro sistema utiliza criterios basados en la robustez estadística y la estructura de los datos industriales:
Random Forest (Modelo Core): El criterio de parada está definido por la convergencia en el espacio de búsqueda durante la fase de tuning. Se utilizan un número fijo de estimadores y límites de profundidad para evitar el sobreajuste.
Optimización de Hiperparámetros (Tuning): El proceso de búsqueda se rige por un número determinista de iteraciones (configuradas en el módulo de tuning). El sistema selecciona la combinación que maximiza el F1-Score macro, asegurando un equilibrio entre precisión y sensibilidad en fallos críticos.
Validación por Grupos: El entrenamiento se valida mediante GroupKFold y StratifiedGroupKFold. Esto garantiza que el sistema solo se considera "entrenado" si es capaz de generalizar su diagnóstico a ciclos de trabajo completos (run_id) que no ha visto previamente.
Refinamiento Neurosimbólico
Una vez finalizado el entrenamiento del modelo tabular, el sistema aplica una capa de criterios expertos. Esta fase no es estadística, sino determinista:
Reglas Físicas: Corrección de predicciones ML mediante umbrales de presión y temperatura.
Consenso por Ciclo: Aplicación de un voto por mayoría sobre el run_id. El criterio de parada aquí es el fin del historial del ciclo; hasta que no se procesa el último minuto del run_id, el sistema no emite el diagnóstico final definitivo.
6.5 Impacto de la arquitectura neurosimbólica
El éxito de estas métricas radica en la superposición de capas lógicas que refinan la salida del algoritmo de Machine Learning puro:
Nivel 1: Modelo Base: El modelo puro del sistema de refrigeración presenta dificultades inherentes al distinguir fallos con firmas térmicas similares (como la presencia de incondensables frente a suciedad severa en el condensador), con F1-Scores en las clases problemáticas iniciales en el rango de 0,32 a 0,40 (ver matriz de confusión representada en la Figura 7) .
##### Figura 7: Matriz de confusión del modelo base de random forest en el sistema de refrigeración.
Nivel 2: Refinamiento Neuro-simbólico: La integración de reglas basadas en la física de fluidos actúa como un filtro correctivo. Al aplicar umbrales físicos sobre las predicciones, el rendimiento en las clases más complejas se eleva drásticamente.
Nivel 3: Voto por Ciclo (Run-id): Esta capa final consolida la estabilidad del diagnóstico, útil también para futuras calibraciones y la resistencia del modelo ante el ruido (ver Tabla 4). En el sistema de aireado, este mecanismo es el responsable de elevar el Accuracy del 0,93 al 1,00, ya que neutraliza lecturas erróneas de sensores o ruidos transitorios mediante un consenso mayoritario a lo largo de todo el ciclo de trabajo. En el sistema de refrigeración, esto hace que el F1-Score de las clases problemáticas se eleve hasta 0,58 y 0,62 (ver matriz de confusión representada en la Figura 8).
##### Figura 8: Matriz de confusión del modelo de random forest con posprocesamiento neurosimbólico y voto por run en el sistema de refrigeración.
6.6 Conclusión de las métricas y el rendimiento
La arquitectura final implementa una estrategia diversificada: un modelo de ensamble estadístico (RF) potenciado por una supervisión lógica Neurosimbólica y un consenso temporal (Voto). Aunque se evaluaron arquitecturas de LTN de forma exploratoria para verificar la viabilidad de modelos basados en tensores lógicos, el pipeline oficial de producción se fundamentará en los Random Forests por su estabilidad y menor requerimiento de recursos en inferencia.
En el aireado, esta combinación elimina cualquier residuo de ruido en los sensores, mientras que en refrigeración, actúa como un sistema experto capaz de discernir fallos cuyas firmas de presión y temperatura son extremadamente similares. En ambos casos, el sistema resultante no sólo es preciso, sino físicamente coherente.
La combinación del modelo Random Forest con el post-procesado avanzado permite que el sistema no solo cumpla, sino que supere las expectativas de fiabilidad industrial. Mientras que el sistema de aireado ofrece una precisión absoluta, el sistema de refrigeración logra filtrar las fluctuaciones naturales del ciclo de compresión, entregando alertas tempranas con una fiabilidad superior al 94%.
# 7 Despliegue e Inferencia del Modelo
## 7.1 Modo de despliegue
El sistema está diseñado para una ejecución local mediante scripts de Python modulares. La interacción se centraliza en una interfaz de línea de comandos (CLI) accesible a través del módulo maestro src.main. En esta ejecución sólo se han incluido los modelos de Random Forest con post procesamiento neurosimbólico, al ser seleccionados por generar unas mejores métricas.
Este diseño permite que el sistema opere como un proceso batch bajo demanda o programado. Al estar basado en una arquitectura de configuración centralizada (config.yaml), se eliminan las dependencias de APIs persistentes, minimizando el coste de operación y mantenimiento.
## 7.2 Mecanismo de inferencia
Para garantizar la integridad de las predicciones, la lógica de inferencia está encapsulada de modo que los datos nuevos reciban exactamente el mismo tratamiento que los de entrenamiento sin intervención manual. El flujo sigue estos pasos:
Carga y Configuración: El sistema consulta el archivo config/config.yaml para determinar el sistema activo (selected_system) y las rutas globales de datos y artefactos.
Tratamiento de Datos: Lectura de ficheros CSV (data/to_predict/input_{system}.csv) y aplicación de limpieza (reordenación temporal por run_id y time_min).
Carga de Artefactos Autogestionada: El sistema extrae de la ruta definida en paths: artifacts los siguientes elementos:
{system}_model.pkl: El modelo predictivo serializado (Random Forest).
{system}_scaler.pkl: Objeto de normalización con las estadísticas congeladas del entrenamiento (específicamente para el pipeline de refrigeración).
Predicción y Refinamiento Neuro-simbólico:
Alineación dinámica de features (el modelo solo selecciona las columnas necesarias ignorando el resto).
Ejecución de la inferencia ML.
Refinamiento mediante reglas de experto y voto por mayoría (estabilidad temporal) para consolidar un único diagnóstico por ciclo.
## 7.3 Interfaz de Línea de Comandos (CLI)
El acceso a las funciones de predicción se realiza de forma simplificada. El usuario no necesita especificar rutas largas, ya que el sistema las hereda del archivo de configuración:
Bash
python -m src.main predict
El comando lee el parámetro selected_system del config.yaml y cargará automáticamente los pesos y reglas de ese sistema. Después, el sistema busca el set de datos en la ruta splits_data (para obtenerlo habrá que haber utilizado antes el comando train) y deposita los resultados en predictions_data, ambas carpetas definidas en el bloque global de rutas del YAML.
## 7.4 Formato de Entradas y Salidas
Entradas para Inferencia:
Formato: Archivo CSV con columnas de sensores.
Metadatos: Presencia de la columna run_id para permitir el agrupamiento por ciclo de trabajo.
Requisito de histórico: Se recomienda un histórico suficiente por ciclo para que las variables de tendencia y medias móviles se estabilicen antes de la toma de decisión final (100 minutos en el sistema de refrigeración, 60 minutos en el de aireado).
Salidas de Inferencia:
Formato: Archivo CSV limpio (unificado por ciclo).
Ubicación: Carpeta definida en paths: predictions_data (por defecto data/predictions/).
Contenido Optimizado: A diferencia de los datos de entrenamiento, el CSV de salida está simplificado para el cliente. Se eliminan todas las variables físicas de entrada, exportando únicamente:
run_id: Identificador del ciclo.
prediction: Etiqueta del fallo detectado
confidence: Probabilidad media de acierto calculada por el modelo durante ese ciclo.
fault_id: (Opcional) Solo se incluyen si los datos de entrada estaban etiquetados, permitiendo auditorías de rendimiento posteriores con la función evaluate.
## 7.5 Protocolo de calibración de los modelos
Para mitigar la posible brecha entre los conjuntos de datos sintéticos y la realidad operativa de cada planta, el sistema dispone de un protocolo de calibración diseñado para adaptar los módulos a las condiciones específicas de cada instalación sin alterar el conocimiento experto inicial. El modelo original se mantiene congelado y sobre este se acopla una capa de calibración estadística basada en el componente CalibratedClassifierCV mediante una regresión isotónica. Este enfoque descarta el riesgo de sobreajuste ante muestras escasas y preserva la integridad del estimador original. La función exacta de este clasificador calibrado consiste en actuar como una capa analítica que toma las puntuaciones brutas emitidas por el modelo base y las ajusta mediante una función monótona creciente, suavizando y corrigiendo las distribuciones de probabilidad en la frontera de decisión para adecuarlas al comportamiento estadístico real del nuevo entorno.
Paralelamente, los umbrales de las reglas físicas del post-proceso neurosimbólico y las distribuciones estadísticas del sistema guardadas operan de forma dinámica, desplazándose automáticamente a partir del cálculo de cuantiles sobre el lote de datos recolectado en la planta, en el caso de los umbrales, y guardándose de forma persistente con nuevos versionado. El protocolo integra un bloque de control basado en volumen crítico que evalúa el tamaño del conjunto de datos: si las muestras son insuficientes para estabilizar la regresión del clasificador, las intensidades probabilísticas del modelo matemático se bloquean de forma segura y la calibración se restringe exclusivamente al reajuste de los umbrales físicos del motor simbólico.
En concreto, mediante el comando especializado calibrate (ver Sección 8.2), el sistema procesa los archivos de datos de la planta (que deben ser colocados en data/raw/real_data_{system}.csv), ejecuta la ingeniería de variables correspondiente y consolida la calibración bajo un esquema de versionado atómico no volátil ligado a timestamps. Este proceso genera paquetes cerrados y paralelos que contienen el clasificador calibrado, las estadísticas actualizadas y los nuevos umbrales mecánicos (ver Sección 8.1), permitiendo la plena trazabilidad de los artefactos y la capacidad de realizar un rollback inmediato a la configuración base original de fábrica. Todo el procedimiento mantiene además un registro automatizado de data drift y métricas de salud del modelo en el histórico ubicado en logs/monitorization_{system}.csv, facilitando a los operadores la monitorización proactiva y garantizando la validez técnica, la seguridad y la robustez del sistema a largo plazo. En inferencia, habrá la posibilidad de realizarla con el modelo inicial o con cualquiera de los modelos calibrados.
# 8 Reproducibilidad y Evidencias
El sistema ha sido diseñado de forma que para cambiar entre sistemas o ajustar el rendimiento, no es necesario modificar los algoritmos, sino interactuar con el archivo maestro de configuración (config.yaml).
## 8.1 Inventario de Artefactos y Entorno
Al completar el pipeline entero, el inventario debe contener los siguientes elementos organizados por el sistema de rutas de config.yaml:
Configuración Maestra (config/):
config.yaml: Archivo central donde se define el selected_system ("refrigeracion" o "aireado"), los hiperparámetros de entrenamiento de cada modelo, las columnas a ignorar (drop_cols) y las rutas globales.
Datasets en crudo  (data/raw/):
Archivos CSV sobre los que se va a realizar el preprocesamiento.
dataset_refrigeracion.csv (sin ninguna modificación con respecto al Simulated Refrigerator Fault Diagnosis Dataset, excepto el nombre). Debe descargarse directamente desde su fuente oficial en Kaggle (y renombrarlo a dataset_refrigeracion.csv) o desde el contenedor externo del proyecto.
dataset_aireado.csv (dataset de origen sintético diseñado específicamente para simular procesos de curado cárnico). Puede obtenerse mediante dos vías: la descarga directa desde el contenedor externo del proyecto o a través de su generación local ejecutando el script scripts/generate_dataset_aireado.py. En este último caso, el comando generará el archivo dataset_aireado.csv en la carpeta data/raw/, listo para que el orquestador del sistema pueda reconocerlo correctamente.
real_data_{system}.csv: Archivos obligatorios para ejecutar la rutina calibrate (ver Sección 8.2). Deben contener datos reales de planta (etiquetados) para realizar el fine-tuning del modelo y la actualización de umbrales dinámicos neurosimbólicos. El sistema buscará estos archivos en data/raw/ para poder calcular la deriva y adaptar el modelo a las condiciones operativas reales.
Nota: El sistema asume que los datasets de ambos sistemas son fuentes íntegras y sin modificaciones externas. Cualquier ausencia de estos archivos o error en el nombre impedirá la ejecución de los comandos de preprocesamiento, entrenamiento, etc. En el caso de los archivos real_data_{system}.csv, se han dejado unos archivos a modo de ejemplo para que se visualice la estructura (exactamente igual a los datasets de cada uno de los sistemas), pero si se quisiese recalibrar el modelo, se deberían de implementar nuevos archivos con los datos reales pertinentes.
Artefactos Generados (models/artifacts/):
Tras completar el ciclo completo (con el sistema específico seleccionado), se deben encontrar:
Sistema Refrigeración:
refrigeracion_model.pkl: Pesos del Random Forest optimizado.
refrigeracion_scaler.pkl: Parámetros de normalización (StandardScaler).
refrigeracion_best_params.pkl/.yaml: Registro de parámetros óptimos hallados en el tuning.
refrigeracion_thresholds.yaml: Umbrales neurosimbólicos de referencia para el sistema de refrigeración, actualizados dinámicamente mediante el comando calibrate (ver Sección 8.2).
refrigeracion_stats.yaml: Perfil estadístico de referencia (línea base) para la detección de drift. Generado con el comando calibrate (ver Sección 8.2).
refrigeracion_thresholds_calibrated_[timestamp].yaml: Registro de umbrales físicos calculados dinámicamente mediante cuantiles sobre el lote de datos reales ingresado durante el proceso de calibración (ver comando calibrate en Sección 8.2).
refrigeracion_stats_calibrated_[timestamp].yaml: Histórico de medias y desviaciones estándar de las variables derivadas del nuevo lote de producción, acoplado al identificador temporal del modelo calibrado para garantizar la trazabilidad.
Sistema Aireado:
aireado_model.pkl: Modelo entrenado (sin escalado).
aireado_best_params.pkl/.yaml: Registro de parámetros óptimos para aireado.
aireado_thresholds.yaml: Umbrales neurosimbólicos dinámicos procesados mediante el comando calibrate para el sistema de aireado. Generado con el comando calibrate (ver Sección 8.2).
aireado_stats.yaml: Perfil estadístico de referencia (línea base) para la detección de drift del sistema de aireado. Generado con el comando calibrate (ver Sección 8.2).
aireado_thresholds_calibrated_[timestamp].yaml: Umbrales físicos dinámicos recalculados a partir del comportamiento real de los ventiladores y los riesgos de encostramiento térmico del lote específico de calibración (ver Sección 8.2).
aireado_stats_calibrated_[timestamp].yaml: Registro específico de los momentos estadísticos calculados para las variables del sistema de aireado durante la ejecución del comando calibrate (ver Sección 8.2).
Nota: debido a su tamaño, refrigeracion_model.pkl deberá de descargarse desde el contenedor externo del proyecto y colocarlo en la ruta correspondiente (models/artifacts/).
Evidencias de Datos y Splits (data/splits/):
Archivos CSV resultantes de la partición estratificada por ciclos, generados tras ejecutar el comando train con el sistema específico seleccionado (ver Sección 8.2) y con ingeniería de variables realizada:
refrigeracion_train.csv / refrigeracion_test.csv
aireado_train.csv / aireado_test.csv
Esto es importante porque la inferencia se realizará directamente sobre los archivos de test situados en esta ruta (dependiendo del sistema seleccionado). Estos archivos también pueden ser descargados desde el contenedor externo del proyecto (y colocados en la ruta correspondiente) si no se desea ejecutar el pipeline de entrenamiento.
### Datos de Entrada para Inferencia (data/to_predict/)
Archivos que representan la "entrada real” dada por el cliente. Estos archivos deben contener datos crudos (sensores directos) sin procesar.
input_refrigeracion.csv / input_aireado.csv
Estos archivos no requieren las columnas de objetivo (fault_id, fault), permitiendo realizar diagnósticos sobre datos nuevos. Al detectar un archivo en esta carpeta, el comando predict activa automáticamente el pipeline de ingeniería de variables (VPD, Lags, indicadores físicos) antes de lanzar la inferencia, replicando el comportamiento del modelo en un entorno de producción real. Si no existiesen estos archivos en la ruta correspondiente, el sistema automáticamente realiza inferencia con los archivos {system}_test.csv localizados en la carpeta data/splits/. Estos archivos también pueden ser descargados desde el contenedor externo del proyecto (y colocados en la ruta correspondiente).
Reportes de Rendimiento (models/metrics/):
Archivos generados tras ejecutar el comando evaluate con el sistema específico seleccionado (ver Sección 8.2):
confusion_matrix_{system}.png: Visualización de errores Tipo I y Tipo II.
classification_report_{system}.csv: Métricas de F1-Score, Precisión y Recall por cada etiqueta de fallo.
### Trazabilidad de Operaciones y Salud (logs/):
pipeline_mantenimiento.log: Registro cronológico de cada evento, advertencia o error del pipeline.
monitorization_{system}.csv: Archivo unificado que registra el Data Drift (tras ejecutar el comando calibrate, ver Sección 8.2) y la Salud/Confianza del modelo (tras ejecutar predict). Este archivo debe ser utilizado para detectar degradación y tomar decisiones de reentrenamiento, centralizando la observabilidad del sistema.
Nota: Se han incluido archivos .gitkeep en las carpetas de datos, modelos y logs para preservar la estructura del árbol de directorios en el control de versiones, cumpliendo con las reglas de exclusión del .gitignore que protegen la privacidad de los datos industriales y el peso del repositorio.
## 8.2 Pasos para Reproducir el Ciclo Completo
Para reproducir el experimento, siga estrictamente este flujo de trabajo, no sin antes asegurarse de que los archivos del contenedor externo del proyecto están situados en las rutas correspondientes (datasets en data/raw/, splits en data/splits/ y el modelo de refrigeración, refrigeracion_model.pkl, en models/artifacts/) :
Paso 0(A): Configuración del entorno
Para recrear el entorno de trabajo, se recomienda el uso de un entorno virtual ejecutando el siguiente flujo (para Windows):
# Instalación del entorno
python -m venv venv
# Activación del entorno
venv\Scripts\activate
pip install -r requirements.txt
En el caso de estar trabajando con Linux/Mac la activación del entorno se hará mediante el comando source venv/bin/activate.
Paso 0(B): Selección del Sistema
Abra config/config.yaml y establezca el parámetro selected_system. Cambiar entre “refrigeracion” o “aireado” según corresponda:
YAML
selected_system: "refrigeracion"	#"refigeracion" u "aireado"

Si se quisiera generar de nuevo el dataset tabular para el modelo de aireado (incluido en el contenedor externo del proyecto en data/raw/) se deberían de utilizar el siguiente comando:
Bash
python scripts/generate_dataset_aireado.py	   #Crea dataset_aireado.csv

Si se quisiera extraer información estadística sobre el dataset del sistema seleccionado (incluido un análisis de data leakage), el comando utilizado sería:
Bash
python -m src.main get_stats
Para obtener información sobre el dataset del sistema utilizado, con descripciones de variables crudas y utilizadas en ingeniería de variables, así como una breve descripción del modelo, el comando utilizado sería:
Bash
python -m src.main get_info

Paso 1: Procesamiento y Limpieza
Ejecute la ingesta de datos crudos. El sistema detectará automáticamente si debe leer dataset_refrigeracion.csv o dataset_aireado.csv en data/raw/.
Bash
python -m src.main data_processing
Al ejecutar este comando se generarán archivos .csv de datasets procesados en data/processed con la ingeniería de variables realizada.
Paso 2: Tuning de Hiperparámetros (opcional)
Inicie la búsqueda aleatoria basada en las estrategias de validación cruzada definidas (StratifiedGroupKFold para refrigeración o GroupKFold para aireado).
Bash
python -m src.main tuning
Se generará un archivo .pkl y un .yaml con los mejores parámetros en models/artifacts/.
Paso 3: Entrenamiento del modelo
El comando train cargará los datos de entrenamiento procesados y los parámetros del YAML para generar el modelo. Para cambiar los hiperparámetros de entrenamiento para cada uno de los modelos, se hará desde el archivo config.yaml. Por defecto estos hiperparámetros están ajustados a los que generaron mejores métricas (mejor F1-macro) después del tuning realizado.
Bash
python -m src.main train

Paso 4: Calibración (opcional)
En este paso, el pipeline ejecuta el protocolo de calibración estadística y adaptabilidad física para adecuar los módulos a las condiciones operativas reales de la planta. A través de este proceso, el sistema analiza el archivo de datos de producción (data/raw/real_data_{system}.csv) para calcular y corregir las distribuciones de probabilidad mediante regresión isotónica sin alterar el estimador base, registrar la deriva estadística en los históricos de monitorización y actualizar dinámicamente los umbrales mecánicos del post-proceso neurosimbólico.
Bash
python -m src.main calibrate

Nota: Este comando sólo se puede/debe ejecutar si se disponen de los archivos data/raw/real_data_{system}.csv específicos para el sistema seleccionado en el config.yaml. La ausencia de estos datos impedirá el cálculo de la deriva y la actualización de los umbrales dinámicos, resultando en una interrupción del proceso de calibración.
Paso 5: Inferencia y reglas neurosimbólicas
Este paso carga el set de test (datos no vistos). En los dos modelos, es también donde se aplica la lógica física de post-procesado para corregir las confusiones neurosimbólicas. Se podrá ejecutar la función predict sin necesidad de tener un dataset supervisado (sin columna ‘fault_id’).
Bash
python -m src.main predict
Nota sobre la Inferencia: El comando es flexible y permite dos modos de uso según la ubicación de los archivos:
Modo Producción: Si existe un archivo en data/to_predict/input_{sistema}.csv, el sistema realizará la ingeniería de variables automáticamente. No requiere supervisión (puede ir sin las columnas target fault_id o fault).
Modo Evaluación: Si no detecta entrada de archivo input del cliente, usará por defecto el set de test en data/splits/.
Archivos input de ejemplo han sido colocados en el contenedor externo del proyecto como referencia en la ruta data/to_predict/. Las predicciones finales se guardan en data/predictions/predictions_{sistema}.csv, incluyendo la clase detectada, la confianza y el estado de salud del modelo.
Paso 6: Evaluación (opcional)
Finalmente, se podrán generar las métricas de validación para asegurar que el modelo cumple con los requisitos de negocio. La función evaluate sólo funcionará en caso de utilizar un dataset supervisado.
Bash
python -m src.main evaluate
Esta función generará un reporte de clasificación que se podrá ver desde la consola, a la vez que será guardado, junto a una imagen de la matriz de confusión, en la carpeta models/metrics.
Para asegurar resultados idénticos, se debe utilizar la semilla random_state = 42 presente en el config.yaml.
## 8.3 Trazabilidad del Código
La arquitectura del software está diseñada bajo un principio de modularidad técnica dentro del directorio src/, facilitando la auditoría forense de cada etapa del pipeline:
Orquestador principal (src/main.py): Centraliza y coordina los comandos de ejecución del sistema (preprocesamiento, entrenamiento, tuning, predicción, calibración y evaluación), garantizando que todos los módulos utilicen la misma configuración global.
Gestión de datos (src/data_processing/): Módulos responsables de la ingeniería de características.
Generación de splits y lógica de entrenamiento y tuning (src/training/): Contiene los scripts de optimización de hiperparámetros y el entrenamiento de los modelos Random Forest para cada sistema. Además, se genera la partición técnica de datos por ciclos (run_id).
Motor de Inferencia y Post-proceso (src/predict/): Encapsula la carga de artefactos y la ejecución de la arquitectura de tres niveles (ML + Reglas Neuro-simbólicas + Voto Temporal).
## 8.4 Evidencias de Resultados
Tras la ejecución del pipeline de evaluación (evaluate) y predicción (predict), el sistema genera automáticamente evidencias documentales en las carpetas de salida definidas en el archivo de configuración:
Reporte de Clasificación (models/metrics/classification_report_{system}.txt): Archivo de texto que detalla las métricas oficiales (Precisión, Recall, F1-Score) por cada tipo de fallo. Estos valores constituyen la evidencia técnica del cumplimiento de los criterios de éxito.
Matriz de Confusión (models/metrics/confusion_matrix_{system}.png): Representación visual que permite auditar las interacciones entre clases y validar la capacidad discriminativa del modelo ante fallos similares.
Predicciones Consolidadas (data/predictions/predictions_{system}.csv): Archivo CSV que contiene el diagnóstico final por ciclo (run_id). Incluye la predicción realizada y el nivel de confianza asociado, permitiendo una trazabilidad completa de las decisiones del modelo para su revisión por parte de los responsables de mantenimiento.
Artefactos de Configuración (models/artifacts/{system}_best_params.yaml): Registro legible de los hiperparámetros óptimos seleccionados por el sistema, garantizando la transparencia sobre la configuración interna de los modelos de producción.
# 9 Limitaciones, Riesgos y Consideraciones
Aunque se han conseguido unas métricas excelentes tanto en el modelo de aireado como en el de refrigeración, los modelos incluidos en este estudio presentan varias limitaciones. La primera y más obvia es que los escenarios de entrenamiento, si bien son precisos, podrían no representar toda la complejidad de una planta industrial en condiciones extremas de operación continua durante años. Esta falta de datos reales a tiempos largos hace que los modelos no puedan detectar fenómenos de degradación mecánica lenta o fallos multivariantes que solo emergen tras largos periodos de explotación.
Con respecto al modelo de refrigeración, la incorporación de variables de lags y deltas introduce restricciones operativas para su despliegue en tiempo real. El modelo requiere un histórico mínimo de hasta 100 minutos para calcular tendencias de largo alcance y derivas térmicas lentas. Debido a esto se generan "puntos ciegos" iniciales durante los primeros tramos de cada ciclo operativo donde la capacidad de diagnóstico es limitada o nula por la presencia de valores vacíos. En el modelo de aireado, se generan limitaciones operativas similares, causadas por la necesidad de un histórico de datos para alcanzar fiabilidad diagnóstica. En este caso, se requiere un histórico de 60 minutos para llegar a una estabilización inicial en variables críticas como la humedad relativa (RH_cab) y el índice de eficiencia de evaporación antes de poder calcular con precisión las tendencias de secado y los deltas de cambio.
Así mismo, debe considerarse que el dataset utilizado para refrigeración proviene de un refrigerador de propósito general y no de un sistema diseñado específicamente para la conservación de embutidos. En un entorno cárnico real, existen especificaciones críticas adicionales que no están representadas en este modelo base. Esta diferencia implica que el sistema actual debe interpretarse como un marco metodológico que requiere una fase de calibración específica para los parámetros térmicos y de humedad propios de los secaderos industriales.
Los resultados de la validación cruzada en el sistema de aireado demuestran que el sistema es extremadamente robusto en entornos controlados, alcanzando métricas perfectas de 1,00 en el modelo de aireado. No obstante, es imperativo señalar que estos resultados se han obtenido sobre un dataset de naturaleza sintética para el aireado y simulada para la refrigeración. Aunque esto permite disponer de un gran volumen de muestras y un etiquetado preciso, existe el riesgo de que el modelo no esté totalmente expuesto a las anomalías aleatorias y derivas sensoriales propias de una instalación física real.
En el modelo de refrigeración, la simulación ha revelado que las clases COND_FOUL_SEVERE (suciedad severa en el condensador) y NON_CONDENSABLES (presencia de gases no condensables) son fácilmente confundibles entre sí debido a que sus firmas térmicas y de presión son casi idénticas. Es precisamente en este punto donde se ha centrado gran parte del análisis neurosimbólico, logrando un F1-score de 0,70 en el mejor de los casos para estas clases, lo que subraya la dificultad técnica de discernir entre fallos con efectos físicos solapados.
Para mitigar la variabilidad en producción, el sistema implementa monitorización continua mediante métricas rolling que evalúan la salud del modelo en tiempo real. Este mecanismo detecta el Concept Drift (degradación por cambios en el entorno o sensores) y ajusta la confianza del diagnóstico según la estabilidad del sistema. Como mejora futura, se recomienda enriquecer el dataset con variables exógenas (consumo eléctrico y desgaste mecánico) para reducir el ruido térmico y aumentar la capacidad de discriminar entre fallos con firmas físicas solapadas, como los gases no condensables y la suciedad severa.
Aunque el proyecto se ha centrado en los sistemas de aireado y refrigeración de embutidos, el modelo debe entenderse como un demostrador técnico de la metodología neurosimbólica. La arquitectura es plenamente extensible a otros segmentos de la cadena de frío, como túneles de congelación o cámaras de conservación, siempre que se disponga de series históricas con la misma trazabilidad y calidad de etiquetado que las utilizadas en este piloto.
Dado el volumen de datos disponible (aunque sean simulados), modelos como el Random Forest pueden asignar probabilidades elevadas en regiones poco soportadas por datos reales, con riesgo de una generalización deficiente ante configuraciones de sensores poco frecuentes o fuera de los límites de la simulación.
La rama Neurosimbólica (LTN) se mantiene como un plano exploratorio de alta fidelidad lógica, aportando una capa de coherencia física que el aprendizaje puramente estadístico podría ignorar, mientras que el pipeline de producción utiliza el ensamble de árboles por su interpretabilidad y menor riesgo de degradación.
Las predicciones deben interpretarse como una aproximación útil bajo las condiciones de diseño, adecuadas para reducir paradas no programadas y optimizar el mantenimiento. Antes de plantear arquitecturas de Deep Learning más complejas, es prioritario ampliar las series con datos de campo reales, lo que permitiría aumentar de forma sustancial la robustez de las predicciones en escenarios extraordinarios no contemplados en las simulaciones iniciales.
# Referencias
[1] A. Andrés, S. Ventanas, J. Ruiz, et al., "Physicochemical changes throughout the ripening of dry cured hams with different salt content and processing conditions," European Food Research and Technology, vol. 221, pp. 30–35, 2005.
[2] F. Toldrá and M. C. Aristoy, "Dry-Cured Ham," in Handbook of Meat Processing, F. Toldrá, Ed. Oxford, UK: Wiley-Blackwell, pp. 351–362, 2010.
[3] W. F. Stoecker and J. W. Jones, Refrigeration and Air Conditioning. McGraw-Hill, 1982.
[4] R. J. Dossat, Principios de Refrigeración. Pearson Educación, 2001.
[5] A. C. Cleland, Food Refrigeration: Processes, Analysis, Design and Simulation. Elsevier Applied Science, 1990.
[6] J. Ruiz-Ramírez, J. Arnau, X. Serra, and P. Gou, "Relationship between water content, NaCl content, pH and texture parameters in dry-cured muscles," Meat Science, vol. 70, no. 4, pp. 579–587, 2005.
[7] L. Imre, "Solar Drying," in Handbook of Industrial Drying, 4th ed., A. S. Mujumdar, Ed. CRC Press, pp. 308–363, 2014.
[8] M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," Journal of Computational Physics, vol. 378, pp. 686–707, 2019.
[9] S. Badreddine, A. d'Avila Garcez, L. Serafini, and M. Spranger, "Logic Tensor Networks," Artificial Intelligence, vol. 303, p. 103649, 2022.
[10] U. Nawaz, M. Anees-ur-Rahaman, and Z. Saeed, "A review of neuro-symbolic AI integrating reasoning and learning for advanced cognitive systems," Intelligent Systems with Applications, vol. 26, p. 200541, 2025.
[11] A. d’Avila Garcez and L. C. Lamb, "Neurosymbolic AI: the 3rd wave," Artificial Intelligence Review, vol. 56, pp. 12387–12406, 2023.
[12] U. Nawaz, M. Anees-ur-Rahaman, and Z. Saeed, "A review of neuro-symbolic AI integrating reasoning and learning for advanced cognitive systems," Intelligent Systems with Applications, vol. 26, p. 200541, 2025.
[13] L. Serafini and A. d'Avila Garcez, "Logic tensor networks: Deep learning and logical reasoning from data and knowledge," arXiv preprint arXiv:1606.04422, 2016.
[14] T. Carraro, L. Serafini, and F. Aiolli, "LTNtorch: PyTorch implementation of Logic Tensor Networks," arXiv preprint arXiv:2409.16045, 2024.
[15] ASHRAE, Refrigeration Handbook. Atlanta, GA: ASHRAE, 2022.
[16] Z. Soltani, K. K. Sørensen, J. Leth, and J. D. Bendtsen, "Fault detection and diagnosis in refrigeration systems using machine learning algorithms," International Journal of Refrigeration, vol. 144, pp. 34-45, 2022.

[TOTAL w:tbl: 6]

=== TBL 0 (8 rows) ===
Título - Proyecto | Modelo de DNSL para la predicción de fallas inminentes en equipos de refrigeraci
Acrónimo | DNSL-EMBUTIDOS
Organismo | AIR Institute
Lote | 2
Entregable | 
Realizado por | Equipo de ciencia de datos del AIR Institute
Versión | 3.0
Fecha versión | 01/07/2026

=== TBL 1 (9 rows) ===
REGISTRO DE CAMBIOS | REGISTRO DE CAMBIOS | REGISTRO DE CAMBIOS | REGISTRO DE CAMBIOS
Versión | Cambio realizado | Responsable | Fecha
1 | Versión inicial | Equipo de ciencia de datos del AIR Institute | 19/03/2026
1.1 | Corrección de fallos tipográficos e introducción de matrices de confusión | Equipo de ciencia de datos del AIR Institute | 13/04/2026
1.2 | Implementación de explicación más detalladas sobre arquitecturas y tiempo requer | Equipo de ciencia de datos del AIR Institute | 06/05/2026
1.3 | Implementación de pipeline de ruido en ambos sistemas | Equipo de ciencia de datos del AIR Institute | 27/05/2026
2.0 | Correcciones implementadas con respecto a la auditoría | Equipo de ciencia de datos del AIR Institute | 16/06/2026
2.1 | Correcciones implementadas con respecto a la auditoría | Equipo de ciencia de datos del AIR Institute | 17/06/2026
3.0 | Correcciones implementadas con respecto a la auditoría | Equipo de ciencia de datos del AIR Institute | 01/07/2026

=== TBL 2 (9 rows) ===
Abreviatura | Descripción
LTN | Logic Tensor Network
DNSL | Deep Neurosimbolic Learning
COP | Coeficiente de Rendimiento
RF | Random Forest
IA | Inteligencia Artificial
EDA | Análisis Exploratorio de Datos
EEI | Energy Efficiency Index
RH | Delta Higroscópico

=== TBL 3 (11 rows) ===
Fase | Paso | Descripción
1 | Ingesta de Datos | Extracción de registros históricos desde archivos .csv con variables de presión 
2 | Mapeo | Verificación de integridad de señales y mapeo de la variable objetivo fault a va
3 | Ingeniería de características | Creación de variables sintéticas: P_ratio, T_lift, EEI, cálculo de T_cond_approa
4 | Limpieza | Eliminación de variables redundantes por multicolinealidad (T_cond_sat, P_suc_ba
5 | Ventaneo Temporal | Generación de lags y deltas para capturar la inercia térmica y la estabilidad de
6 | Particionado por run_id | División del dataset en Entrenamiento (80%) y Test (20%) mediante un Split Estra
7 | Normalización (si existiese) | Aplicación de un StandardScaler ajustado únicamente sobre el conjunto de entrena
8 | Optimización (Tuning) | Búsqueda de hiperparámetros mediante RandomizedSearchCV utilizando GroupKFold pa
9 | Capa neurosimbólica (si existiese) | Fase de post-procesamiento basada en reglas físicas para refinar la clasificació
10 | Voto por run (si existiese) | Consolidación de la predicción final mediante un consenso mayoritario de todos l

=== TBL 4 (6 rows) ===
Proceso | Duración medida | Hardware (CPU • RAM)
Entrenamiento modelo aireado neurosimbolico | < 1 min con tuning < 5 s  con solo train | Intel Core i9-14900K (24 núcleos, 32 hilos) • 64 GB RAM
Entrenamiento modelo aireado LTN | 4.6 s (200 epochs) | Intel Core i9-14900K (24 núcleos, 32 hilos) • 64 GB RAM
Entrenamiento modelo refrigeración neurosimbólico | < 35 min  con tuning.  < 2 min con solo train | Intel Core i9-14900K (24 núcleos, 32 hilos) • 64 GB RAM
Entrenamiento modelo refrigeración LTN | <19 min (600 epochs) | Intel Core i9-14900K (24 núcleos, 32 hilos) • 64 GB RAM
Inferencia (en todos los modelos) | < 5s | Intel Core i9-14900K (24 núcleos, 32 hilos) • 64 GB RAM

=== TBL 5 (17 rows) ===
Sistema | Modelo | Postprocesado | Acc. (avg) | Precision (avg) | Recall (avg) | F1-Macro
Aireado | Random Forest | Puro (30% ruido) | 0,84 | 0,86 | 0,84 | 0,84
Aireado | Random Forest | Puro | 0,97 | 0,97 | 0,97 | 0,97
Aireado | Random Forest | + NS (30% ruido) | 0,84 | 0,85 | 0,84 | 0,84
Aireado | Random Forest | + NS | 0,97 | 0,97 | 0,97 | 0,97
Aireado | Random Forest | + NS + Voto por Run (30% ruido) | 1,00 | 1,00 | 1,00 | 1,00
Aireado | Random Forest | + NS + Voto por Run | 1,00 | 1,00 | 1,00 | 1,00
Aireado | LTN | Configuración final | 1,00 | 1,00 | 1,00 | 1,00
Refrigeración | Random Forest | Puro (10% ruido) | 0,88 | 0,87 | 0,88 | 0,87
Refrigeración | Random Forest | Puro | 0,90 | 0,90 | 0,90 | 0,90
Refrigeración | Random Forest | + NS (10% ruido) | 0,90 | 0,90 | 0,90 | 0,90
Refrigeración | Random Forest | + Neurosimbólico | 0,94 | 0,94 | 0,94 | 0,94
Refrigeración | Random Forest | + NS. + Voto (10% ruido) | 0,95 | 0,95 | 0,95 | 0,95
Refrigeración | Random Forest | + Neuro. + Voto | 0,94 | 0,94 | 0,94 | 0,94
Refrigeración | LTN | Configuración final | 0,91 | 0,91 | 0,92 | 0,91
Refrigeración | LTN | + Neurosimbólico | 0,91 | 0,91 | 0,92 | 0,91
Refrigeración | LTN | + Neuro. + Voto | 0,92 | 0,91 | 0,92 | 0,92