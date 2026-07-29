# Mapeo de contenido: ficha bruta de la IA → ficha con formato DatagIA

Cuando se integra un modelo, la IA (Claude Code / el agente de integración)
genera una ficha técnica y una ficha funcional "en bruto": Markdown/docx con
prosa larga, sin el formato corporativo. Esta guía explica cómo condensar
ese contenido en el esquema de `reference/esquema_datos.md` para que el
resultado final tenga el mismo formato para todos los modelos (ver
`assets/ficha_tecnica_template.docx` / `ficha_funcional_template.docx` y los
ejemplos ya publicados, p. ej. el modelo `a25 · wine-sulphite`).

**Regla general: esto no es un find-and-replace.** Las fichas en bruto suelen
tener 3-6 páginas de prosa (arquitectura, hiperparámetros, verificación
post-integración, requisitos de hardware...). La ficha final tiene un
número fijo de secciones/tablas. Hay que **leer, entender y resumir**, no
copiar párrafos enteros.

## Ficha técnica: de dónde sale cada campo

| Campo del esquema | Búscalo en la ficha bruta en... |
|---|---|
| `model_key`, `version` | Cabecera del documento o metadatos del modelo (`plantilla_metadatos_modelo_v3` si existe) |
| `descripcion` | Resumen de 1-2 frases de la sección "Arquitectura" o del título; qué es el modelo (tipo de red/algoritmo + para qué) |
| `task_type` | Se infiere: regresión, clasificación, optimización... |
| `framework` | Sección de arquitectura (PyTorch, scikit-learn, pygad, etc.) |
| `artefacto` | Nombre de fichero(s) .pkl/.pt mencionados |
| `objetivo` | Sección "Objetivo y KPI" / "Descripción del Problema de Negocio" de la ficha funcional en bruto, reescrito en 2-3 frases técnicas |
| `caso_uso` | Sección "Modos de Uso" (predict/optimize/batch), resumida a un párrafo |
| `inputs` | Tabla "Variables de entrada" / "Entradas y Salidas" — copiar nombre/tipo/descripción tal cual, son datos exactos |
| `outputs` | Tabla de salidas del modo principal (normalmente "predict" o "inline") — añadir `rango` e `interpretacion` si la ficha bruta los da; si no, inferir un rango razonable a partir de las métricas y anotar la interpretación en una frase |
| `metricas` | Sección "Métricas de Rendimiento" (usar las métricas declaradas en el hold-out test, no la verificación post-integración caso a caso) — la `interpretacion` es una frase en lenguaje llano de qué significa ese valor |
| `limitaciones` | Puede no existir como sección explícita en la ficha bruta: derivar de las restricciones de dominio, el alcance de los datos de entrenamiento, y cualquier aviso (⚠) presente en el documento |
| `fichero_nombre` / `fichero_registros` / `fichero_resultado` | Sección de "Verificación post-integración" / golden dataset — resume el fichero de prueba usado y qué se espera al ejecutarlo |
| `observaciones_tecnicas` | Sección "Requisitos de Hardware y Entorno" — condensar a un párrafo corto (CPU/GPU, versiones de librerías, tamaño de artefactos, recomendaciones de monitorización) |

**Tono de la ficha técnica:** preciso, denso, para un ingeniero. Usa
terminología técnica sin explicarla. Frases cortas, con cifras y unidades.
No repitas aquí "vulgarizaciones" de negocio — eso va en la funcional.

## Ficha funcional: de dónde sale cada campo

| Campo del esquema | Búscalo en la ficha bruta en... |
|---|---|
| `descripcion` / `objetivo` / `caso_uso` | Sección "Descripción del Problema de Negocio" / "Objetivo y KPI" de la ficha funcional en bruto — reescribir en lenguaje de negocio, sin nombres de variables ni jerga (nada de "MLP", "ReLU", "RandomForestRegressor"...) |
| `que_no_responde` | Suele no estar explícito: inferir de las limitaciones técnicas qué pregunta NO contesta el modelo (p. ej. "no da un precio garantizado, da una recomendación") |
| `entradas_contexto` | Frase corta: de dónde vienen los datos que pide el modelo (sensor IIoT, analítica de laboratorio, formulario del usuario...) |
| `inputs` | Misma lista que en la técnica pero **sin** columna de tipo/obligatoriedad — solo nombre + descripción en lenguaje llano |
| `outputs` | Igual: nombre + descripción llana + cómo interpretar el resultado en la práctica |
| `ejemplo_interpretacion` | Inventar/derivar un ejemplo numérico realista usando los rangos típicos de las métricas ("si el modelo predice X, se interpreta como...") |
| `margen_error` | Traducir el MAE/RMSE técnico a lenguaje llano ("una diferencia inferior a X no es significativa") |
| `revisar_manualmente` | Casos límite mencionados en restricciones de dominio, sesgos del dataset, o rangos de validez — cada uno como una frase-bullet independiente |
| `que_no_hace` | Negaciones explícitas: qué no sustituye, qué no garantiza, qué está fuera de alcance |
| `puede_equivocarse` | Resumir el % de mejora sobre baseline / la incertidumbre del modelo, terminando siempre con que es una ayuda a la decisión, no una garantía |
| `kpis` | Sección "Objetivo y KPI" de la ficha bruta — reformular cada KPI técnico como un umbral entendible por el usuario de negocio |
| `limitaciones` | Mismas limitaciones que en la técnica, reescritas en lenguaje llano (qué puede fallar y cuándo no usar el modelo) |

**Tono de la ficha funcional:** cercano, en segunda persona ("tú"/"introduces"/"obtienes"),
sin jerga de ML. Nunca menciones nombres de librerías, arquitecturas de red,
hiperparámetros o nombres de variables internas. Piensa en el lector como un
técnico de bodega/planta, no un ingeniero de datos.

## Flujo de trabajo recomendado

1. Lee la ficha técnica y funcional en bruto (pandoc o `python-docx`, ver el
   skill `docx` para extracción).
2. Rellena un único JSON por modelo con **todas** las claves de
   `reference/esquema_datos.md` (ambos esquemas combinados). Si dudas sobre
   un dato, vuelve a leer la ficha bruta — no inventes cifras.
3. Genera ambos documentos:
   ```bash
   python3 scripts/generar_ficha.py --tipo tecnica   --datos datos_aXX.json --out aXX_ficha_tecnica_v2.docx
   python3 scripts/generar_ficha.py --tipo funcional --datos datos_aXX.json --out aXX_ficha_funcional_v2.docx
   ```
4. Verifica visualmente (convertir a PDF y mirar cada página — ver skill `docx`,
   sección "Verify the output") antes de entregar.
5. Compara con un ejemplo ya publicado (p. ej. `a25 · wine-sulphite`) para
   confirmar que la estructura de secciones coincide exactamente.
