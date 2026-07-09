# Esquema de datos para las plantillas

Cada plantilla (`ficha_tecnica_template.docx` y `ficha_funcional_template.docx`)
se rellena con `scripts/generar_ficha.py` a partir de un JSON. Puedes usar
un único JSON por modelo con todas las claves (de ambos esquemas) y pasarlo
a los dos `--tipo`; cada plantilla solo usa lo que necesita.

Todos los valores son strings (o listas de objetos). No dejes claves vacías:
si un dato no aplica, escribe `"N/A"` o una frase corta explicando por qué.

## Ficha técnica (`--tipo tecnica`)

Claves de nivel superior:

| Clave | Tipo | Descripción |
|---|---|---|
| `model_id` | string | Identificador corto tipo `a35` (el que usa ITACYL para el modelo) |
| `model_slug` | string | Slug técnico del modelo, ej. `dairy-ann-cleaning-cost` |
| `fecha` | string | Ej. `"Julio 2026"` |
| `model_key` | string | Nombre interno del modelo (model_key real del sistema) |
| `version` | string | Versión del modelo, ej. `"3.2"` |
| `descripcion` | string | 2-4 frases: qué hace el modelo y para qué sirve |
| `task_type` | string | `regression`, `classification`, `optimization`, etc. |
| `framework` | string | Framework/librería principal, ej. `"PyTorch"`, `"scikit-learn (RandomForestRegressor)"` |
| `artefacto` | string | Nombre(s) de fichero de artefacto(s) del modelo |
| `objetivo` | string | Para qué problema de negocio existe el modelo (párrafo) |
| `caso_uso` | string | Cuándo/cómo se usa en la operativa real (párrafo) |
| `inputs` | lista de objetos | ver abajo |
| `outputs` | lista de objetos | ver abajo |
| `metricas` | lista de objetos | ver abajo |
| `limitaciones` | lista de objetos | ver abajo |
| `fichero_nombre` | string | Nombre del fichero de prueba de referencia |
| `fichero_registros` | string | Nº de registros/muestras del fichero de prueba |
| `fichero_resultado` | string | Resultado esperado al ejecutar ese fichero |
| `observaciones_tecnicas` | string | Requisitos hardware/software, librerías, tamaño artefactos, notas de monitorización |

`inputs[]` (Tabla 2 — Campos de entrada):
```json
{"nombre": "temp_entrada_leche", "tipo": "float", "descripcion": "Temperatura de entrada de la leche (°C)", "obligatorio": "Sí", "valor_defecto": "-"}
```

`outputs[]` (Tabla 3 — Campos de salida):
```json
{"nombre": "consumo_agua_l", "tipo": "float (L)", "descripcion": "Consumo de agua predicho", "rango": "positivo continuo", "interpretacion": "valores más bajos indican mayor eficiencia"}
```

`metricas[]` (Tabla 4 — Métricas de evaluación):
```json
{"nombre": "MAE", "valor": "341.58 L", "descripcion": "Error absoluto medio en test hold-out", "interpretacion": "el modelo se equivoca de media X litros"}
```

`limitaciones[]` (Tabla 5 — Limitaciones conocidas):
```json
{"tipo": "Solo vino blanco", "descripcion": "El modelo fue entrenado exclusivamente con..."}
```

## Ficha funcional (`--tipo funcional`)

| Clave | Tipo | Descripción |
|---|---|---|
| `model_id`, `model_slug`, `fecha` | string | igual que en la ficha técnica |
| `descripcion` | string | Qué hace el modelo, en lenguaje de negocio (sin jerga técnica) |
| `objetivo` | string | Para qué sirve, en lenguaje de negocio |
| `caso_uso` | string | Cuándo lo usa el usuario final, paso a paso conceptual |
| `que_no_responde` | string | Una frase: qué NO contesta el modelo (para evitar expectativas erróneas) |
| `entradas_contexto` | string | Frase introductoria antes de la tabla de inputs (de dónde vienen los datos) |
| `inputs` | lista de objetos | `{"nombre": ..., "descripcion": ...}` (sin tipo/obligatoriedad, es la vista de negocio) |
| `outputs` | lista de objetos | `{"nombre": ..., "descripcion": ..., "interpretacion": ...}` |
| `ejemplo_interpretacion` | string | Ejemplo concreto de cómo leer un resultado típico |
| `margen_error` | string | Explicación en lenguaje llano del margen de error esperado |
| `revisar_manualmente` | lista de strings | Frases, una por caso en que el usuario debe revisar a mano |
| `que_no_hace` | lista de strings | Frases, una por cada cosa que el modelo NO hace/garantiza |
| `puede_equivocarse` | string | Honestidad sobre el rendimiento: sí puede fallar, en qué medida, y que es una ayuda, no una garantía |
| `kpis` | lista de objetos | `{"kpi": ..., "valor_umbral": ..., "descripcion": ...}` |
| `limitaciones` | lista de objetos | igual formato que en la ficha técnica, pero puede repetirse con redacción más llana |

## Notas de implementación

- Las listas (`inputs`, `outputs`, `metricas`, `limitaciones`, `kpis`,
  `revisar_manualmente`, `que_no_hace`) pueden tener **cualquier número de
  elementos** (incluido 1): la plantilla duplica la fila/párrafo
  automáticamente vía `docxtpl`.
- No modifiques las plantillas `.docx` a mano: si necesitas cambiar el
  diseño (colores, fuentes, orden de columnas), edita `ficha_tecnica_template.docx`
  / `ficha_funcional_template.docx` en Word y vuelve a insertar las etiquetas
  Jinja (`{{ campo }}`, `{%tr for %}` / `{%tr endfor %}`, `{%p for %}` / `{%p endfor %}`)
  siguiendo exactamente el patrón de 3 filas/párrafos descrito en
  `reference/notas_tecnicas_plantilla.md`.
