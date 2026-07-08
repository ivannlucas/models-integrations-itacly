# Notas técnicas de las plantillas (para mantenimiento futuro)

Las plantillas usan [docxtpl](https://docxtpl.readthedocs.io/) (Jinja2 sobre
`.docx`). Si necesitas tocar el diseño en Word, ten en cuenta lo siguiente
para no romper las listas de longitud variable (inputs, outputs, métricas,
limitaciones, kpis, bullets de "qué no hace"/"revisar manualmente").

## Namespace del XML

Estos documentos concretos (heredados de los ejemplos `a25_v2`) tienen
`word/document.xml` con el prefijo de namespace `ns0:` en lugar del habitual
`w:`. `docxtpl` busca literalmente `<w:tr`, `<w:p`, etc., así que **si el
namespace no es `w:`, las etiquetas `{%tr %}` / `{%p %}` no se reconocen**
(Jinja lanza `Encountered unknown tag`). Por eso `generar_ficha.py` normaliza
el prefijo a `w:` después de cada render (`fix_ns_prefix`). Si editas las
plantillas en Word y las vuelves a guardar, Word usará `w:` de forma nativa
y ese paso deja de ser necesario (pero no molesta dejarlo).

## Patrón de fila repetida (`{%tr %}`)

**No pongas `{%tr for %}` y `{%tr endfor %}` en la misma fila.** El
preprocesador de docxtpl busca, para cada etiqueta `{%tr ...%}`, la
primera `</w:tr>` que encuentre después — si ambas etiquetas están en la
misma fila, la primera coincidencia (`for`) se "come" toda la fila
(incluida la celda con `endfor`) y el resultado es una lista vacía.

El patrón correcto usa **tres filas consecutivas**:

```
Fila A (marcador, desaparece):   celda 1 = "{%tr for x in lista %}"   (resto de celdas vacías)
Fila B (plantilla, se repite):   celda 1 = "{{ x.campo1 }}"  celda 2 = "{{ x.campo2 }}" ...
Fila C (marcador, desaparece):   celda 1 = "{%tr endfor %}"           (resto de celdas vacías)
```

Solo la fila B se repite en la salida, una vez por elemento de `lista`.

## Patrón de párrafo repetido (`{%p %}`)

Mismo problema, misma solución, pero con párrafos en vez de filas: usa
**tres párrafos consecutivos** (marcador `for`, párrafo plantilla, marcador
`endfor`), nunca dos.

```
Párrafo A: "{%p for item in lista %}"
Párrafo B: "•  {{ item }}"
Párrafo C: "{%p endfor %}"
```

## Tablas de longitud fija (no usan bucle)

Las tablas clave-valor (Información general del modelo, Descripción
general, Fichero de prueba de referencia) tienen un número fijo de filas
conocido de antemano: ahí simplemente se sustituye el texto de la celda
por `{{ campo }}`, sin bucles.

## Si necesitas añadir una columna o cambiar el orden

1. Edita la plantilla en Word (no toques el XML a mano).
2. Localiza las 3 filas/párrafos del bloque afectado.
3. Actualiza las etiquetas Jinja manteniendo el patrón de arriba.
4. Actualiza también `reference/esquema_datos.md` si cambias los nombres de
   campo, y `scripts/generar_ficha.py` si necesitas lógica nueva.
5. Vuelve a probar con datos de ejemplo (ver "Verify the output" en el
   skill `docx`) antes de dar por bueno el cambio.
