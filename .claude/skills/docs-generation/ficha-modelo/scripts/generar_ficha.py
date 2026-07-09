#!/usr/bin/env python3
"""
Genera una ficha técnica o funcional de modelo DatagIA a partir de un JSON
de datos y la plantilla docxtpl correspondiente.

Uso:
    python3 generar_ficha.py --tipo tecnica   --datos datos_a35.json --out a35_ficha_tecnica_v2.docx
    python3 generar_ficha.py --tipo funcional --datos datos_a35.json --out a35_ficha_funcional_v2.docx

El JSON de datos debe seguir el esquema descrito en reference/esquema_datos.md.
Un mismo fichero de datos puede alimentar ambas plantillas: cada plantilla
solo usa las claves que necesita e ignora el resto.
"""
import argparse
import json
import sys
import zipfile
import shutil
from pathlib import Path

from docxtpl import DocxTemplate

SKILL_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = {
    "tecnica": SKILL_DIR / "assets" / "ficha_tecnica_template.docx",
    "funcional": SKILL_DIR / "assets" / "ficha_funcional_template.docx",
}


def fix_ns_prefix(path):
    """docxtpl necesita que document.xml use el prefijo 'w:' para el
    namespace de wordprocessingml. Si el motor de renderizado deja algún
    prefijo genérico (ns0, ns1...) para ese namespace, lo normalizamos."""
    tmp = str(path) + ".tmp"
    with zipfile.ZipFile(path, "r") as zin:
        names = zin.namelist()
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for name in names:
                data = zin.read(name)
                if name == "word/document.xml":
                    text = data.decode("utf-8")
                    if "xmlns:w=" not in text:
                        import re
                        m = re.search(r'xmlns:(ns\d+)="http://schemas\.openxmlformats\.org/wordprocessingml/2006/main"', text)
                        if m:
                            prefix = m.group(1)
                            text = text.replace(f"xmlns:{prefix}=", "xmlns:w=")
                            text = text.replace(f"{prefix}:", "w:")
                    data = text.encode("utf-8")
                zout.writestr(name, data)
    shutil.move(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tipo", required=True, choices=["tecnica", "funcional"])
    ap.add_argument("--datos", required=True, help="Fichero JSON con los datos del modelo")
    ap.add_argument("--out", required=True, help="Ruta del .docx de salida")
    ap.add_argument("--template", help="Ruta alternativa a la plantilla (por defecto usa assets/)")
    args = ap.parse_args()

    template_path = Path(args.template) if args.template else TEMPLATES[args.tipo]
    if not template_path.exists():
        sys.exit(f"No se encuentra la plantilla: {template_path}")

    with open(args.datos, encoding="utf-8") as f:
        ctx = json.load(f)

    tpl = DocxTemplate(str(template_path))
    tpl.render(ctx, autoescape=True)
    tpl.save(args.out)
    fix_ns_prefix(args.out)
    print(f"OK: generado {args.out}")


if __name__ == "__main__":
    main()
