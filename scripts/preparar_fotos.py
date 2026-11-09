"""Prepara las fotos propias del equipo antes de subirlas al repo.

Para cada foto:
1. Corrige la orientación según la etiqueta EXIF (fotos de celular).
2. Reduce el lado mayor a 1024 px manteniendo la proporción.
3. Guarda en JPEG sin metadatos (se elimina el GPS y los datos del celular).
4. Renombra como ``<iniciales>_<fruta>_<nn>.jpg`` dentro de
   ``data/propias/raw/<fruta>/<calidad>/``.

Uso (desde la raíz del repo):
    python scripts/preparar_fotos.py --origen "C:/fotos/banano_buena" \
        --fruta banano --calidad buena --iniciales jjl

El número ``nn`` continúa desde la última foto que ya exista con esas
iniciales y esa fruta, así varias corridas no sobrescriben archivos.
"""

import argparse
import re
import sys
from pathlib import Path

from PIL import Image, ImageOps

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.config import CALIDADES, EXTENSIONES, FOTOS_PROPIAS, FRUTAS  # noqa: E402

LADO_MAX = 1024
CALIDAD_JPEG = 90


def siguiente_numero(carpeta_fruta: Path, iniciales: str, fruta: str) -> int:
    """Devuelve el siguiente consecutivo libre para ``iniciales`` y ``fruta``."""
    patron = re.compile(rf"^{iniciales}_{fruta}_(\d+)\.jpg$")
    usados = [
        int(m.group(1))
        for p in carpeta_fruta.rglob("*.jpg")
        if (m := patron.match(p.name))
    ]
    return max(usados, default=0) + 1


def preparar_imagen(ruta: Path, destino: Path) -> None:
    """Corrige orientación, redimensiona y guarda sin metadatos."""
    with Image.open(ruta) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((LADO_MAX, LADO_MAX), Image.LANCZOS)
        # Se copian solo los píxeles para no arrastrar EXIF ni ICC
        limpia = Image.frombytes("RGB", img.size, img.tobytes())
        limpia.save(destino, "JPEG", quality=CALIDAD_JPEG, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--origen", required=True, type=Path, help="Carpeta con las fotos originales")
    parser.add_argument("--fruta", required=True, choices=FRUTAS)
    parser.add_argument("--calidad", required=True, choices=CALIDADES)
    parser.add_argument("--iniciales", required=True, help="Iniciales del autor en minúscula, p. ej. jjl")
    args = parser.parse_args()

    iniciales = args.iniciales.lower()
    if not re.fullmatch(r"[a-z]{2,4}", iniciales):
        parser.error("Las iniciales deben tener entre 2 y 4 letras sin tildes")

    fotos = sorted(p for p in args.origen.iterdir() if p.suffix.lower() in EXTENSIONES)
    if not fotos:
        print(f"No se encontraron imágenes en {args.origen}")
        return

    carpeta_fruta = FOTOS_PROPIAS / args.fruta
    destino_dir = carpeta_fruta / args.calidad
    destino_dir.mkdir(parents=True, exist_ok=True)
    n = siguiente_numero(carpeta_fruta, iniciales, args.fruta)

    for foto in fotos:
        nombre = f"{iniciales}_{args.fruta}_{n:02d}.jpg"
        preparar_imagen(foto, destino_dir / nombre)
        print(f"{foto.name} -> {destino_dir.relative_to(RAIZ) / nombre}")
        n += 1

    print(f"\n{len(fotos)} fotos listas. Agregue una fila por foto en data/propias/etiquetas.csv")


if __name__ == "__main__":
    main()
