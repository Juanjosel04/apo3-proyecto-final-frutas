"""Valida ``data/propias/etiquetas.csv`` contra las fotos de ``data/propias/raw``.

Revisa que:
- el CSV tenga exactamente las columnas esperadas;
- cada fila apunte a una foto que exista y esté en la carpeta de su fruta y calidad;
- los valores de fruta, calidad, tamaño y lugar sean válidos;
- el diámetro sea un número positivo y la fecha tenga formato AAAA-MM-DD;
- no haya fotos sin etiqueta ni filas repetidas.

Uso (desde la raíz del repo):
    python scripts/validar_etiquetas.py

Devuelve código de salida 1 si encuentra errores.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.config import (  # noqa: E402
    CALIDADES,
    COLUMNAS_ETIQUETAS,
    ETIQUETAS_PROPIAS,
    FOTOS_PROPIAS,
    FRUTAS,
    LUGARES,
    TAMANOS,
)


def validar(ruta_csv: Path = ETIQUETAS_PROPIAS, carpeta: Path = FOTOS_PROPIAS) -> list[str]:
    """Devuelve la lista de errores encontrados (vacía si todo está bien)."""
    errores = []
    df = pd.read_csv(ruta_csv, dtype=str).fillna("")

    if list(df.columns) != COLUMNAS_ETIQUETAS:
        return [f"Columnas incorrectas: {list(df.columns)}. Se esperaba {COLUMNAS_ETIQUETAS}"]

    repetidos = df[df.duplicated("archivo", keep=False)]["archivo"].unique()
    errores += [f"Archivo repetido en el CSV: {a}" for a in repetidos]

    for i, fila in df.iterrows():
        linea = i + 2  # +1 por encabezado, +1 porque las filas empiezan en 1
        ruta = carpeta / fila["fruta"] / fila["calidad"] / fila["archivo"]
        if fila["fruta"] not in FRUTAS:
            errores.append(f"Línea {linea}: fruta inválida '{fila['fruta']}'")
        if fila["calidad"] not in CALIDADES:
            errores.append(f"Línea {linea}: calidad inválida '{fila['calidad']}'")
        if fila["tamano"] not in TAMANOS:
            errores.append(f"Línea {linea}: tamaño inválido '{fila['tamano']}'")
        if fila["lugar"] not in LUGARES:
            errores.append(f"Línea {linea}: lugar inválido '{fila['lugar']}'")
        if not ruta.exists():
            errores.append(f"Línea {linea}: no existe {ruta.relative_to(RAIZ)}")
        try:
            if float(fila["diametro_cm"]) <= 0:
                raise ValueError
        except ValueError:
            errores.append(f"Línea {linea}: diámetro inválido '{fila['diametro_cm']}'")
        try:
            datetime.strptime(fila["fecha"], "%Y-%m-%d")
        except ValueError:
            errores.append(f"Línea {linea}: fecha inválida '{fila['fecha']}' (use AAAA-MM-DD)")
        if not fila["autor"]:
            errores.append(f"Línea {linea}: falta el autor")

    etiquetadas = set(df["archivo"])
    for foto in carpeta.rglob("*.jpg"):
        if foto.name not in etiquetadas:
            errores.append(f"Foto sin etiqueta: {foto.relative_to(RAIZ)}")

    return errores


def main() -> None:
    errores = validar()
    if errores:
        print(f"Se encontraron {len(errores)} errores:")
        for e in errores:
            print(f"  - {e}")
        sys.exit(1)
    n = len(pd.read_csv(ETIQUETAS_PROPIAS))
    print(f"etiquetas.csv correcto: {n} fotos etiquetadas.")


if __name__ == "__main__":
    main()
