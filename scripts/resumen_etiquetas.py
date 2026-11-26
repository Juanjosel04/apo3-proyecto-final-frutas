"""Resumen de las etiquetas de las fotos propias.

Muestra cuántas fotos hay por fruta y calidad, por autor y cuántas no tienen
diámetro medido.

Uso (desde la raíz del repo):
    python scripts/resumen_etiquetas.py
"""

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.config import CALIDADES, ETIQUETAS_PROPIAS  # noqa: E402


def resumen(ruta: Path = ETIQUETAS_PROPIAS) -> pd.DataFrame:
    """Tabla fruta x calidad con el número de fotos."""
    df = pd.read_csv(ruta)
    tabla = pd.crosstab(df["fruta"], df["calidad"], margins=True, margins_name="total")
    columnas = [c for c in CALIDADES if c in tabla.columns] + ["total"]
    return tabla[columnas]


def main() -> None:
    df = pd.read_csv(ETIQUETAS_PROPIAS)
    print(f"Fotos etiquetadas: {len(df)}\n")
    print(resumen())
    print("\nFotos por autor:")
    print(df["autor"].value_counts().to_string())
    sin_diametro = df["diametro_cm"].isna().sum()
    print(f"\nSin diámetro medido: {sin_diametro} de {len(df)}")


if __name__ == "__main__":
    main()
