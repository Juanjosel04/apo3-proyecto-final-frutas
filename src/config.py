"""Configuración global del proyecto: rutas, semillas y parámetros comunes."""

from pathlib import Path

# Rutas
RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
DATA_RAW_KAGGLE = DATA / "raw" / "kaggle"
DATA_INTERIM = DATA / "interim"
DATA_PROCESSED = DATA / "processed"
DATA_PROPIAS = DATA / "propias"
FOTOS_PROPIAS = DATA_PROPIAS / "raw"
ETIQUETAS_PROPIAS = DATA_PROPIAS / "etiquetas.csv"

EXPERIMENTS = RAIZ / "experiments"
RESULTADOS = EXPERIMENTS / "results"
FIGURAS = EXPERIMENTS / "figures"
CHECKPOINTS = EXPERIMENTS / "checkpoints"

# Dataset de Kaggle (ryandpark/fruit-quality-classification)
KAGGLE_DATASET = "ryandpark/fruit-quality-classification"

# Reproducibilidad
SEMILLA = 42

# Tamaño de entrada de la CNN
TAM_IMAGEN = (128, 128)

# Extensiones de imagen aceptadas
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
