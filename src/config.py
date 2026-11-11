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

# Frutas del dataset (nombres en español usados en carpetas y etiquetas)
FRUTAS = ["manzana", "banano", "guayaba", "lima", "naranja", "granada"]

# Frutas elegidas para el proyecto (ver docs/crisp-dm/02_comprension_datos.md)
FRUTAS_PROYECTO = ["manzana", "banano", "naranja"]

# Clases de calidad del proyecto
CALIDADES = ["buena", "regular", "mala"]

# Equivalencia entre la calidad del dataset y la del proyecto. Mientras "regular"
# no esté etiquetada se entrena con dos clases; al llenar ETIQUETAS_REGULAR
# basta con cambiar CLASES_ENTRENAMIENTO a CALIDADES.
CALIDAD_DESDE_DATASET = {"good": "buena", "bad": "mala"}
CLASES_ENTRENAMIENTO = ["buena", "mala"]
ETIQUETAS_REGULAR = DATA_PROCESSED / "etiquetas_regular.csv"

# Recortes segmentados que usan los modelos
RECORTES = DATA_INTERIM / "recortes"
LADO_RECORTE = 192  # se guardan a 192 px y la CNN los reduce a TAM_IMAGEN

# Partición estratificada (fracciones de validación y prueba)
FRACCION_VAL = 0.15
FRACCION_TEST = 0.15

# Etiquetas de las fotos propias
TAMANOS = ["pequeno", "mediano", "grande"]
LUGARES = ["casa", "supermercado", "plaza"]
COLUMNAS_ETIQUETAS = ["archivo", "fruta", "calidad", "diametro_cm", "tamano",
                      "autor", "fecha", "lugar", "notas"]

# Extensiones de imagen aceptadas
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
