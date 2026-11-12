# Fase 3 — Preparación de los datos

Evidencia: [`notebooks/02_preparacion_segmentacion.ipynb`](../../notebooks/02_preparacion_segmentacion.ipynb) (segmentación y tamaño), [`notebooks/03_modelos_clasicos.ipynb`](../../notebooks/03_modelos_clasicos.ipynb) (recortes y características) y el código de `src/data/` y `src/features/`.

## 3.1 Selección de los datos

- **Frutas:** manzana, banano y naranja (justificación en la [fase 2](02_comprension_datos.md#24-decisiones-que-salen-de-esta-fase)).
- **Imágenes:** las individuales *good* y *bad* de esas frutas: 6.865 imágenes. Las escenas *mixed* se dejan para el etiquetado de la clase "regular" (sección 3.5).
- **Etiquetas del proyecto:** *good* → `buena` y *bad* → `mala` (`CALIDAD_DESDE_DATASET` en `src/config.py`).

## 3.2 Limpieza: segmentación y recorte

Pipeline por imagen (`python -m src.data.recortes`, unos 4 minutos con 11 procesos):

1. Máscara de la fruta con GrabCut inicializado con Otsu (`src/data/segmentacion.py`, ver notebook 02).
2. Diámetro equivalente normalizado $d = 2\sqrt{A/\pi}/\max(H, W)$ (`src/data/tamano.py`).
3. Recorte cuadrado centrado en la fruta, con un margen del 6 %, **fondo pintado de negro** y tamaño fijo de 192×192 px. Se conserva la proporción de la fruta.

El recorte ataca dos de los sesgos encontrados en el EDA: el fondo desaparece y todas las imágenes quedan con la misma resolución.

**Control de calidad de la segmentación.** Una máscara que cubre menos del 2 % o más del 98 % de la imagen se marca como fallo (`seg_ok = False`) y se excluye. Todos los fallos fueron máscaras demasiado pequeñas:

| Fruta | Fallos en *bad* | Fallos en *good* |
|---|---:|---:|
| Banano | 60 de 1.087 (5,5 %) | 177 de 1.113 (15,9 %) |
| Manzana | 15 de 1.141 (1,3 %) | 103 de 1.149 (9,0 %) |
| Naranja | 7 de 1.159 (0,6 %) | 8 de 1.216 (0,7 %) |

Quedan **6.495 imágenes**. Los fallos se concentran en *good* de banano y manzana, que son las fotos con racimos que llenan la imagen o fondos de pasto (ver limitaciones en notebook 02). Después del filtro las clases siguen casi balanceadas (936 a 1.208 por fruta y clase).

**Limitación conocida.** En muchas fotos *bad* la fruta está sostenida con la mano y GrabCut incluye parte de la mano en la máscara, porque el color de la piel se parece al de la fruta. Eso deja en el recorte una pista que no es la fruta. No se corrigió porque requeriría un segmentador entrenado; se tiene en cuenta al interpretar los resultados y se controla con las fotos propias, que se toman sin manos.

**Reproducibilidad de la segmentación.** GrabCut inicializa sus mezclas de gaussianas con k-means aleatorio, así que sin semilla la misma imagen puede dar máscaras algo distintas. Se detectó después de generar los recortes de esta etapa. Desde entonces `segmentacion.py` fija la semilla de OpenCV y usa la máscara de Otsu cuando GrabCut deja casi nada. El efecto medido está en la [fase 5](05_evaluacion.md#53-por-qué-estos-resultados-son-optimistas). Los recortes se regenerarán al reentrenar con tres clases.

## 3.3 Construcción de los datos: características manuales

`python -m src.features.extraccion` calcula 78 características por recorte, **solo sobre los píxeles de la máscara** (fórmulas en [`docs/formulacion_matematica.md`](../formulacion_matematica.md#4-características-manuales-modelos-clásicos)):

| Grupo | Cantidad | Qué mide |
|---|---:|---|
| `hist_` | 32 | Histogramas de H, S y V |
| `mom_` | 18 | Media, desviación y asimetría de HSV y CIELab |
| `glcm_` | 10 | Textura de Haralick a distancias 1 y 3 |
| `lbp_` | 10 | Patrones binarios locales uniformes |
| `forma_` | 5 | Área relativa, circularidad, excentricidad, solidez y aspecto |
| `def_` | 3 | Proporción de píxeles oscuros, cafés y poco saturados |

No se incluye el tamaño de la fruta entre las características de calidad: en el dataset la distancia de la cámara cambia de foto a foto, así que el tamaño en píxeles diría más de la foto que de la fruta.

Resultado: `data/processed/caracteristicas_kaggle.csv` (6.495 × 78, versionado).

Para la CNN no se calculan características: usa los mismos recortes reducidos a 128×128.

## 3.4 Partición

`asignar_particion` (en `src/data/recortes.py`) divide las 6.865 imágenes en **70 % entrenamiento, 15 % validación y 15 % prueba**, estratificando por fruta y calidad original, con semilla 42. La partición se hace antes del filtro de segmentación y con la calidad del dataset, para que no cambie cuando se etiquete la clase "regular".

| | Entrenamiento | Validación | Prueba |
|---|---:|---:|---:|
| Imágenes (después del filtro) | 4.563 | 966 | 966 |

- **Modelos clásicos:** `GridSearchCV` con validación cruzada estratificada de 5 pliegues sobre entrenamiento + validación (5.529 imágenes). La prueba no se toca hasta el final.
- **CNN:** entrenamiento para ajustar los pesos, validación para *early stopping* y prueba al final.
- **Prueba externa:** las fotos propias del equipo (fase 3 del plan).

No hay duplicados exactos entre particiones (MD5 único por imagen, verificado en la fase 2).

## 3.5 Preparado para tres clases

Mientras la clase "regular" no esté etiquetada se entrena con `buena` y `mala`. El código ya soporta tres clases:

1. Se llena `data/processed/etiquetas_regular.csv` con las columnas `ruta,calidad`, una fila por imagen *bad* que se reetiquete (`regular` o `mala`), usando los criterios de [`docs/protocolo_fotos.md`](../protocolo_fotos.md).
2. En `src/config.py` se cambia `CLASES_ENTRENAMIENTO = ["buena", "mala"]` por `CLASES_ENTRENAMIENTO = CALIDADES`.
3. Se vuelve a correr `python -m src.training.entrenar clasicos` y `python -m src.training.entrenar cnn`.

`cargar_conjunto` aplica las etiquetas del archivo encima de las del dataset; los modelos usan `class_weight="balanced"` y la CNN pesos por clase, porque "regular" quedará minoritaria; y las métricas (`src/evaluation/evaluar.py`) ya calculan F1 por clase y AUC uno contra el resto para $K = 3$ (probado en `tests/test_modelos.py`).

Los recortes de `mixed_quality` (`data/interim/mixed/recortes_mixed.csv`) se podrán agregar al índice de recortes como datos de entrenamiento adicionales cuando se etiqueten sus columnas `valido` y `calidad` (falta implementarlo).
