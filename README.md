# Clasificador de calidad y tamaño de frutas

Este proyecto hace parte del curso **Algoritmos y Programación III (APO III)**, semestre 2026-2, del programa de Ingeniería de Sistemas de la Universidad Icesi, Cali, Colombia.

#### Estado del proyecto: Activo

## Integrantes

**Líder del equipo: [Juan José López Díaz](https://github.com/Juanjosel04) (@Juanjosel04)**

**Profesor: [PENDIENTE: nombre del profesor]**

#### Otros integrantes

| Nombre | Usuario de GitHub |
|---|---|
| [PENDIENTE: nombre] | [PENDIENTE: usuario GitHub] |
| [PENDIENTE: nombre] | [PENDIENTE: usuario GitHub] |

## Contacto
* Preguntas o comentarios: abrir un *issue* en este repositorio o escribir al líder del equipo.

## Introducción y objetivo

En plazas de mercado, supermercados y plantas de empaque la fruta se clasifica casi siempre a mano, lo que es lento, subjetivo y genera pérdidas: en Colombia se pierde o desperdicia cerca del 34 % de la oferta de alimentos (DNP, 2016). El objetivo de este proyecto es construir un sistema de visión por computadora que, a partir de la **foto de una fruta sobre un fondo simple**, indique su **clase de calidad** (buena, regular o mala) y **estime su tamaño** (pequeño, mediano o grande, con el diámetro equivalente en píxeles normalizados). El sistema se entrena con un dataset público y se evalúa también con fotos tomadas por el equipo, para medir qué tan bien generaliza. Se despliega en una aplicación Streamlit que acepta una imagen o la cámara.

### Métodos
* Visión por computadora clásica (segmentación por color, umbral de Otsu, morfología, GrabCut, watershed)
* Extracción de características (color, textura, forma)
* Aprendizaje automático (regresión logística, SVM, Random Forest)
* Aprendizaje profundo (CNN simple entrenada desde cero)
* Visualización de datos y análisis exploratorio
* Metodología CRISP-DM

### Tecnologías
* Python 3.12
* NumPy, pandas, Matplotlib, seaborn
* OpenCV, scikit-image
* scikit-learn
* TensorFlow / Keras
* Streamlit
* Jupyter
* pytest

## Descripción del proyecto

**Datos.**
1. Dataset *Fruit Quality Classification* de Kaggle ([ryandpark/fruit-quality-classification](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification)), copia de FruitNet (Meshram y Patil, 2022): 19.526 imágenes de 6 frutas (manzana, banano, guayaba, lima, naranja y granada) en calidad *good*, *bad* y *mixed*. El proyecto trabaja con **manzana, banano y naranja** (justificación en [docs/crisp-dm/02_comprension_datos.md](docs/crisp-dm/02_comprension_datos.md)). No se incluye en el repositorio; se descarga con `src/data/descargar_datos.py`.
2. Carpeta `mixed_quality` del mismo dataset: cada foto tiene varias frutas, que se separan y recortan una por una con `src/data/segmentacion.py`.
3. Fotos propias del equipo (meta: 50–60), tomadas según el [protocolo de fotos](docs/protocolo_fotos.md) y etiquetadas en `data/propias/etiquetas.csv`. Se usan como conjunto de prueba externo.

**Preguntas que guían el trabajo.**
* ¿Qué tan bien se distingue la calidad de una fruta con características de color, textura y forma frente a una CNN entrenada desde cero?
* ¿Cuánto cae el desempeño al pasar de las imágenes del dataset a fotos tomadas con celular?
* ¿Qué tan preciso es el diámetro estimado a partir de la máscara frente al diámetro medido con regla?

**Modelado.** Segmentación clásica (Otsu sobre la distancia de color al fondo, refinada con GrabCut), estimación de tamaño por diámetro equivalente $d = 2\sqrt{A/\pi}$, línea base con regresión logística, SVM y Random Forest ajustados con `GridSearchCV` y validación cruzada estratificada, y una CNN de 2–3 bloques convolucionales. La formulación completa está en [docs/formulacion_matematica.md](docs/formulacion_matematica.md).

**Retos.** El EDA mostró que en el dataset las fotos *good* y *bad* se tomaron con fondos, resoluciones y nitidez distintos, así que un modelo puede aprender el fondo en lugar de la calidad; por eso se segmenta la fruta antes de extraer características y se evalúa con fotos propias. Otros retos: el dataset solo trae calidad binaria (se necesitan 3 clases), fondos con textura, frutas que se tocan en `mixed_quality` y subjetividad de las etiquetas.

**Metodología.** El trabajo sigue CRISP-DM; cada fase está documentada en [docs/crisp-dm/](docs/crisp-dm/) y resumida en el diagrama [docs/diagramas/crisp_dm.svg](docs/diagramas/crisp_dm.svg). La ética y los impactos están en [docs/etica_e_impactos.md](docs/etica_e_impactos.md).

## Primeros pasos

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/Juanjosel04/apo3-proyecto-final-frutas.git
   cd apo3-proyecto-final-frutas
   ```
2. Crear el entorno e instalar dependencias (Python 3.12):
   ```bash
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Descargar el dataset de Kaggle (unos 3,3 GB; queda en `data/raw/kaggle/`, ignorado por git) y generar los datos intermedios:
   ```bash
   python -m src.data.descargar_datos
   python -m src.data.preprocesamiento
   ```
4. Recortar las frutas, extraer características y entrenar (en CPU tarda unos 4 min, 1 min, 1 min, 30 min, 20 min y 2 h, respectivamente):
   ```bash
   python -m src.data.recortes
   python -m src.features.extraccion
   python -m src.training.entrenar sesgo
   python -m src.training.entrenar clasicos
   python -m src.training.entrenar cnn --variante base
   python -m src.training.entrenar cnn --variante doble
   ```
   Si el entrenamiento de la CNN se interrumpe, al volver a correr el mismo comando continúa desde la última época guardada.
5. Correr las pruebas:
   ```bash
   pytest
   ```
6. Abrir los notebooks de `notebooks/` en orden.

Las instrucciones detalladas están en [docs/instalacion.md](docs/instalacion.md).

### Estructura

```
├── app/                 aplicación Streamlit
├── data/
│   ├── raw/kaggle/      dataset descargado (no se versiona)
│   ├── interim/         recortes de mixed_quality y máscaras (no se versiona)
│   ├── processed/       características en CSV
│   └── propias/         fotos del equipo y etiquetas.csv
├── docs/
│   ├── crisp-dm/        documentación por fase
│   └── diagramas/       diagramas (fuente + SVG/PDF)
├── experiments/         resultados, figuras y modelos
├── informe/             informe IEEE y guion del video
├── notebooks/           exploración y resultados
├── scripts/             herramientas para las fotos propias
├── src/                 código reutilizable
└── tests/               pruebas unitarias
```

## Notebooks y entregables

* [Comprensión de los datos (EDA)](notebooks/01_comprension_datos_eda.ipynb)
* [Preparación y segmentación](notebooks/02_preparacion_segmentacion.ipynb)
* [Modelos clásicos: características, línea base, SVM y Random Forest](notebooks/03_modelos_clasicos.ipynb)
* [CNN desde cero](notebooks/04_cnn.ipynb)
* [Evaluación comparativa](notebooks/05_evaluacion_comparativa.ipynb)
* [Informe final (IEEE)](informe/informe_final.docx)
* Video — [PENDIENTE: enlace]

## Créditos y fuentes de terceros

* Dataset: R. Park, *Fruit Quality Classification*, Kaggle (licencia CC0), https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification. Es una copia de **FruitNet**: V. Meshram y K. Patil, "FruitNet: Indian fruits image dataset with quality for machine learning applications," *Data in Brief*, vol. 40, art. 107686, 2022, doi: 10.1016/j.dib.2021.107686.
* Estructura del repositorio y formato del README basados en la plantilla del curso (M. Sarria-Paja, https://github.com/miltonsarria/ETLInfrati), que a su vez se basa en la del Data Science Working Group de Code for San Francisco (https://github.com/sfbrigade/data-science-wg).
* GrabCut: C. Rother, V. Kolmogorov y A. Blake, "GrabCut: Interactive foreground extraction using iterated graph cuts," *ACM Trans. Graph.*, 2004 (implementación de OpenCV).
* Watershed con marcadores: tutorial de OpenCV (https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html) y ejemplo "Watershed segmentation" de scikit-image (https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html).

## Licencia

Código bajo licencia MIT (ver [LICENSE](LICENSE)). El dataset de Kaggle conserva la licencia de su autor.
