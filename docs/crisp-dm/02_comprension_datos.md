# Fase 2 — Comprensión de los datos

Evidencia: [`notebooks/01_comprension_datos_eda.ipynb`](../../notebooks/01_comprension_datos_eda.ipynb), figuras en [`experiments/figures/`](../../experiments/figures/) (prefijo `eda_`) y tablas en [`experiments/results/`](../../experiments/results/).

## 2.1 Recolección inicial

| Fuente | Descripción | Uso en el proyecto |
|---|---|---|
| *Fruit Quality Classification* (Kaggle, R. Park, licencia CC0) [1] | Copia de **FruitNet** [2]: fotos tomadas con celular en Maharashtra (India), con distintos fondos e iluminación | Entrenamiento, validación y prueba interna |
| Carpeta `Mixed Qualit_Fruits` del mismo dataset | Escenas con varias frutas de calidad mezclada, sin etiqueta por fruta | Se separan las frutas y se etiquetan a mano (enriquecimiento) |
| Fotos propias del equipo | 50–60 fotos tomadas en Cali con el [protocolo de fotos](../protocolo_fotos.md) | Prueba externa (generalización) y validación del tamaño en cm |

El dataset se descarga con `python -m src.data.descargar_datos` (unos 3,3 GB). El inventario completo, con una fila por imagen, está en `data/processed/inventario_kaggle.csv`.

## 2.2 Descripción de los datos

- **19.526 imágenes**, todas legibles y sin duplicados exactos (verificado con MD5).
- **6 frutas:** manzana, banano, guayaba, lima, naranja y granada.
- **Calidad original binaria:** *good* (11.664) y *bad* (6.788), más 1.074 escenas *mixed*.
- **Resolución:** mayoría de 256×256 y 192×256. Hay 564 imágenes *good* sin reducir (3000×4000 y 8000×6000).

| Fruta | bad | good | mixed | Razón good/bad |
|---|---:|---:|---:|---:|
| Banano | 1.087 | 1.113 | 285 | 1,02 |
| Granada | 1.187 | 5.940 | 125 | 5,00 |
| Guayaba | 1.129 | 1.152 | 148 | 1,02 |
| Lima | 1.085 | 1.094 | 278 | 1,01 |
| Manzana | 1.141 | 1.149 | 113 | 1,01 |
| Naranja | 1.159 | 1.216 | 125 | 1,05 |
| **Total** | **6.788** | **11.664** | **1.074** | 1,72 |

## 2.3 Exploración

**Desbalance.** Cinco frutas están balanceadas. Granada concentra el 38,6 % de las imágenes individuales con razón 5:1; si se usa, hay que submuestrear o ponderar por clase.

**Sesgo de captura.** Es el hallazgo más importante. Las dos clases se fotografiaron en condiciones distintas (figura `eda_condiciones_captura`):
- Todas las imágenes de alta resolución son *good*.
- En cinco de las seis frutas el fondo de las *bad* es más claro: la mediana del brillo del borde está entre 186 y 223, contra 142 a 202 en las *good*. Las *bad* se tomaron casi siempre sobre baldosa clara y sostenidas con la mano; las *good*, sobre pasto, pisos y telas.
- En guayaba y granada la nitidez (varianza del Laplaciano) de las *good* es unas 15 veces la de las *bad*.

Un clasificador entrenado con la imagen completa puede aprender esos atajos (fondo, resolución, nitidez) en lugar de la calidad de la fruta, y fallar con fotos nuevas. Esto justifica cuatro decisiones de la preparación:
1. Segmentar la fruta y calcular las características solo sobre sus píxeles.
2. Redimensionar todas las imágenes a un tamaño común.
3. Aumentar los datos con variaciones de brillo y fondo.
4. Medir la generalización con las fotos propias.

**Color y calidad.** Sobre la fruta segmentada, el histograma de tono de las frutas malas se desplaza hacia el café/naranja en banano, guayaba y lima (figura `eda_histograma_tono`). La proporción de píxeles oscuros ($V < 80$) es mayor en las malas de granada, manzana y banano, pero **no** en naranja ni guayaba, donde la fruta buena es verde y más oscura (figura `eda_pixeles_oscuros`). Las características de color sirven, pero su relación con la calidad depende de la fruta.

**Calidad de los datos.** Muchas imágenes "individuales" tienen varias frutas, manos o fondos con textura, así que el supuesto de "una fruta sobre fondo simple" no se cumple en el dataset (figura `eda_ejemplos`). Sí se cumplirá en las fotos propias y en el uso de la app.

**mixed_quality.** Son escenas de 256×192 con 3,8 a 6,8 frutas detectadas en promedio según la fruta (muestra de 300), sobre pasto o láminas de color (figuras `eda_mixed_ejemplos` y `eda_mixed_conteo`).

## 2.4 Decisiones que salen de esta fase

1. **Frutas: banano, guayaba y naranja** (propuesta, pendiente de confirmar por el equipo). Están balanceadas (1.087–1.216 imágenes por clase), son baratas y fáciles de conseguir en plazas y supermercados de Cali, y presentan defectos variados (sobremaduración, manchas, golpes, hongos). Lima queda como alternativa. Granada se descarta por el desbalance 5:1 y porque es difícil de conseguir; manzana, porque en Colombia casi toda es importada y es difícil encontrarla en mal estado para las fotos.
2. **Clases de calidad: buena, regular y mala.** El dataset solo trae *good* y *bad*, y el enunciado pide al menos tres. Propuesta:
   - **buena** = *good*;
   - **mala** = *bad* con defecto extenso (podredumbre, hongo, golpe grande);
   - **regular** = defecto leve (manchas pequeñas, golpe leve, sobremaduración). Se construye revisando a mano las *bad* (con apoyo de la proporción de área con defecto calculada sobre la máscara) y etiquetando los recortes de *mixed*.

   Las definiciones están en [`docs/protocolo_fotos.md`](../protocolo_fotos.md).
3. **Segmentación robusta a fondos con textura:** GrabCut inicializado con Otsu (ver fase 3).
4. **Tamaño:** en el dataset solo se puede estimar el tamaño relativo. El tamaño físico se valida con las fotos propias.

## Referencias

[1] R. Park, "Fruit Quality Classification," Kaggle, 2022. [En línea]. Disponible: https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification

[2] V. Meshram and K. Patil, "FruitNet: Indian fruits image dataset with quality for machine learning applications," *Data in Brief*, vol. 40, art. 107686, 2022, doi: 10.1016/j.dib.2021.107686.
