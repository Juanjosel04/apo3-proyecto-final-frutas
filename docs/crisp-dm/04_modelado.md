# Fase 4 — Modelado

Evidencia: [`notebooks/03_modelos_clasicos.ipynb`](../../notebooks/03_modelos_clasicos.ipynb), [`notebooks/04_cnn.ipynb`](../../notebooks/04_cnn.ipynb), resultados en [`experiments/results/`](../../experiments/results/) (prefijos `clasicos_`, `cnn_`, `sesgo_`) y código en `src/models/` y `src/training/entrenar.py`. Formulación en [`docs/formulacion_matematica.md`](../formulacion_matematica.md), sección 5.

Todos los resultados de esta fase son con **dos clases** (`buena` y `mala`); la clase `regular` se agrega cuando esté etiquetada (ver [fase 3](03_preparacion_datos.md#35-preparado-para-tres-clases)).

## 4.1 Técnicas elegidas y alternativas descartadas

| Técnica | Rol | Por qué |
|---|---|---|
| Regresión logística (L2) | Línea base | Modelo lineal, rápido e interpretable: fija el piso que deben superar los demás |
| SVM con kernel RBF | Modelo clásico 1 | Frontera no lineal con pocas decenas de características y unos miles de ejemplos; se controla con dos hiperparámetros ($C$, $\gamma$) |
| Random Forest | Modelo clásico 2 | No necesita escalado, captura interacciones entre características y da importancias |
| CNN de 3 bloques desde cero | Modelo profundo | Aprende sus propias características de color y textura a partir de los píxeles de la fruta |

Descartadas:
- **KNN:** con 78 características estandarizadas la distancia euclidiana pierde contraste y la predicción es lenta (compara con todo el entrenamiento); la SVM con RBF cubre el mismo tipo de frontera local.
- **XGBoost:** es opcional en el enunciado y el Random Forest ya representa a los ensambles de árboles.
- **Transfer learning (p. ej. MobileNet):** el enunciado pide una CNN entrenada desde cero; queda como comparación opcional para la fase de evaluación.
- **CNN con `Flatten`:** con entrada de 128×128 la capa densa tendría ~2,1 millones de parámetros; con *global average pooling* tiene ~8 mil (formulación, sección 5.4).
- **CNN sobre la imagen completa:** aprendería el fondo y la nitidez, que predicen la calidad en este dataset (sección 4.3).

## 4.2 Diseño de la prueba

- Partición fija 70/15/15 estratificada por fruta y calidad, semilla 42 ([fase 3](03_preparacion_datos.md#34-partición)).
- **Clásicos:** `GridSearchCV` con validación cruzada estratificada de 5 pliegues sobre entrenamiento + validación (5.529 imágenes), métrica F1 macro, `class_weight="balanced"` y escalado dentro del `Pipeline`. El mejor modelo se reentrena con las 5.529 imágenes. La SVM se calibra después con Platt (5 pliegues) para dar probabilidades.
- **CNN:** entrenamiento (4.563) para los pesos, validación (966) para *early stopping* y reducción de la tasa de aprendizaje.
- **Prueba (966 imágenes):** solo al final, una vez por modelo. Se reportan F1 macro con intervalo bootstrap del 95 %, recall de `mala` y AUC.

Rejillas (`src/models/clasicos.py`):

| Modelo | Hiperparámetros explorados | Combinaciones |
|---|---|---:|
| Regresión logística | $C \in \{0{,}01;\ 0{,}1;\ 1;\ 10;\ 100\}$ | 5 |
| SVM (RBF) | $C \in \{0{,}1;\ 1;\ 10;\ 100\}$, $\gamma \in \{\text{scale};\ 0{,}001;\ 0{,}01;\ 0{,}1\}$ | 16 |
| Random Forest | árboles $\in \{200, 500\}$, profundidad $\in \{\text{sin límite}, 12\}$, hojas mín. $\in \{1, 4\}$, características por nodo $\in \{\sqrt{p},\ 0{,}3p\}$ | 16 |

## 4.3 Atajo de las condiciones de captura

Antes de comparar modelos se midió cuánto se puede predecir la calidad **sin mirar la fruta**, solo con variables de la imagen completa: brillo, saturación, contraste, nitidez, brillo del fondo y resolución (`python -m src.training.entrenar sesgo`).

| Alcance | Regresión logística | Random Forest |
|---|---:|---:|
| Tres frutas | 0,806 | **0,947** |
| Manzana | 0,854 | 0,946 |
| Banano | 0,898 | 0,964 |
| Naranja | 0,888 | 0,963 |

(F1 macro, validación cruzada de 5 pliegues; `experiments/results/sesgo_captura.csv`.)

Un Random Forest con siete números que describen la foto, no la fruta, llega a F1 = 0,95. Es una muestra clara de *shortcut learning* [1]: el dataset permite resolver la tarea por la vía equivocada. Consecuencias:
1. Se justifica entrenar solo con la fruta segmentada, con el fondo en negro y a resolución fija.
2. Aun así, la segmentación no borra todo: el brillo general de la foto cambia el color de la fruta, la nitidez cambia la textura y en las *bad* a veces queda parte de la mano. Por eso **un F1 alto en la prueba interna no demuestra que el modelo mire los defectos**.
3. La evidencia decisiva es la prueba externa con fotos propias, tomadas con el mismo protocolo para las dos clases.

## 4.4 ¿Un modelo por fruta o uno con la fruta como variable?

Se probaron las tres configuraciones con los tres modelos clásicos (`experiments/results/clasicos_cv_resumen.csv` y `clasicos_test.csv`; para "por fruta" el F1 de CV es el promedio de las tres frutas ponderado por el número de imágenes):

| Configuración | Modelo | F1 CV | F1 prueba | IC 95 % prueba |
|---|---|---:|---:|---|
| Global (sin fruta) | Reg. logística | 0,911 | 0,902 | [0,880; 0,920] |
| | SVM | 0,982 | 0,983 | [0,974; 0,991] |
| | Random Forest | 0,966 | 0,970 | [0,959; 0,979] |
| **Global + fruta (one-hot)** | Reg. logística | 0,913 | 0,901 | [0,880; 0,919] |
| | **SVM** | **0,982** | **0,986** | **[0,977; 0,992]** |
| | Random Forest | 0,968 | 0,970 | [0,959; 0,979] |
| Un modelo por fruta | Reg. logística | 0,955 | 0,962 | [0,949; 0,973] |
| | SVM | 0,984 | 0,985 | [0,976; 0,992] |
| | Random Forest | 0,974 | 0,971 | [0,960; 0,981] |

Lectura:
- Para la **regresión logística** separar por fruta sube el F1 de 0,90 a 0,96. Confirma lo que mostró el EDA: la relación entre color y calidad cambia con la fruta (por ejemplo, más píxeles oscuros indican mala calidad en banano y manzana, pero no en naranja), y un modelo lineal único no puede representar esa interacción. Darle la fruta como *one-hot* no ayuda, porque solo mueve el intercepto.
- Para la **SVM y el Random Forest** las tres configuraciones quedan dentro del mismo intervalo de confianza: los modelos no lineales aprenden la interacción fruta × característica por sí mismos.

**Decisión: un modelo global con la fruta como variable (`global_fruta`).** Razones:
1. Mismo desempeño que tres modelos separados (diferencias de 0,001 a 0,003, muy por debajo del ancho de los intervalos).
2. Cuando se agregue la clase `regular` habrá pocos ejemplos por fruta; un modelo global entrena con todos los ejemplos de esa clase a la vez, mientras que tres modelos se repartirían una clase ya minoritaria.
3. Un solo modelo que versionar, calibrar y cargar en la app.
4. La fruta igual hace falta para el tamaño (los umbrales son por fruta), así que no es un requisito extra. La app la toma de un clasificador de fruta (Random Forest sobre las mismas características: F1 = 0,968 en prueba, `clasificador_fruta.json`) y deja que el usuario la corrija; como la variable fruta casi no cambia la predicción de calidad de la SVM (0,983 sin ella, 0,986 con ella), un error de fruta tiene poco efecto en la calidad.

La CNN es global por construcción: ve la fruta en la imagen.

## 4.5 Modelos clásicos (configuración `global_fruta`)

| Modelo | Mejores hiperparámetros | F1 entrenamiento | F1 CV | Brecha |
|---|---|---:|---:|---:|
| Reg. logística | $C = 10$ | 0,918 | 0,913 | 0,005 |
| SVM (RBF) | $C = 10$, $\gamma = $ `scale` ($1/(p\,\mathrm{Var}(X)) \approx 1/81$) | 0,998 | 0,982 | 0,016 |
| Random Forest | 500 árboles, sin límite de profundidad, $\sqrt{p}$ por nodo, hoja mín. 1 | 1,000 | 0,968 | 0,032 |

- La **regresión logística** casi no sobreajusta (brecha 0,005) pero tiene sesgo alto: su frontera lineal se queda corta.
- La **SVM** memoriza casi todo el entrenamiento, pero su F1 de CV (0,982 ± 0,007) y el de prueba (0,986) coinciden, así que generaliza dentro del dataset. La rejilla muestra los dos extremos (figura `svm_rejilla`): con $\gamma = 0{,}1$ y $C \ge 10$ el F1 de entrenamiento es 1,000 y el de CV baja a 0,973 (sobreajuste: el kernel es tan local que memoriza); con $\gamma = 0{,}001$ y $C = 0{,}1$ ambos quedan en 0,86 (subajuste: la frontera es casi lineal).
- El **Random Forest** tiene F1 de entrenamiento de 1,0 (árboles sin podar) y la mayor brecha; es el comportamiento esperado de un bosque sin límite de profundidad, y la búsqueda prefirió esa opción porque dio el mejor CV.

Las curvas de aprendizaje, la importancia por grupo de características y las 15 características más importantes del Random Forest están en el notebook 03 (figuras `clasicos_curvas_aprendizaje` y `clasicos_importancias`).

## 4.6 CNN desde cero

Arquitectura en `src/models/cnn.py` (formulación, sección 5.4): entrada de 128×128×3 con el fondo en negro; aumentación con volteos, rotación de ±180° y zoom de ±10 %; tres bloques convolucionales con 32, 64 y 128 filtros, normalización por lotes y *max-pooling*; *global average pooling*; capa densa de 64 neuronas; *dropout* 0,4/0,2; softmax. Optimizador Adam (tasa inicial $10^{-3}$, se reduce a la mitad cada 4 épocas sin mejora), entropía cruzada con pesos por clase, lotes de 32 y *early stopping* sobre la pérdida de validación.

Fue un proceso iterativo de dos versiones (`experiments/results/cnn_base_*` y `cnn_doble_*`):

| Variante | Conv. por bloque | Parámetros | Épocas (mejor) | F1 entrenamiento | F1 validación | F1 prueba [IC 95 %] |
|---|---:|---:|---|---:|---:|---|
| base | 1 | 102.306 | 35 (25) | 0,939 | 0,936 | 0,924 [0,907; 0,941] |
| **doble** | 2 | 296.738 | 66 (54) | 0,967 | 0,959 | **0,951 [0,937; 0,965]** |

- **base:** entrenamiento y validación casi iguales y por debajo de los modelos clásicos, es decir **subajuste**. El problema era la capacidad, no el sobreajuste, así que no tenía sentido regularizar más.
- **doble:** con una segunda convolución por bloque, cada neurona del último bloque ve una región más grande (campo receptivo de 36 px en vez de 22 px antes del *pooling* global) y la red puede combinar más patrones de textura. Subieron las tres particiones (+0,03 en validación y prueba) y la brecha entre entrenamiento y prueba quedó en 0,016. Es la **variante final**.
- Entrenar en CPU es lento: unos 100 s por época para la variante doble (66 épocas, cerca de 2 horas de cómputo). Por eso se agregó un respaldo por época (`BackupAndRestore`) que permite reanudar si el equipo se apaga, cosa que pasó durante este trabajo.

**Errores de la CNN** (notebook 04, figura `cnn_errores`). Los más confiados son manzanas rojas sanas predichas como malas. En el dataset el 57 % de las manzanas *bad* son rojas ($a^*$ medio > 150), contra el 28 % de las *good*: la variedad de manzana está correlacionada con la etiqueta, y la red aprendió "rojo intenso → mala". Los demás errores son recortes con mano, papel o fruta partida (fallos de segmentación).

La CNN queda por debajo de la SVM (0,951 contra 0,986; McNemar $p < 0{,}001$, ver [fase 5](05_evaluacion.md)). Con unas 4.500 imágenes y sin preentrenamiento, a la red le cuesta más aprender lo que las características de color ya resumen bien. Un modelo con *transfer learning* (opcional en el enunciado) podría cerrar esa diferencia; se deja como comparación para la fase de evaluación con fotos propias.

## 4.7 Resumen de la fase

- Modelos entrenados: regresión logística (línea base), SVM, Random Forest y CNN desde cero (dos variantes). Todos con semilla 42, partición fija y artefactos en `experiments/`.
- Mejor modelo en la prueba interna: **SVM con kernel RBF**, configuración `global_fruta` ($C = 10$, $\gamma$ = `scale`).
- Aprendizaje de la iteración: medir el atajo de captura y revisar los errores mostró que parte del desempeño viene del dataset (condiciones de captura, variedad de manzana) y no solo de los defectos. Eso orienta la evaluación externa y el protocolo de fotos propias.

## Referencias

[1] R. Geirhos *et al.*, "Shortcut learning in deep neural networks," *Nat. Mach. Intell.*, vol. 2, pp. 665–673, 2020.
