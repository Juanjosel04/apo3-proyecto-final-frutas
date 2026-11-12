# Fase 5 — Evaluación

Evidencia: [`notebooks/05_evaluacion_comparativa.ipynb`](../../notebooks/05_evaluacion_comparativa.ipynb) y `experiments/results/comparacion_*.csv`, `mcnemar_clasico_vs_cnn.json` y `tiempos_inferencia_ms.csv`.

Esta fase tiene dos partes:
1. **Evaluación interna** con la partición de prueba del dataset de Kaggle (966 imágenes), hecha en esta etapa.
2. **Evaluación externa** con las fotos propias del equipo, que mide la generalización real y valida el tamaño en centímetros. [PENDIENTE: fase 3 del plan, cuando las fotos estén en el repositorio.]

## 5.1 Evaluación interna (dos clases)

| Modelo | F1 macro [IC 95 %] | Recall `mala` [IC 95 %] | AUC |
|---|---|---|---:|
| Regresión logística (línea base) | 0,901 [0,880; 0,919] | 0,917 [0,888; 0,940] | 0,965 |
| **SVM (RBF)** | **0,986 [0,977; 0,992]** | **0,988 [0,977; 0,996]** | **0,998** |
| Random Forest | 0,970 [0,959; 0,979] | 0,970 [0,954; 0,983] | 0,991 |
| CNN base | 0,924 [0,907; 0,941] | 0,941 [0,919; 0,961] | 0,974 |
| CNN doble | 0,951 [0,937; 0,965] | 0,955 [0,935; 0,972] | 0,984 |

Modelos clásicos en la configuración `global_fruta`; intervalos por *bootstrap* percentil con 1.000 remuestreos.

**Significancia.** En la prueba de McNemar entre la SVM y la CNN doble, la SVM acierta 34 imágenes que la CNN falla y la CNN acierta solo 1 que la SVM falla ($\chi^2 = 29{,}3$, $p < 0{,}001$). Ambas fallan 13. La diferencia no es azar.

**Por fruta (F1 macro en prueba):**

| Modelo | Banano | Manzana | Naranja |
|---|---:|---:|---:|
| Regresión logística | 0,939 | 0,878 | 0,890 |
| SVM | 1,000 | 0,976 | 0,983 |
| Random Forest | 0,996 | 0,948 | 0,969 |
| CNN doble | 0,989 | 0,933 | 0,938 |

En banano todos los modelos son casi perfectos: el banano dañado se pone negro y el sano es amarillo, así que el color separa las clases casi por completo. La manzana es la fruta más difícil.

**Sobreajuste (F1 macro):**

| Modelo | Entrenamiento | Validación | Prueba | Brecha entrenamiento − prueba |
|---|---:|---:|---:|---:|
| Regresión logística | 0,918 | 0,913 | 0,901 | 0,017 |
| SVM | 0,998 | 0,982 | 0,986 | 0,013 |
| Random Forest | 1,000 | 0,968 | 0,970 | 0,030 |
| CNN base | 0,939 | 0,936 | 0,924 | 0,015 |
| CNN doble | 0,967 | 0,959 | 0,951 | 0,016 |

Para los clásicos, "entrenamiento" y "validación" son los promedios de la validación cruzada de 5 pliegues. Ningún modelo muestra sobreajuste dañino: la validación y la prueba coinciden en todos.

**Tiempo de inferencia por imagen** (de la imagen original a la predicción, CPU de un portátil, promedio de 50 imágenes): segmentación y recorte 211 ms, características 10 ms, SVM 2 ms, CNN 25 ms. En total, unos **223 ms con la SVM y 235 ms con la CNN**, dentro del criterio de éxito (< 2 s).

## 5.2 Evaluación frente a los criterios de éxito

| Criterio ([fase 1](01_comprension_negocio.md#15-criterios-de-éxito)) | Meta | Resultado | Estado |
|---|---|---|---|
| Calidad en el test del dataset (F1 macro) | ≥ 0,85 | 0,986 (SVM), 0,951 (CNN doble) | Cumple con dos clases. Falta medirlo con tres. |
| Generalización a fotos propias (F1 macro) | ≥ 0,70 y caída documentada | — | [PENDIENTE: fotos propias] |
| Recall de la clase `mala` | ≥ 0,85 | 0,988 (SVM) | Cumple en el dataset |
| Mejora sobre la línea base | > 0 | +0,085 (SVM contra regresión logística) | Cumple |
| Error relativo del diámetro | ≤ 15 % | — | [PENDIENTE: fotos propias] |
| Segmentación correcta | ≥ 95 % de máscaras | 94,6 % pasan el filtro automático de área | Revisión visual pendiente |
| Tiempo por imagen | < 2 s sin GPU | 0,22 s (SVM), 0,24 s (CNN) | Cumple |

## 5.3 Por qué estos resultados son optimistas

La prueba interna comparte los sesgos del dataset. Tres evidencias de esta etapa:
1. **Condiciones de captura:** brillo, nitidez, fondo y resolución de la imagen completa predicen la calidad con F1 ≈ 0,95 sin mirar la fruta ([fase 4](04_modelado.md#43-atajo-de-las-condiciones-de-captura)).
2. **Variedad de manzana:** el 57 % de las manzanas *bad* son rojas, contra el 28 % de las *good*. Los errores más confiados de la CNN son manzanas rojas sanas.
3. **Segmentación imperfecta:** en algunas *bad* la máscara incluye la mano. Además, GrabCut inicializa sus mezclas de gaussianas de forma aleatoria. Desde esta versión se fija la semilla de OpenCV y, si GrabCut deja una máscara casi vacía, se usa la de Otsu (`src/data/segmentacion.py`). Los recortes y modelos de esta fase se generaron antes de ese cambio. Con el código actual, en una muestra de 300 imágenes la mediana del cambio de área de la máscara es del 0,4 %, pero el 15 % cambia más del 10 %, y 80 de cada 100 fallos anteriores se recuperan. Al reentrenar con tres clases se regenerarán recortes, características y modelos con el código actual.

Por eso **no se puede decir todavía cuál modelo es mejor en la práctica**. La SVM gana en el dataset, pero se apoya casi solo en el color (notebook 03), que es justo lo que más cambia con la luz y la variedad. La decisión final se toma con las fotos propias.

## 5.4 Decisiones y siguientes pasos

1. **Modelo propuesto para la app:** SVM `global_fruta` calibrada, con la CNN doble disponible para comparar. Se revisará después de la prueba externa.
2. **Fotos propias:** incluir manzanas rojas y verdes en las tres clases, y las tres frutas sobre el mismo fondo, para que ni la variedad ni el fondo indiquen la clase ([protocolo de fotos](../protocolo_fotos.md)).
3. **Clase `regular`:** etiquetar, cambiar a tres clases y reentrenar con el código actual ([fase 3](03_preparacion_datos.md#35-preparado-para-tres-clases)).
4. **Opcional:** comparar con *transfer learning* (capas congeladas) en la evaluación externa.
