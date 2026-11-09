# Fase 1 — Comprensión del negocio

## 1.1 Contexto

En plazas de mercado, centrales de abastos, supermercados y plantas de empaque, la fruta se clasifica casi siempre a mano: una persona mira cada unidad y decide si es de primera, de segunda o si se descarta. Ese proceso es lento, depende del cansancio y del criterio de cada seleccionador, y no deja registro. El resultado son lotes poco uniformes, reclamos, precios mal fijados y fruta que se pierde por no separarse a tiempo.

El problema es relevante en Colombia. Según el Departamento Nacional de Planeación (DNP), cada año se pierden o desperdician cerca de **9,76 millones de toneladas de alimentos, el 34 % de la oferta alimentaria nacional** [1]. A nivel mundial, la FAO estima que alrededor del **14 % de los alimentos se pierde entre la cosecha y la venta minorista** [2]. Las frutas y hortalizas son de los productos más perecederos, así que una clasificación más rápida y consistente ayuda a decidir a tiempo qué se vende fresco, qué se procesa (jugos, pulpas) y qué se descarta.

## 1.2 Problema de negocio

Construir un sistema que, a partir de **una foto de una fruta sobre un fondo simple**, entregue:

1. su **clase de calidad** (buena, regular o mala), y
2. una **estimación de su tamaño** (pequeño, mediano o grande, más el diámetro equivalente normalizado),

y que se pueda usar desde una interfaz sencilla cargando una imagen o usando la cámara.

## 1.3 Partes interesadas

| Actor | Interés |
|---|---|
| Pequeños productores y vendedores de plaza | Clasificar más rápido y vender por calidad, sin equipos costosos |
| Plantas de empaque y comercializadoras | Lotes uniformes, menos devoluciones, trazabilidad del criterio |
| Consumidor final | Recibir fruta en buen estado |
| Seleccionadores manuales | Herramienta de apoyo; riesgo de reemplazo (ver `docs/etica_e_impactos.md`) |
| Equipo y curso APO III | Aplicar CRISP-DM, ML clásico y deep learning en un problema real |

## 1.4 Objetivos

**Objetivo de negocio.** Apoyar la clasificación de calidad y tamaño de frutas con un criterio uniforme y reproducible que reduzca el tiempo de selección y el paso de fruta dañada al consumidor.

**Objetivos de minería de datos.**
1. Segmentar automáticamente la fruta del fondo, incluso en imágenes con varias frutas (`mixed_quality`).
2. Entrenar y comparar al menos dos modelos clásicos (SVM y Random Forest, con regresión logística como línea base) sobre características de color, textura y forma, y una CNN simple entrenada desde cero.
3. Estimar el tamaño con el diámetro equivalente $d = 2\sqrt{A/\pi}$ del área segmentada.
4. Medir la generalización con fotos propias tomadas por el equipo (prueba externa).
5. Desplegar el mejor modelo en una aplicación Streamlit con carga de imagen y cámara.

## 1.5 Criterios de éxito

| Criterio | Métrica | Meta inicial |
|---|---|---|
| Clasificación de calidad (test del dataset) | F1 macro | ≥ 0,85 |
| Generalización (fotos propias) | F1 macro | ≥ 0,70 y caída frente al test interno documentada |
| Seguridad del consumidor | Recall de la clase *mala* | ≥ 0,85 |
| Mejora sobre la línea base | Diferencia de F1 macro contra regresión logística | > 0 con el modelo elegido |
| Tamaño (fotos propias) | Error relativo del diámetro estimado contra el medido en cm | ≤ 15 % |
| Segmentación | Inspección visual sobre una muestra aleatoria | ≥ 95 % de máscaras correctas |
| Despliegue | Tiempo de respuesta de la app por imagen | < 2 s en un portátil sin GPU |

Las metas son de partida; se revisan en la fase de evaluación con los resultados reales.

## 1.6 Supuestos, restricciones y riesgos

- **Una fruta por foto y fondo uniforme** en la entrada de la app (lo indica el enunciado).
- En el dataset no se conoce la distancia de la cámara, así que el tamaño es **relativo** a la imagen. Para el tamaño físico se usan las fotos propias, tomadas a una distancia fija de unos 30 cm.
- Las etiquetas de calidad son en parte subjetivas; por eso se definen por escrito (ver `docs/protocolo_fotos.md`).
- Riesgo de **cambio de dominio** entre las fotos del dataset y las del celular. Se mide con la prueba externa.
- Recursos: portátiles sin GPU dedicada, por eso la CNN es pequeña.
- Cronograma: [PENDIENTE: fecha de entrega].

## 1.7 Plan del proyecto (por fases CRISP-DM)

| Fase | Entregables en el repo |
|---|---|
| 1. Negocio | Este documento |
| 2. Datos | `notebooks/01_comprension_datos_eda.ipynb`, `docs/crisp-dm/02_comprension_datos.md` |
| 3. Preparación | `src/data/segmentacion.py`, `src/data/tamano.py`, `notebooks/02_preparacion_segmentacion.ipynb` |
| 4. Modelado | `src/features/`, `src/models/`, notebooks 03 y 04 |
| 5. Evaluación | Notebook 05, `experiments/results/`, `experiments/figures/` |
| 6. Despliegue | `app/app_streamlit.py` |

## Referencias

[1] Departamento Nacional de Planeación, *Pérdida y desperdicio de alimentos en Colombia: estudio de la Dirección de Seguimiento y Evaluación de Políticas Públicas*. Bogotá, Colombia: DNP, 2016.

[2] FAO, *The State of Food and Agriculture 2019. Moving Forward on Food Loss and Waste Reduction*. Roma, Italia: FAO, 2019.
