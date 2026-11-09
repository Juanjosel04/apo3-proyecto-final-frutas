# Ética e impactos de la solución

> **Estado:** borrador inicial (FASE 1). La versión final de esta sección, la revisión de la tabla ética y el análisis final de impactos quedan a cargo de [PENDIENTE: nombre compañero A].

**Nota de contexto.** El documento de competencias del curso menciona un "sistema de anotación de video" y la "percepción de vigilancia". Ese texto viene de otro semestre; este proyecto clasifica frutas a partir de fotografías. Las dimensiones éticas y de impacto se adaptan a la agroindustria y a la cámara que usaría una línea de selección o la aplicación.

Códigos de referencia:
- ACM, *Code of Ethics and Professional Conduct* (2018) [1].
- IEEE, *Code of Ethics* (Policy 7.8) [2].
- Ley 842 de 2003 de Colombia, que incluye el código de ética para el ejercicio de la ingeniería [3].

## 1. Situaciones éticas y decisiones adoptadas (PI1)

| # | Situación o decisión del proyecto | Riesgo ético | Principio relacionado | Medida adoptada |
|---|---|---|---|---|
| 1 | Toma de fotos propias en plazas de mercado y supermercados | Aparecen personas, caras, placas o marcas sin consentimiento | ACM 1.6 (respetar la privacidad); IEEE I.1 (proteger la privacidad de otros) | Protocolo de fotos: solo la fruta sobre fondo propio, sin personas ni marcas, permiso del vendedor. `scripts/preparar_fotos.py` borra los metadatos EXIF (incluida la ubicación GPS) |
| 2 | Uso del dataset público de Kaggle y de la plantilla del curso | Usar datos o código ajeno sin dar crédito es fraude académico | ACM 1.5 (respetar el trabajo de otros); IEEE I.5 (dar crédito a las contribuciones de otros) | Se cita el dataset (autor y URL) en el README, el informe y el código. No se redistribuye el dataset en el repo; se descarga con un script |
| 3 | Sesgo del modelo por tipo de fruta, iluminación, cámara o desbalance de clases | El sistema funciona bien en las fotos de estudio y mal en las del celular, o favorece la clase mayoritaria | ACM 1.4 (ser justo); ACM 2.5 (evaluaciones completas de los riesgos) | EDA del desbalance, partición estratificada, F1 macro como métrica principal, pesos por clase y evaluación aparte con fotos propias (prueba externa) |
| 4 | Falsos positivos de calidad: una fruta *mala* clasificada como *buena* | Producto en mal estado que llega al consumidor (salud y confianza) | ACM 1.2 (evitar el daño); IEEE I.1 (seguridad y bienestar del público) | Se reporta el recall de la clase *mala* y la matriz de confusión. La app muestra la confianza y recomienda revisión humana si es baja |
| 5 | Decisiones automáticas sin supervisión humana | Se descartan lotes o se fijan precios solo con la salida del modelo | ACM 2.5; ACM 3.7 (cuidado con sistemas que se integran a procesos de la sociedad) | El sistema se plantea como **apoyo** al seleccionador, no como reemplazo. Se documentan las limitaciones (tamaño relativo, frutas no vistas) |
| 6 | Afirmaciones sobre el desempeño del sistema | Exagerar la exactitud en el informe o el video | IEEE I.5 (ser honesto y realista con las estimaciones) | Todas las cifras salen de `experiments/results/`. Se reportan también los fallos y el desempeño en datos propios |
| 7 | Manejo de credenciales (token de Kaggle, cuentas) | Exponer tokens o contraseñas en un repositorio público (el repositorio plantilla del curso trae credenciales de bases de datos expuestas) | ACM 2.9 (sistemas seguros); ACM 1.7 (confidencialidad) | `kaggle.json` y `.env` están en `.gitignore` y nunca se versionan |
| 8 | Datos compartidos entre los grupos del curso | Mezclar etiquetas de otros grupos sin saber su criterio, o publicar fotos ajenas | ACM 1.5; ACM 1.3 (honestidad) | Se usa el mismo formato de etiquetas, se registra el autor de cada foto en `etiquetas.csv` y se documenta qué datos son propios y cuáles compartidos |
| 9 | [PENDIENTE: compañero A, completar o ajustar tras las fases de modelado y despliegue] | | | |

## 2. Matriz de impactos (PI2) — análisis inicial

| Dimensión | Impacto positivo | Riesgo | Mitigación propuesta |
|---|---|---|---|
| **Social** | Criterio de calidad más uniforme entre vendedores; menos producto dañado para el consumidor; herramienta accesible para pequeños productores con un celular | Desplazamiento de seleccionadores manuales; desconfianza si el sistema se equivoca; cámara en la línea percibida como vigilancia del trabajador | Diseño como apoyo y no como reemplazo; la cámara enfoca solo la banda o el fondo de la fruta; capacitación y explicación de la confianza de cada predicción |
| **Económica** | Clasificación más rápida y barata; mejor precio para fruta bien clasificada; menos pérdidas poscosecha | Costo de cámaras y equipos; errores que generan devoluciones; dependencia de un modelo que se degrada con nuevas variedades | Modelos clásicos ligeros que corren en hardware barato; reentrenamiento con datos locales; métricas por clase para estimar el costo de los errores |
| **Ambiental** | Menos desperdicio de alimentos al separar a tiempo la fruta que aún sirve para otros usos (jugos, compost) | Consumo energético de entrenar y ejecutar una CNN; residuos electrónicos | Comparar el tiempo de inferencia de la CNN contra SVM/RF y preferir el modelo más liviano con desempeño similar; posible uso de Raspberry Pi (bajo consumo) |
| **Global** | El método se puede adaptar a otras frutas y países | Normas de calidad distintas por país (por ejemplo, NTC de ICONTEC en Colombia, estándares de USDA o de la Unión Europea); variedades locales que no están en el dataset | Clases configurables; documentar la definición de cada clase; reentrenar con fotos locales como las de este proyecto |

**Análisis final de impactos:** [PENDIENTE: compañero A, FASE 5, con los resultados medidos (tiempos de inferencia, desempeño en datos propios)].

## Referencias

[1] Association for Computing Machinery, "ACM Code of Ethics and Professional Conduct," 2018. [En línea]. Disponible: https://www.acm.org/code-of-ethics

[2] IEEE, "IEEE Code of Ethics," IEEE Policies, Sec. 7.8. [En línea]. Disponible: https://www.ieee.org/about/corporate/governance/p7-8.html

[3] Congreso de Colombia, "Ley 842 de 2003," *Diario Oficial*, no. 45.340, 2003.
