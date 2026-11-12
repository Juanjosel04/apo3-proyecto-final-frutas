# Formulación matemática del problema

Versión extendida de la sección de formulación del informe (indicador PI3). Se irá completando con los resultados de cada fase.

## 1. Variables y espacio del problema

Una fotografía en color de una fruta se representa como un tensor

$$
\mathbf{x} \in \{0,\dots,255\}^{H \times W \times 3},
$$

con $H$ filas, $W$ columnas y tres canales (RGB). El sistema busca una función

$$
f:\ \mathbf{x} \longmapsto (\hat{y},\ \hat{s},\ \hat{d}),
$$

donde

- $\hat{y} \in \mathcal{C} = \{c_1,\dots,c_K\}$ es la clase de calidad ($K \ge 3$; en este proyecto $\mathcal{C} = \{\text{buena}, \text{regular}, \text{mala}\}$),
- $\hat{d} \in \mathbb{R}_{>0}$ es el diámetro equivalente normalizado de la fruta,
- $\hat{s} \in \{\text{pequeño}, \text{mediano}, \text{grande}\}$ es la categoría de tamaño obtenida a partir de $\hat{d}$.

El problema se descompone en tres subproblemas encadenados:

1. **Segmentación:** $g(\mathbf{x}) = M \in \{0,1\}^{H\times W}$, la máscara binaria de la fruta.
2. **Estimación de tamaño:** $\hat{d} = h(M)$ y $\hat{s} = q(\hat{d})$, sin aprendizaje (geometría).
3. **Clasificación de calidad:** $\hat{y} = \arg\max_k\ p_\theta(c_k \mid \mathbf{x})$, con parámetros $\theta$ aprendidos.

## 2. Segmentación

La imagen se lleva a un espacio de color donde la fruta se separa mejor del fondo (canal de saturación $S$ de HSV o canal $a^*$/$b^*$ de CIELab). Sobre ese canal $I(i,j) \in \{0,\dots,255\}$ se aplica el umbral de Otsu [1], que elige

$$
t^* = \arg\max_{t}\ \sigma_B^2(t), \qquad \sigma_B^2(t) = \omega_0(t)\,\omega_1(t)\,\big[\mu_0(t) - \mu_1(t)\big]^2,
$$

donde $\omega_0, \omega_1$ son las proporciones de píxeles por debajo y por encima de $t$, y $\mu_0, \mu_1$ sus medias. Maximizar la varianza entre clases equivale a minimizar la varianza intra-clase, por lo que no se necesita fijar un umbral a mano para cada foto.

La máscara inicial $M_0 = [I > t^*]$ se limpia con morfología matemática usando un elemento estructurante elíptico $B$:

$$
M = \big((M_0 \ominus B) \oplus B\big) \bullet B,
$$

es decir, una apertura (quita ruido pequeño) seguida de un cierre (rellena huecos como el pedúnculo o los brillos). Finalmente se conserva el contorno conexo de mayor área y se rellena su interior.

**Refinamiento con GrabCut.** El EDA mostró que muchas fotos del dataset tienen fondos con textura (pasto, telas, baldosas), donde un solo umbral no basta. Por eso la máscara de Otsu se usa como semilla de GrabCut [6]: los píxeles con $\Delta E$ y saturación por encima de sus umbrales de Otsu se marcan como "probable fruta", la franja del borde como "fondo seguro", y se minimiza la energía

$$
E(\boldsymbol{\alpha}) = \sum_{p} -\log p\big(z_p \mid \alpha_p\big) + \lambda \sum_{(p,q)\in\mathcal{N}} [\alpha_p \ne \alpha_q]\, e^{-\beta \lVert z_p - z_q\rVert^2},
$$

donde $\alpha_p \in \{0,1\}$ es la etiqueta del píxel $p$, $p(z_p\mid\alpha_p)$ es la verosimilitud del color bajo una mezcla de gaussianas de fruta o de fondo, y el segundo término penaliza cortar entre píxeles vecinos de color parecido. La minimización se hace con un corte mínimo de grafo, alternando con la reestimación de las mezclas.

**Separación de frutas en `mixed_quality`.** Sobre la máscara se calcula la transformada de distancia $D(p) = \min_{q \notin M} \lVert p - q \rVert$. El centro de cada fruta aproximadamente redonda es un máximo local de $D$; esos máximos son los marcadores del algoritmo watershed aplicado a $-D$, que ubica la frontera entre dos frutas en contacto en el "cuello" donde $D$ es mínima. Se descartan regiones con radio inscrito $r_{in} = \max_{p\in R} D(p)$ pequeño o con compacidad $r_{in} / \sqrt{A/\pi} < 0{,}35$ (franjas del fondo), y cada región restante se recorta por su caja envolvente.

## 3. Estimación de tamaño

El área en píxeles de la fruta es $A = \sum_{i,j} M(i,j)$. El diámetro del círculo con la misma área es

$$
d_{px} = 2\sqrt{\frac{A}{\pi}}.
$$

Para que el valor no dependa de la resolución de la foto se normaliza por el lado mayor de la imagen:

$$
\hat{d} = \frac{d_{px}}{\max(H, W)} \in (0, 1].
$$

La categoría se asigna con umbrales por tipo de fruta $f$, calculados como percentiles del conjunto de entrenamiento:

$$
q(\hat{d}) =
\begin{cases}
\text{pequeño}, & \hat{d} < P_{33}^{(f)} \\
\text{mediano}, & P_{33}^{(f)} \le \hat{d} < P_{66}^{(f)} \\
\text{grande}, & \hat{d} \ge P_{66}^{(f)}
\end{cases}
$$

Con las fotos propias, tomadas a una distancia fija de unos 30 cm, existe un factor de escala aproximadamente constante $k$ (cm por píxel), y se espera $d_{cm} \approx k\, d_{px}$. Ese factor se estima por mínimos cuadrados con el diámetro medido con regla, $\hat{k} = \sum_i d_{px,i}\,d_{cm,i} / \sum_i d_{px,i}^2$ (el valor que minimiza $\sum_i (d_{cm,i} - k\,d_{px,i})^2$), y el error del estimador se reporta como MAE y error relativo. Para el banano, que es alargado, el diámetro equivalente no es su largo; por eso $k$ se ajusta por fruta y para el banano se compara contra el largo medido.

**Limitación conocida:** en el dataset de Kaggle no se conoce la distancia de la cámara, así que $\hat{d}$ mide el tamaño *relativo dentro de la foto*, no el tamaño físico. Esto se discute en el informe.

## 4. Características manuales (modelos clásicos)

Cada fruta segmentada se recorta a un cuadrado de $192\times192$ px con el fondo en negro y se resume en un vector $\boldsymbol{\phi}(\mathbf{x}, M) \in \mathbb{R}^{78}$ calculado solo sobre los píxeles con $M(i,j)=1$ (implementación: `src/features/extraccion.py`):

| Grupo | Dim. | Características | Justificación |
|---|---:|---|---|
| Histogramas de color | 32 | Histogramas normalizados de H (16 bins), S (8) y V (8): $h_b = \frac{1}{A}\sum_{i,j} M(i,j)\,[H(i,j) \in \text{bin } b]$ | La madurez y la podredumbre cambian el tono |
| Momentos de color | 18 | Media $\mu$, desviación $\sigma$ y asimetría $\gamma = \mathbb{E}[(X-\mu)^3]/\sigma^3$ de cada canal HSV y CIELab | Resumen compacto de la distribución de color |
| Textura GLCM | 10 | Contraste $\sum_{a,b}(a-b)^2P(a,b)$, disimilitud, homogeneidad $\sum_{a,b} \frac{P(a,b)}{1+(a-b)^2}$, energía y correlación, a distancias 1 y 3 px, promediadas en 4 ángulos [2] | Moho, arrugas y piel rota aumentan el contraste local |
| Textura LBP | 10 | Histograma de patrones binarios locales uniformes con $P=8$, $R=1$ [7] | Microtextura invariante a cambios monótonos de brillo |
| Forma | 5 | Área relativa, circularidad $4\pi A/P^2$, excentricidad $\sqrt{1-\lambda_2/\lambda_1}$ (valores propios de la matriz de momentos centrales), solidez $A/A_{hull}$, relación de aspecto | Frutas deformadas, aplastadas o abiertas |
| Defectos | 3 | Proporción de píxeles oscuros $\frac{1}{A}\sum M\,[V<80]$, cafés ($5\le H\le 25$, $S\ge 60$, $V<150$) y poco saturados ($S<40$) | Manchas, golpes y moho |

Para que el fondo no entre en la GLCM, la fruta se cuantiza en los niveles $1,\dots,31$ y el fondo queda en el nivel 0; luego se eliminan la fila y la columna 0 de la matriz $P$ y se renormaliza. El LBP se calcula solo en el interior de la máscara (erosionada un píxel) para no contar el borde fruta-fondo como textura.

Antes de entrenar se estandariza cada característica con la media y la desviación del conjunto de entrenamiento:

$$
z_j = \frac{\phi_j - \mu_j^{train}}{\sigma_j^{train}}.
$$

Esto es necesario para la SVM con kernel RBF, que depende de distancias euclidianas, y para la regresión logística regularizada, cuya penalización supone escalas comparables; sin estandarizar, la característica de mayor rango (por ejemplo los momentos de color, de 0 a 255) dominaría la distancia. El escalado va dentro de un `Pipeline`, así que en cada pliegue de la validación cruzada se ajusta solo con los datos de entrenamiento de ese pliegue.

## 5. Aprendizaje como minimización del riesgo empírico

Con un conjunto de entrenamiento $\{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ se busca

$$
\theta^* = \arg\min_{\theta}\ \frac{1}{n}\sum_{i=1}^{n} \mathcal{L}\big(f_\theta(\mathbf{x}_i), y_i\big) + \lambda\,\Omega(\theta),
$$

donde $\mathcal{L}$ es la pérdida y $\Omega$ un regularizador que controla el sobreajuste.

### 5.1 Línea base: regresión logística multinomial
$p(c_k \mid \mathbf{z}) = \mathrm{softmax}(W\mathbf{z} + \mathbf{b})_k$, entrenada con entropía cruzada y penalización $L_2$.

### 5.2 SVM con margen suave y kernel RBF [3]

$$
\min_{\mathbf{w}, b, \xi}\ \frac{1}{2}\lVert\mathbf{w}\rVert^2 + C\sum_{i}\xi_i \quad \text{s.a.}\quad y_i(\mathbf{w}^\top\varphi(\mathbf{z}_i) + b) \ge 1 - \xi_i,\ \ \xi_i \ge 0,
$$

con $K(\mathbf{z}, \mathbf{z}') = \exp(-\gamma\lVert\mathbf{z}-\mathbf{z}'\rVert^2)$. $C$ controla el compromiso margen/errores y $\gamma$ el alcance del kernel; ambos se ajustan con búsqueda en rejilla. El caso multiclase se resuelve uno contra uno.

La SVM entrega una distancia al hiperplano $s(\mathbf{z}) = \mathbf{w}^\top\varphi(\mathbf{z}) + b$, no una probabilidad. Para mostrar una confianza en la app se calibra con el escalado de Platt [8], $\hat{p}(\text{mala}\mid\mathbf{z}) = 1/(1 + e^{A s(\mathbf{z}) + B})$, donde $A$ y $B$ se ajustan por máxima verosimilitud con predicciones fuera de pliegue (validación cruzada de 5 pliegues).

### 5.3 Random Forest [4]
Ensamble de $B$ árboles entrenados sobre muestras bootstrap y subconjuntos aleatorios de características. Cada nodo elige la partición que más reduce la impureza de Gini

$$
G(t) = 1 - \sum_{k=1}^{K} p_{k}(t)^2 .
$$

La predicción es el voto mayoritario. El promedio de árboles poco correlacionados reduce la varianza sin aumentar mucho el sesgo.

### 5.4 CNN simple entrenada desde cero [5]
Tres bloques, cada uno con $k$ capas [convolución $3\times3$ + normalización por lotes [9] + ReLU] y un *max-pooling* $2\times2$ (32, 64 y 128 filtros), seguidos de *global average pooling* [10], una capa densa de 64 neuronas y softmax (implementación en `src/models/cnn.py`). Se entrenaron dos variantes: `base` con $k=1$ (102 mil parámetros) y `doble` con $k=2$, más profunda, para corregir el subajuste de la primera (ver `docs/crisp-dm/04_modelado.md`). Una capa convolucional calcula

$$
(\mathbf{X} * \mathbf{K})(i,j) = \sum_{m}\sum_{n}\sum_{c} \mathbf{X}(i+m, j+n, c)\,\mathbf{K}(m,n,c) + b,
$$

y la salida es $\hat{\mathbf{p}} = \mathrm{softmax}(\mathbf{o})$, con $\mathrm{softmax}(\mathbf{o})_k = e^{o_k}/\sum_j e^{o_j}$. Se minimiza la entropía cruzada ponderada por clase:

$$
\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} w_k\, y_{ik}\log \hat{p}_{ik}, \qquad w_k = \frac{n}{K\, n_k},
$$

donde $n_k$ es el número de ejemplos de la clase $k$; así las clases minoritarias pesan más. La normalización por lotes estandariza cada activación con la media y varianza del lote, $\hat{a} = (a-\mu_B)/\sqrt{\sigma_B^2+\epsilon}$, y luego aplica una escala y un desplazamiento aprendidos; estabiliza el entrenamiento desde cero. El *global average pooling* promedia cada mapa de características, $g_c = \frac{1}{HW}\sum_{i,j} F(i,j,c)$; con él la capa densa recibe 128 valores en vez de $16\cdot16\cdot128 = 32\,768$, lo que reduce los parámetros de esa capa de ~2,1 millones a ~8 mil y con ello el sobreajuste. La aumentación aplica en cada época volteos horizontales y verticales, una rotación $\theta \sim U(-180°, 180°)$ y un zoom de $\pm10\,\%$, transformaciones que no cambian la calidad de la fruta; el relleno es negro, igual que el fondo del recorte. Se regulariza con *dropout* [11] ($p=0{,}4$ y $0{,}2$) y *early stopping* sobre la pérdida de validación (paciencia 10, se restauran los mejores pesos), y la tasa de aprendizaje de Adam se reduce a la mitad si la pérdida de validación no mejora en 4 épocas.

## 6. Evaluación

- Partición estratificada entrenamiento / validación / prueba con semilla 42, y las fotos propias como **prueba externa** (cambio de dominio).
- Ajuste de hiperparámetros con validación cruzada estratificada de $k$ pliegues (GridSearchCV).
- Métricas por clase: $\mathrm{Prec}_k = \frac{TP_k}{TP_k + FP_k}$, $\mathrm{Rec}_k = \frac{TP_k}{TP_k + FN_k}$, $F1_k = \frac{2\,\mathrm{Prec}_k\mathrm{Rec}_k}{\mathrm{Prec}_k + \mathrm{Rec}_k}$, y **F1 macro** $= \frac{1}{K}\sum_k F1_k$ como métrica principal, porque no se deja dominar por la clase mayoritaria.
- Matriz de confusión y análisis de errores. En calidad, confundir *mala* con *buena* es el error más costoso (una fruta dañada llega al consumidor), por eso se reporta aparte el recall de *mala*.
- Sobreajuste: brecha entre métricas de entrenamiento y validación, curvas de aprendizaje de los modelos clásicos y curvas de pérdida de la CNN.
- Incertidumbre: intervalo de confianza del 95 % del F1 macro en prueba por *bootstrap* percentil [12]: se remuestrean con reemplazo los $n$ ejemplos de prueba 1.000 veces y se toman los percentiles 2,5 y 97,5 de la métrica.
- Atajos: se mide qué tan bien predicen la calidad solo las condiciones de captura (brillo, nitidez, fondo, resolución). Si lo hacen bien, un buen resultado en la prueba interna no garantiza que el modelo mire la fruta (*shortcut learning* [13]); por eso la prueba decisiva es la externa.

## 7. Por qué es un problema complejo de ingeniería

- **Variabilidad visual alta:** varias frutas, variedades, colores de madurez y tipos de defecto en la misma clase.
- **Condiciones de captura no controladas:** iluminación, sombras, fondo y cámara cambian entre el dataset y las fotos del celular (cambio de dominio).
- **Etiquetas subjetivas:** la frontera entre *regular* y *mala* depende del criterio humano.
- **Desbalance de clases** y datos heterogéneos (imágenes individuales, imágenes con varias frutas, fotos propias).
- **Dos salidas distintas** (clasificación y medición geométrica) que dependen de una segmentación correcta.
- **Restricciones de despliegue:** inferencia en tiempo casi real con cámara y hardware modesto.

## Referencias

[1] N. Otsu, "A threshold selection method from gray-level histograms," *IEEE Trans. Syst., Man, Cybern.*, vol. 9, no. 1, pp. 62–66, 1979.

[2] R. M. Haralick, K. Shanmugam, and I. Dinstein, "Textural features for image classification," *IEEE Trans. Syst., Man, Cybern.*, vol. SMC-3, no. 6, pp. 610–621, 1973.

[3] C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.

[4] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[5] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," *Proc. IEEE*, vol. 86, no. 11, pp. 2278–2324, 1998.

[6] C. Rother, V. Kolmogorov, and A. Blake, "'GrabCut': Interactive foreground extraction using iterated graph cuts," *ACM Trans. Graph.*, vol. 23, no. 3, pp. 309–314, 2004.

[7] T. Ojala, M. Pietikäinen, and T. Mäenpää, "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 24, no. 7, pp. 971–987, 2002.

[8] J. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," in *Advances in Large Margin Classifiers*, MIT Press, 1999, pp. 61–74.

[9] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. 32nd Int. Conf. Mach. Learn. (ICML)*, 2015, pp. 448–456.

[10] M. Lin, Q. Chen, and S. Yan, "Network in network," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2014, arXiv:1312.4400.

[11] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *J. Mach. Learn. Res.*, vol. 15, pp. 1929–1958, 2014.

[12] B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York, NY, USA: Chapman & Hall, 1993.

[13] R. Geirhos *et al.*, "Shortcut learning in deep neural networks," *Nat. Mach. Intell.*, vol. 2, pp. 665–673, 2020.
