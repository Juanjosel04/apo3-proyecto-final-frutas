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

es decir, una apertura (quita ruido pequeño) seguida de un cierre (rellena huecos como el pedúnculo o los brillos). Finalmente se conserva el contorno conexo de mayor área y se rellena su interior. En las imágenes de `mixed_quality` (varias frutas por foto) se conservan todos los componentes conexos con área mayor a un umbral relativo $A_{\min} = \alpha\,HW$, y cada uno se recorta por su caja envolvente.

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

Con las fotos propias, tomadas a una distancia fija de unos 30 cm, existe un factor de escala aproximadamente constante $k$ (cm por píxel), y se espera $d_{cm} \approx k\, d_{px}$. Ese factor se estima por mínimos cuadrados con el diámetro medido con regla, $\hat{k} = \sum_i d_{px,i}\,d_{cm,i} / \sum_i d_{px,i}^2$, y el error del estimador se reporta como MAE y error relativo.

**Limitación conocida:** en el dataset de Kaggle no se conoce la distancia de la cámara, así que $\hat{d}$ mide el tamaño *relativo dentro de la foto*, no el tamaño físico. Esto se discute en el informe.

## 4. Características manuales (modelos clásicos)

Cada imagen segmentada se resume en un vector $\boldsymbol{\phi}(\mathbf{x}, M) \in \mathbb{R}^p$ calculado solo sobre los píxeles de la fruta:

| Grupo | Características | Justificación |
|---|---|---|
| Color | Histograma de H y S (bins), momentos de color por canal: media $\mu$, desviación $\sigma$ y asimetría $\gamma = \mathbb{E}[(X-\mu)^3]/\sigma^3$ | La madurez y la podredumbre cambian el tono y la saturación |
| Textura | GLCM: contraste, homogeneidad, energía, correlación [2]; o histograma LBP | Moho y arrugas aumentan el contraste local |
| Forma | Área, perímetro $P$, circularidad $4\pi A/P^2$, excentricidad de la elipse equivalente | Frutas deformadas o aplastadas |
| Defectos | Proporción de píxeles oscuros: $\frac{1}{A}\sum M(i,j)\,[V(i,j) < \tau]$ | Manchas y golpes son más oscuros que la piel sana |

Antes de entrenar se estandariza cada característica con la media y la desviación del conjunto de entrenamiento:

$$
z_j = \frac{\phi_j - \mu_j^{train}}{\sigma_j^{train}}.
$$

Esto es necesario para SVM con kernel RBF y KNN, que dependen de distancias euclidianas; sin estandarizar, la característica de mayor rango (por ejemplo el área) dominaría la distancia.

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

### 5.3 Random Forest [4]
Ensamble de $B$ árboles entrenados sobre muestras bootstrap y subconjuntos aleatorios de características. Cada nodo elige la partición que más reduce la impureza de Gini

$$
G(t) = 1 - \sum_{k=1}^{K} p_{k}(t)^2 .
$$

La predicción es el voto mayoritario. El promedio de árboles poco correlacionados reduce la varianza sin aumentar mucho el sesgo.

### 5.4 CNN simple entrenada desde cero [5]
Bloques convolución + ReLU + max-pooling, seguidos de capas densas y softmax. Una capa convolucional calcula

$$
(\mathbf{X} * \mathbf{K})(i,j) = \sum_{m}\sum_{n}\sum_{c} \mathbf{X}(i+m, j+n, c)\,\mathbf{K}(m,n,c) + b,
$$

y la salida es $\hat{\mathbf{p}} = \mathrm{softmax}(\mathbf{o})$, con $\mathrm{softmax}(\mathbf{o})_k = e^{o_k}/\sum_j e^{o_j}$. Se minimiza la entropía cruzada ponderada por clase:

$$
\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} w_k\, y_{ik}\log \hat{p}_{ik}, \qquad w_k = \frac{n}{K\, n_k},
$$

donde $n_k$ es el número de ejemplos de la clase $k$; así las clases minoritarias pesan más. Se usan aumentación de datos, *dropout* y *early stopping* sobre la pérdida de validación.

## 6. Evaluación

- Partición estratificada entrenamiento / validación / prueba con semilla 42, y las fotos propias como **prueba externa** (cambio de dominio).
- Ajuste de hiperparámetros con validación cruzada estratificada de $k$ pliegues (GridSearchCV).
- Métricas por clase: $\mathrm{Prec}_k = \frac{TP_k}{TP_k + FP_k}$, $\mathrm{Rec}_k = \frac{TP_k}{TP_k + FN_k}$, $F1_k = \frac{2\,\mathrm{Prec}_k\mathrm{Rec}_k}{\mathrm{Prec}_k + \mathrm{Rec}_k}$, y **F1 macro** $= \frac{1}{K}\sum_k F1_k$ como métrica principal, porque no se deja dominar por la clase mayoritaria.
- Matriz de confusión y análisis de errores. En calidad, confundir *mala* con *buena* es el error más costoso (una fruta dañada llega al consumidor), por eso se reporta aparte el recall de *mala*.
- Sobreajuste: brecha entre métricas de entrenamiento y validación, y curvas de pérdida de la CNN.

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
