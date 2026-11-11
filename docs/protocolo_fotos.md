# Protocolo de recolección y etiquetado de fotos propias

Las fotos propias son el **conjunto de prueba externo** del proyecto: miden si los modelos entrenados con el dataset de Kaggle funcionan con fotos tomadas en Cali con un celular. También sirven para validar el estimador de tamaño contra el diámetro real.

## Meta

- **50–60 fotos en total** (el enunciado pide mínimo 30–50), unas 18–20 por integrante.
- **Frutas:** manzana, banano y naranja (la justificación está en `docs/crisp-dm/02_comprension_datos.md`).
- Repartir más o menos por igual las tres clases de calidad en cada fruta (unas 6 fotos por fruta y clase).

## Definición de las clases de calidad

| Clase | Criterio | Ejemplos |
|---|---|---|
| **buena** | Sin defectos visibles, color y madurez propios para la venta | Manzana firme, de color parejo y sin golpes; banano amarillo sin manchas o con pocas pecas; naranja de color uniforme |
| **regular** | Defecto leve que no afecta el consumo: manchas pequeñas (menos de ~10 % de la superficie visible), golpe leve, sobremaduración | Manzana con un golpe pequeño o la piel algo arrugada; banano con muchas pecas cafés; naranja con una zona golpeada |
| **mala** | Defecto extenso: podredumbre, hongo, golpe fuerte, piel rota o deformación | Manzana con zonas blandas cafés o podrida; banano negro o abierto; naranja con moho |

Si hay duda entre dos clases, se elige la peor y se anota en `notas`.

## Cómo tomar cada foto

1. **Fondo:** una hoja o cartulina blanca, siempre la misma.
2. **Una sola fruta por foto**, centrada y completa (sin cortar bordes).
3. **Distancia fija de unos 30 cm**, desde arriba (vista cenital). Apoyar el celular en un vaso o una caja para que la distancia sea la misma en todas las fotos; esto permite convertir píxeles a centímetros.
4. Luz natural o luz blanca, **sin flash** y sin sombras fuertes.
5. **Medir el diámetro real** con regla o cinta, en centímetros, en la parte más ancha de la fruta vista desde arriba (para el banano, el largo en línea recta de punta a punta) y anotarlo.
6. Opcional: una segunda foto de la misma fruta girada (anotar en `notas` que es la misma fruta).
7. **No fotografiar personas, caras, placas ni marcas.** En la plaza, pedir permiso al vendedor antes de tomar la foto.

## Cómo subir las fotos al repositorio

1. Pasar las fotos del celular a una carpeta del computador, una carpeta por fruta y calidad (por ejemplo `C:\fotos\banano_buena`).
2. Desde la raíz del repo, con el entorno activado, ejecutar:
   ```bash
   python scripts/preparar_fotos.py --origen "C:/fotos/banano_buena" --fruta banano --calidad buena --iniciales jjl
   ```
   El script corrige la orientación, reduce la foto a 1024 px por el lado mayor, **borra los metadatos (incluido el GPS)** y la guarda como `data/propias/raw/<fruta>/<calidad>/<iniciales>_<fruta>_<nn>.jpg`.
3. Agregar una fila por foto en `data/propias/etiquetas.csv`:

   | Columna | Valores |
   |---|---|
   | `archivo` | nombre que dio el script, p. ej. `jjl_banano_07.jpg` |
   | `fruta` | `manzana`, `banano`, `naranja` |
   | `calidad` | `buena`, `regular`, `mala` |
   | `diametro_cm` | número con punto decimal, p. ej. `6.5` |
   | `tamano` | `pequeno`, `mediano`, `grande` (a criterio del autor, comparando con las demás frutas de su tipo) |
   | `autor` | iniciales, p. ej. `jjl` |
   | `fecha` | `AAAA-MM-DD` |
   | `lugar` | `casa`, `supermercado`, `plaza` |
   | `notas` | opcional |

4. Validar antes de hacer commit:
   ```bash
   python scripts/validar_etiquetas.py
   ```
5. Hacer commit y push desde la cuenta propia de GitHub.

## Compartir con el curso

Además del repo, la carpeta de fotos y el CSV se suben al mecanismo común que acuerde el curso [PENDIENTE: enlace], con el mismo formato de etiquetas.
