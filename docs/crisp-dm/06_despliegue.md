# Fase 6 — Despliegue

## 6.1 Aplicación

App web en Streamlit (`app/app_streamlit.py`), que se ejecuta localmente:

```bash
streamlit run app/app_streamlit.py
```

| Elemento | Detalle |
|---|---|
| Entrada | Foto subida (JPG, PNG o WEBP, hasta 20 MB) o tomada con la cámara (`st.camera_input`) |
| Preprocesamiento | Corrección de orientación EXIF y reducción del lado mayor a 1024 px |
| Fruta | Detectada por el Random Forest o indicada por el usuario |
| Calidad | SVM `global_fruta` (por defecto) o CNN `doble`, con la probabilidad de cada clase |
| Tamaño | Categoría relativa (pequeño, mediano o grande) y diámetro normalizado |
| Revisión humana | Aviso cuando la confianza es menor a 0,70 o la máscara ocupa menos del 2 % o más del 98 % de la imagen |
| Trazabilidad | Contorno de la máscara, recorte que recibe el modelo, tiempo de proceso e historial de la sesión descargable en CSV |

La lógica está en `src/inference/predecir.py` (clase `Clasificador`), que repite los pasos del entrenamiento: segmentación, recorte, características, fruta, calidad y tamaño. Los modelos se cargan una vez por sesión (`st.cache_resource`). Los modelos finales están versionados en `experiments/checkpoints/`:
- `clasico_global_fruta_svm.joblib`
- `clasificador_fruta_rf.joblib`
- `cnn_calidad_doble.keras`

## 6.2 Verificación

- `tests/test_predecir.py`: predicción completa sobre una foto propia, fruta indicada, entradas inválidas y aviso de revisión.
- `tests/test_app.py`: la app carga sin errores (`streamlit.testing.v1.AppTest`).
- Prueba manual con fotos propias en el navegador: unos 150–250 ms por imagen en CPU, dentro del criterio de menos de 2 s.

## 6.3 Limitaciones del despliegue

- Los modelos son de **dos clases** (`buena`, `mala`). La clase `regular` requiere etiquetar datos y reentrenar (fase 3, sección 3.5).
- La **generalización a fotos reales es baja** ([fase 5](05_evaluacion.md#56-evaluación-externa-con-fotos-propias)). Por eso la app muestra la confianza, el recorte y el aviso de revisión: es una herramienta de apoyo, no un reemplazo del seleccionador.
- El tamaño es **relativo al encuadre**. Para convertirlo a centímetros hace falta una distancia de cámara fija y el factor de escala de `src/data/tamano.py` (`estimar_factor_escala`).
- Supone **una sola fruta** sobre un fondo claro y liso.
