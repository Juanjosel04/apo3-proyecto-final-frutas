# Guía de instalación

## Requisitos

- Windows 10/11, macOS o Linux. Probado en Windows 11.
- Python 3.12.
- Unos 4 GB libres para el dataset de Kaggle (3,3 GB). El dataset solo hace falta para reentrenar; la app y las pruebas funcionan sin él.
- Cámara web, opcional, para tomar la foto desde la app.

## Instalación paso a paso

```bash
git clone https://github.com/Juanjosel04/apo3-proyecto-final-frutas.git
cd apo3-proyecto-final-frutas
py -3.12 -m venv .venv
.venv\Scripts\activate            # en macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest
```

`pytest` debe terminar sin fallos. Los modelos finales vienen en `experiments/checkpoints/`, así que no hace falta entrenar para usar la app.

## Usar la app

```bash
streamlit run app/app_streamlit.py
```

Se abre en `http://localhost:8501`. Se sube una foto o se usa la cámara, y la app muestra la fruta, la calidad con su confianza y el tamaño relativo. La cámara del navegador funciona en `localhost`; desde otra máquina el navegador exige HTTPS.

## Descarga del dataset (solo para reentrenar)

```bash
python -m src.data.descargar_datos
```

Usa `kagglehub` y no pide token para este dataset. Si Kaggle lo pidiera, se crea `kaggle.json` en la cuenta de Kaggle y se guarda en `~/.kaggle/` (nunca dentro del repositorio).

## Reentrenar desde cero

```bash
python -m src.data.preprocesamiento
python -m src.data.recortes
python -m src.features.extraccion
python -m src.training.entrenar sesgo
python -m src.training.entrenar clasicos
python -m src.training.entrenar cnn --variante doble
```

En CPU los modelos clásicos tardan unos 30 minutos y la CNN unas 2 horas. Si la CNN se interrumpe, el mismo comando continúa desde la última época.

## Notebooks

Se abren en orden desde `notebooks/` (`jupyter notebook`). Para ejecutarlos desde la terminal, el entorno debe estar activado, porque el kernel usa el `python` del `PATH`.

## Problemas frecuentes

| Problema | Solución |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Ejecutar los comandos desde la raíz del repositorio |
| `No module named 'cv2'` dentro de un notebook | Activar `.venv` antes de abrir Jupyter |
| Advertencia al cargar los `.joblib` | Instalar `scikit-learn` 1.9, la versión con la que se guardaron (está en `requirements.txt`) |
| La cámara no aparece en la app | Dar permiso de cámara al navegador y abrir la app en `localhost` |
| "No se encontró ninguna fruta" | Tomar la foto desde arriba, con una sola fruta sobre un fondo claro y liso |
