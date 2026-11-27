"""Estilos de la app: tema oscuro con paneles de vidrio translúcido.

Los paneles usan ``backdrop-filter`` (desenfoque y saturación del fondo), un borde
claro de 1 px y un brillo interior arriba, sobre un fondo negro con luces suaves
de color para que el vidrio se note.
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --fondo: #000000;
  --vidrio: rgba(255, 255, 255, 0.07);
  --vidrio-alto: rgba(255, 255, 255, 0.13);
  --borde: rgba(255, 255, 255, 0.14);
  --brillo: inset 0 1px 0 rgba(255, 255, 255, 0.22), inset 0 -1px 0 rgba(255, 255, 255, 0.04);
  --sombra: 0 18px 50px rgba(0, 0, 0, 0.55);
  --blur: blur(28px) saturate(180%);
  --texto: #f5f5f7;
  --suave: rgba(235, 235, 245, 0.60);
  --tenue: rgba(235, 235, 245, 0.30);
  --azul: #0a84ff;
  --buena: #30d158;
  --regular: #ff9f0a;
  --mala: #ff453a;
  --fuente: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI Variable Display", "Inter", system-ui, sans-serif;
}

html, body, .stApp, [class*="css"], button, input { font-family: var(--fuente) !important; }
.stApp { color: var(--texto); background: var(--fondo); }
.stApp::before {
  content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(700px 500px at 12% 8%, rgba(10, 132, 255, 0.22), transparent 65%),
    radial-gradient(650px 480px at 92% 18%, rgba(191, 90, 242, 0.18), transparent 65%),
    radial-gradient(800px 520px at 60% 105%, rgba(48, 209, 88, 0.14), transparent 65%);
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.4rem; max-width: 1200px; position: relative; z-index: 1; }

/* Vidrio: mezcla común a todos los paneles */
.vidrio, [data-testid="stSidebar"] > div:first-child,
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]),
[data-testid="stFileUploaderDropzone"], [data-testid="stExpander"] details {
  background: var(--vidrio) !important;
  backdrop-filter: var(--blur); -webkit-backdrop-filter: var(--blur);
  border: 1px solid var(--borde) !important;
  box-shadow: var(--sombra), var(--brillo);
}

/* Barra lateral flotante */
[data-testid="stSidebar"] { background: transparent; border: none; }
[data-testid="stSidebar"] > div:first-child {
  margin: 14px; width: calc(100% - 28px); height: calc(100% - 28px); border-radius: 28px;
  overflow: hidden !important; scrollbar-width: none;
}
[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar { display: none; }
/* Sin línea ni control para cambiar el ancho: el panel es fijo */
[data-testid="stSidebar"] > div:last-child:not(:first-child) { display: none; }
[data-testid="stSidebarHeader"] { height: auto; padding: 14px 16px 0; }
[data-testid="stLogoSpacer"] { display: none; }
[data-testid="stSidebarUserContent"] { padding-top: 0; padding-bottom: 12px; }
.titulo-lateral { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; margin: 0 0 6px; }

/* Cabecera estilo título grande de iOS */
.cabecera { padding: 4px 2px 14px; }
.cabecera h1 { font-size: 2.6rem; font-weight: 700; letter-spacing: -0.035em; margin: 0; color: var(--texto); }
.cabecera p { color: var(--suave); margin: 4px 0 0; font-size: 1rem; }

/* Controles segmentados como cápsulas de vidrio */
[data-testid="stButtonGroup"] > div {
  background: var(--vidrio); border: 1px solid var(--borde); border-radius: 999px;
  padding: 4px; gap: 2px; width: fit-content; flex-wrap: wrap;
  backdrop-filter: var(--blur); -webkit-backdrop-filter: var(--blur); box-shadow: var(--brillo);
}
[data-testid="stButtonGroup"] button {
  background: transparent !important; border: none !important; border-radius: 999px !important;
  color: var(--suave) !important; padding: 6px 16px !important; min-height: 0 !important;
  font-weight: 600; transition: background .2s, color .2s;
}
[data-testid="stButtonGroup"] button:hover { color: var(--texto) !important; background: rgba(255,255,255,0.06) !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"] {
  background: var(--vidrio-alto) !important; color: var(--azul) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.25), 0 4px 14px rgba(0,0,0,0.35);
}
/* En la barra lateral las opciones van en una rejilla de 2 columnas dentro del marco */
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stButtonGroup"]),
[data-testid="stSidebar"] [data-testid="stButtonGroup"] { width: 100% !important; }
[data-testid="stSidebar"] [data-testid="stButtonGroup"] > div {
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); width: 100%; max-width: none;
  border-radius: 22px; padding: 5px; gap: 4px;
}
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button {
  width: 100%; border-radius: 17px !important; justify-content: center; padding: 7px 10px !important;
}
[data-testid="stWidgetLabel"] p { color: var(--suave); font-weight: 600; font-size: 0.82rem; }

/* Carga de archivo y cámara */
[data-testid="stFileUploaderDropzone"] { border-radius: 24px; border-style: dashed !important; }
[data-testid="stFileUploaderDropzone"] * { color: var(--suave); }
[data-testid="stCameraInput"] > div, [data-testid="stCameraInput"] video { border-radius: 24px; overflow: hidden; }

/* Contenedores con borde */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) { border-radius: 28px; padding: 6px; }
[data-testid="stImage"] img { border-radius: 18px; }
[data-testid="stImageCaption"], [data-testid="caption"] { color: var(--suave) !important; }

/* Botones */
.stButton > button, .stDownloadButton > button {
  border-radius: 999px; border: 1px solid var(--borde); color: var(--texto);
  background: var(--vidrio-alto); box-shadow: var(--brillo);
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--azul); color: var(--azul); }

[data-testid="stExpander"] details { border-radius: 22px; }
iframe[title*="streamlit_webrtc"] { border-radius: 24px; }

/* Tarjetas propias */
.vidrio { border-radius: 26px; padding: 18px 22px; margin-bottom: 14px; }
.heroe { background: linear-gradient(160deg, rgba(255,255,255,0.12), rgba(255,255,255,0.04)) !important; }
.etiqueta { font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--suave); font-weight: 600; }
.valor { font-size: 2.6rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1.1; margin-top: 4px; }
.valor.chico { font-size: 1.35rem; color: var(--texto); }
.detalle { color: var(--suave); font-size: 0.95rem; margin-top: 4px; }
.barras { margin-top: 12px; }
.fila { display: flex; justify-content: space-between; font-size: 0.92rem; color: var(--texto); }
.barra { height: 8px; border-radius: 999px; background: rgba(255,255,255,0.10); overflow: hidden; margin: 6px 0 12px; }
.barra > span { display: block; height: 100%; border-radius: 999px; transition: width .4s ease; }
.lista { padding: 4px 22px; }
.item {
  display: flex; justify-content: space-between; align-items: baseline; gap: 12px;
  padding: 12px 0; border-bottom: 0.5px solid rgba(255,255,255,0.12); font-size: 0.95rem;
}
.item:last-child { border-bottom: none; }
.item > span:first-child { color: var(--texto); }
.item > span:last-child { color: var(--suave); text-align: right; }
.item small { color: var(--tenue); }
.lista.lateral { padding: 2px 16px; margin: 10px 0 14px; border-radius: 20px; }
.lista.lateral .item { font-size: 0.82rem; padding: 9px 0; }
.vacio { padding: 26px 24px; }
.aviso {
  background: rgba(255, 159, 10, 0.14); border: 1px solid rgba(255, 159, 10, 0.35);
  border-radius: 20px; padding: 12px 16px; color: #ffd60a; font-size: 0.92rem; margin-bottom: 14px;
  backdrop-filter: var(--blur); -webkit-backdrop-filter: var(--blur);
}
.nota { color: var(--suave); font-size: 0.82rem; line-height: 1.5; }
</style>
"""

COLOR_CALIDAD = {"buena": "var(--buena)", "regular": "var(--regular)", "mala": "var(--mala)"}
# RGB para dibujar el contorno sobre la foto
COLOR_CONTORNO = {"buena": (48, 209, 88), "regular": (255, 159, 10), "mala": (255, 69, 58)}
