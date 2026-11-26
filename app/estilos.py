"""Estilos de la app (efecto de vidrio esmerilado sobre un fondo con degradados)."""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --vidrio: rgba(255, 255, 255, 0.42);
  --vidrio-fuerte: rgba(255, 255, 255, 0.62);
  --borde: rgba(255, 255, 255, 0.70);
  --sombra: 0 10px 40px rgba(31, 41, 55, 0.10);
  --brillo: inset 0 1px 0 rgba(255, 255, 255, 0.85);
  --texto: #1f2937;
  --suave: #4b5563;
  --buena: #16a34a;
  --mala: #dc2626;
  --regular: #d97706;
}

html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; color: var(--texto); }

.stApp {
  background:
    radial-gradient(1100px 600px at 8% 0%, rgba(253, 224, 71, 0.45), transparent 60%),
    radial-gradient(900px 600px at 95% 10%, rgba(248, 113, 113, 0.35), transparent 60%),
    radial-gradient(900px 700px at 50% 100%, rgba(74, 222, 128, 0.35), transparent 60%),
    linear-gradient(135deg, #fdf6ec 0%, #eef7f1 100%);
  background-attachment: fixed;
}

header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }

/* Paneles de vidrio */
[data-testid="stSidebar"] > div:first-child,
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {
  background: var(--vidrio);
  backdrop-filter: blur(22px) saturate(170%);
  -webkit-backdrop-filter: blur(22px) saturate(170%);
  border: 1px solid var(--borde) !important;
  border-radius: 24px;
  box-shadow: var(--sombra), var(--brillo);
}
[data-testid="stSidebar"] { background: transparent; }
[data-testid="stSidebar"] > div:first-child { margin: 12px; height: calc(100% - 24px); }

[data-testid="stFileUploaderDropzone"], [data-testid="stCameraInput"] > div {
  background: var(--vidrio-fuerte);
  border: 1.5px dashed rgba(31, 41, 55, 0.18);
  border-radius: 20px;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 6px; background: var(--vidrio); padding: 6px; border-radius: 999px;
  border: 1px solid var(--borde); width: fit-content; box-shadow: var(--brillo);
}
.stTabs [data-baseweb="tab"] { border-radius: 999px; padding: 6px 18px; height: auto; }
.stTabs [aria-selected="true"] { background: var(--vidrio-fuerte); box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

.stButton > button, .stDownloadButton > button {
  border-radius: 999px; border: 1px solid var(--borde);
  background: var(--vidrio-fuerte); backdrop-filter: blur(12px); box-shadow: var(--brillo);
}

/* Componentes propios */
.cabecera { padding: 6px 4px 18px; }
.cabecera h1 { font-size: 2.3rem; font-weight: 800; letter-spacing: -0.03em; margin: 0; }
.cabecera p { color: var(--suave); margin: 6px 0 0; font-size: 1.02rem; }

.tarjeta {
  background: var(--vidrio); backdrop-filter: blur(22px) saturate(170%);
  -webkit-backdrop-filter: blur(22px) saturate(170%);
  border: 1px solid var(--borde); border-radius: 22px; padding: 18px 20px;
  box-shadow: var(--sombra), var(--brillo); margin-bottom: 14px;
}
.etiqueta { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--suave); font-weight: 600; }
.valor { font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.15; margin-top: 4px; }
.detalle { color: var(--suave); font-size: 0.9rem; margin-top: 4px; }
.pastilla {
  display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 0.85rem;
  background: rgba(255,255,255,0.7); border: 1px solid var(--borde);
}
.barra { height: 10px; border-radius: 999px; background: rgba(31,41,55,0.08); overflow: hidden; margin: 6px 0 10px; }
.barra > span { display: block; height: 100%; border-radius: 999px; }
.fila { display: flex; justify-content: space-between; font-size: 0.9rem; }
.aviso {
  background: rgba(254, 243, 199, 0.75); border: 1px solid rgba(217, 119, 6, 0.35);
  border-radius: 18px; padding: 12px 16px; color: #92400e; font-size: 0.92rem; margin-bottom: 14px;
}
.nota { color: var(--suave); font-size: 0.82rem; }
.valor.chico { font-size: 1.45rem; }
.tarjeta.doble { display: flex; flex-wrap: wrap; gap: 18px; }
.tarjeta.doble > div { flex: 1 1 160px; min-width: 0; }
</style>
"""

COLOR_CALIDAD = {"buena": "var(--buena)", "regular": "var(--regular)", "mala": "var(--mala)"}
