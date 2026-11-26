from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
APP = Path(__file__).resolve().parents[1] / "app" / "app_streamlit.py"


def test_app_carga_sin_errores():
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    textos = " ".join(m.value for m in at.markdown)
    assert "Calidad y tamaño de frutas" in textos
    assert "Sube una foto" in textos
