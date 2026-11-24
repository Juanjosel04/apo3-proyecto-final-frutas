import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validar_etiquetas import validar  # noqa: E402

from src.config import COLUMNAS_ETIQUETAS  # noqa: E402


def _preparar(tmp_path, filas):
    raw = tmp_path / "raw"
    foto = raw / "manzana" / "buena" / "jjl_manzana_01.jpg"
    foto.parent.mkdir(parents=True)
    foto.write_bytes(b"x")
    csv = tmp_path / "etiquetas.csv"
    lineas = [",".join(COLUMNAS_ETIQUETAS)] + filas
    csv.write_text("\n".join(lineas) + "\n", encoding="utf8")
    return csv, raw


FILA = "jjl_manzana_01.jpg,manzana,buena,{d},mediano,jjl,2026-11-23,casa,nota"


def test_diametro_vacio_es_aviso(tmp_path):
    errores, avisos = validar(*_preparar(tmp_path, [FILA.format(d="")]))
    assert errores == []
    assert len(avisos) == 1 and "sin diámetro" in avisos[0]


def test_diametro_valido_sin_avisos(tmp_path):
    errores, avisos = validar(*_preparar(tmp_path, [FILA.format(d="7.4")]))
    assert errores == [] and avisos == []


@pytest.mark.parametrize("d", ["-3", "0", "abc"])
def test_diametro_invalido_es_error(tmp_path, d):
    errores, _ = validar(*_preparar(tmp_path, [FILA.format(d=d)]))
    assert any("diámetro inválido" in e for e in errores)


def test_fruta_fuera_del_proyecto(tmp_path):
    fila = FILA.format(d="").replace(",manzana,", ",guayaba,")
    errores, _ = validar(*_preparar(tmp_path, [fila]))
    assert any("fruta inválida" in e for e in errores)


def test_foto_sin_etiqueta(tmp_path):
    csv, raw = _preparar(tmp_path, [])
    errores, _ = validar(csv, raw)
    assert any("sin etiqueta" in e for e in errores)
