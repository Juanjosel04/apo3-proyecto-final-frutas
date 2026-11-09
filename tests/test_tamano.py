import cv2
import numpy as np
import pytest

from src.data.tamano import (
    area_mascara,
    calcular_umbrales,
    categoria_tamano,
    diametro_equivalente,
    diametro_normalizado,
    estimar_factor_escala,
    estimar_tamano,
)


def circulo(radio, lado=400):
    m = np.zeros((lado, lado), np.uint8)
    cv2.circle(m, (lado // 2, lado // 2), radio, 255, -1)
    return m


def test_diametro_de_un_circulo_coincide_con_el_real():
    m = circulo(80)
    d = diametro_equivalente(area_mascara(m))
    assert d == pytest.approx(160, rel=0.01)


def test_diametro_area_cero_y_negativa():
    assert diametro_equivalente(0) == 0
    with pytest.raises(ValueError):
        diametro_equivalente(-1)


def test_diametro_normalizado_usa_el_lado_mayor():
    assert diametro_normalizado(100, (200, 400)) == pytest.approx(0.25)


def test_normalizado_no_depende_de_la_resolucion():
    m1 = circulo(50, lado=200)
    m2 = cv2.resize(m1, (400, 400), interpolation=cv2.INTER_NEAREST)
    d1 = estimar_tamano(m1)["diametro_norm"]
    d2 = estimar_tamano(m2)["diametro_norm"]
    assert d1 == pytest.approx(d2, rel=0.02)


def test_umbrales_y_categorias():
    umbrales = calcular_umbrales(np.arange(1, 101))
    assert umbrales[0] < umbrales[1]
    assert categoria_tamano(10, umbrales) == "pequeno"
    assert categoria_tamano(50, umbrales) == "mediano"
    assert categoria_tamano(90, umbrales) == "grande"


def test_factor_escala_recupera_k():
    d_px = np.array([100.0, 150.0, 200.0])
    assert estimar_factor_escala(d_px, 0.05 * d_px) == pytest.approx(0.05)


def test_estimar_tamano_con_umbrales():
    r = estimar_tamano(circulo(150), umbrales=(0.2, 0.5))
    assert r["categoria"] == "grande"
    assert set(r) == {"area_px", "diametro_px", "diametro_norm", "categoria"}
