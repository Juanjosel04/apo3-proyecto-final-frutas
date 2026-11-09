import cv2
import numpy as np
import pytest

from src.data.segmentacion import (
    aplicar_mascara,
    color_fondo,
    mascara_fruta,
    recortar_fruta,
    separar_frutas,
)


def imagen_sintetica(centros, radio=60, fondo=(235, 235, 235), color=(30, 120, 220), tam=(400, 600)):
    """Fondo claro con círculos de color (frutas simuladas)."""
    img = np.full((*tam, 3), fondo, np.uint8)
    for c in centros:
        cv2.circle(img, c, radio, color, -1)
    ruido = np.random.default_rng(42).normal(0, 4, img.shape)
    return np.clip(img + ruido, 0, 255).astype(np.uint8)


def iou(a, b):
    a, b = a > 0, b > 0
    return (a & b).sum() / (a | b).sum()


def test_color_fondo_es_el_del_borde():
    img = imagen_sintetica([(300, 200)])
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    esperado = cv2.cvtColor(np.uint8([[[235, 235, 235]]]), cv2.COLOR_BGR2LAB)[0, 0]
    assert np.allclose(color_fondo(lab), esperado, atol=3)


def test_mascara_de_una_fruta():
    img = imagen_sintetica([(300, 200)], radio=90)
    real = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(real, (300, 200), 90, 255, -1)
    assert iou(mascara_fruta(img), real) > 0.95


def test_mascara_con_fondo_oscuro():
    img = imagen_sintetica([(300, 200)], radio=90, fondo=(20, 20, 20), color=(40, 200, 90))
    real = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(real, (300, 200), 90, 255, -1)
    assert iou(mascara_fruta(img), real) > 0.95


def test_rellena_brillos_internos():
    img = imagen_sintetica([(300, 200)], radio=90)
    cv2.circle(img, (290, 190), 15, (235, 235, 235), -1)  # brillo del color del fondo
    m = mascara_fruta(img)
    assert m[190, 290] == 255


@pytest.mark.parametrize("centros", [
    [(120, 120), (300, 120), (480, 120)],
    [(120, 100), (480, 100), (120, 300), (480, 300)],
])
def test_separar_frutas_separadas(centros):
    img = imagen_sintetica(centros, radio=50)
    assert len(separar_frutas(img)) == len(centros)


def test_separar_frutas_que_se_tocan():
    img = imagen_sintetica([(240, 200), (355, 200)], radio=60)  # se solapan unos píxeles
    frutas = separar_frutas(img, separar_contacto=True)
    assert len(frutas) == 2


def test_recortes_contienen_la_fruta():
    img = imagen_sintetica([(150, 200), (450, 200)], radio=70)
    for f in separar_frutas(img):
        assert f.imagen.shape[:2] == f.mascara.shape
        assert f.area == pytest.approx(np.pi * 70 ** 2, rel=0.1)


def test_aplicar_y_recortar():
    img = imagen_sintetica([(300, 200)], radio=80)
    m = mascara_fruta(img)
    solo = aplicar_mascara(img, m)
    assert solo[5, 5].sum() == 0
    rec, mrec = recortar_fruta(img, m)
    assert rec.shape[0] < img.shape[0] and mrec.shape == rec.shape[:2]


@pytest.mark.parametrize("metodo", ["otsu", "grabcut"])
def test_ambos_metodos_con_fondo_uniforme(metodo):
    img = imagen_sintetica([(300, 200)], radio=90)
    real = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(real, (300, 200), 90, 255, -1)
    assert iou(mascara_fruta(img, metodo), real) > 0.93


def test_grabcut_con_fondo_texturizado():
    # Fondo gris con baldosas y manchas (poco saturado) y fruta naranja saturada
    rng = np.random.default_rng(0)
    img = np.full((400, 600, 3), 150, np.uint8)
    for x in range(0, 600, 60):
        cv2.line(img, (x, 0), (x, 400), (90, 90, 90), 3)
    for _ in range(300):
        c = tuple(int(v) for v in rng.integers(0, [600, 400]))
        g = int(rng.integers(60, 230))
        cv2.circle(img, c, int(rng.integers(2, 8)), (g, g, g), -1)
    cv2.circle(img, (300, 200), 90, (30, 140, 240), -1)
    real = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(real, (300, 200), 90, 255, -1)
    assert iou(mascara_fruta(img, "grabcut"), real) > 0.9


def test_metodo_invalido():
    with pytest.raises(ValueError):
        mascara_fruta(imagen_sintetica([(300, 200)]), "kmeans")
