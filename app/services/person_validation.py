from __future__ import annotations

from typing import Any

from app.domain.models.camera_event import BoundingBox


def _obter_limites(caixa: Any) -> tuple[int, int, int, int]:
    x1 = int(caixa.x)
    y1 = int(caixa.y)
    x2 = x1 + int(caixa.largura)
    y2 = y1 + int(caixa.altura)

    return x1, y1, x2, y2


def calcular_metricas_sobreposicao(caixa_a: Any, caixa_b: Any) -> tuple[float, float]:
    """
    Retorna:

    - IoU entre as caixas;
    - cobertura da menor caixa pela interseção.
    """
    ax1, ay1, ax2, ay2 = _obter_limites(caixa_a)
    bx1, by1, bx2, by2 = _obter_limites(caixa_b)

    intersecao_x1 = max(ax1, bx1)
    intersecao_y1 = max(ay1, by1)
    intersecao_x2 = min(ax2, bx2)
    intersecao_y2 = min(ay2, by2)

    largura_intersecao = max(0, intersecao_x2 - intersecao_x1)
    altura_intersecao = max(0, intersecao_y2 - intersecao_y1)
    area_intersecao = largura_intersecao * altura_intersecao

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    area_uniao = area_a + area_b - area_intersecao

    iou = area_intersecao / area_uniao if area_uniao > 0 else 0.0
    cobertura_menor = area_intersecao / min(area_a, area_b)

    return iou, cobertura_menor


def validar_boxes_camera_com_yolo(
    caixas_camera: list[BoundingBox],
    caixas_yolo: list[BoundingBox],
    iou_minimo: float = 0.12,
    cobertura_minima: float = 0.50,
) -> list[BoundingBox]:
    """
    Mantém somente as caixas da câmera que possuem correspondência
    espacial com uma pessoa detectada pelo YOLO.

    A caixa original da câmera é preservada para que o restante do
    pipeline continue funcionando sem alterações.
    """
    if not caixas_camera:
        return []

    if not caixas_yolo:
        return []

    caixas_validadas: list[BoundingBox] = []

    for caixa_camera in caixas_camera:
        caixa_confirmada = False

        for caixa_yolo in caixas_yolo:
            iou, cobertura = calcular_metricas_sobreposicao(caixa_camera, caixa_yolo)
            if iou >= iou_minimo or cobertura >= cobertura_minima:
                caixa_confirmada = True
                break

        if caixa_confirmada:
            caixas_validadas.append(caixa_camera)

    return caixas_validadas
