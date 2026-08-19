"""
Seleção e validação das caixas de pessoa vindas do evento da câmera.

Extraído de CameraEventPipeline: aqui vive tudo que decide "quais
caixas são pessoas de verdade" antes de seguir para rastreamento.
"""

import asyncio
import logging

from app.domain.models.camera_event import BoundingBox, CameraEvent
from app.services.person_detector_yolo import detectar_pessoas_yolo
from app.services.person_validation import (
    calcular_metricas_sobreposicao,
    validar_boxes_camera_com_yolo,
)

logger = logging.getLogger(__name__)


def selecionar_boxes_pessoas(evento: CameraEvent) -> list[BoundingBox]:
    """Remove caixas inválidas, duplicadas e a caixa da imagem inteira."""
    if evento.imagem is None:
        return []

    largura_imagem = evento.imagem.largura
    altura_imagem = evento.imagem.altura
    if not largura_imagem or not altura_imagem:
        return []

    area_imagem = largura_imagem * altura_imagem
    assinaturas: set[tuple[int, int, int, int]] = set()
    caixas_validas: list[BoundingBox] = []

    for caixa in evento.bounding_boxes or []:
        if caixa.largura <= 0 or caixa.altura <= 0:
            continue

        x1 = max(0, min(caixa.x, largura_imagem - 1))
        y1 = max(0, min(caixa.y, altura_imagem - 1))
        x2 = max(x1 + 1, min(caixa.x2, largura_imagem))
        y2 = max(y1 + 1, min(caixa.y2, altura_imagem))
        largura = x2 - x1
        altura = y2 - y1
        percentual = largura * altura / area_imagem * 100

        if percentual >= 60 or percentual < 0.05:
            continue

        assinatura = (x1, y1, x2, y2)
        if assinatura in assinaturas:
            continue
        assinaturas.add(assinatura)

        mesma_caixa = (
            x1 == caixa.x
            and y1 == caixa.y
            and largura == caixa.largura
            and altura == caixa.altura
        )
        if mesma_caixa:
            caixa_ajustada = caixa
        else:
            caixa_ajustada = caixa.model_copy(
                update={
                    "x": x1,
                    "y": y1,
                    "largura": largura,
                    "altura": altura,
                    "x2": x2,
                    "y2": y2,
                }
            )

        caixas_validas.append(caixa_ajustada)

    if not caixas_validas and evento.bounding_box_escolhida is not None:
        caixas_validas.append(evento.bounding_box_escolhida)

    caixas_validas.sort(
        key=lambda caixa: (caixa.x + caixa.largura / 2, caixa.y + caixa.altura / 2)
    )
    return caixas_validas


async def validar_com_yolo(
    evento: CameraEvent,
    bounding_boxes_camera: list[BoundingBox],
    pacote_id: str,
) -> tuple[list[BoundingBox], set[int]]:
    """
    Roda o YOLO sobre a imagem para confirmar quais caixas da câmera são
    pessoas de verdade.

    Devolve todas as pessoas detectadas pelo YOLO (para
    tracker/painel/face) e o conjunto de índices, dentro dessa lista,
    que correspondem à caixa que efetivamente gerou o evento de alarme
    da câmera.

    Se o YOLO não puder rodar (sem imagem salva ou falha na inferência),
    usa as caixas da câmera como estão, todas como alvo de alerta.
    """
    bounding_boxes_yolo: list[BoundingBox] = []
    yolo_executado = False
    caminho_original = evento.imagem.caminho_original if evento.imagem else None

    if caminho_original:
        try:
            bounding_boxes_yolo = await asyncio.to_thread(
                detectar_pessoas_yolo, caminho_original
            )
            yolo_executado = True
        except Exception:
            logger.exception(
                "[%s] Falha ao validar pessoas com YOLO; usando "
                "temporariamente as caixas da câmera",
                pacote_id,
            )
    else:
        logger.warning(
            "[%s] Imagem sem caminho original; validação YOLO não pôde ser executada",
            pacote_id,
        )

    if not yolo_executado:
        return bounding_boxes_camera, set(range(len(bounding_boxes_camera)))

    bounding_boxes_evento = validar_boxes_camera_com_yolo(
        caixas_camera=bounding_boxes_camera,
        caixas_yolo=bounding_boxes_yolo,
    )

    quantidade_rejeitada = len(bounding_boxes_camera) - len(bounding_boxes_evento)
    logger.info(
        "[%s] Validação YOLO: camera=%s yolo=%s validadas=%s rejeitadas=%s",
        pacote_id,
        len(bounding_boxes_camera),
        len(bounding_boxes_yolo),
        len(bounding_boxes_evento),
        quantidade_rejeitada,
    )
    if quantidade_rejeitada > 0:
        logger.info("[%s] Possível falso positivo rejeitado pelo YOLO", pacote_id)

    if not bounding_boxes_evento:
        return [], set()

    # Todas as pessoas detectadas pelo YOLO seguem para tracker/painel/face.
    bounding_boxes = bounding_boxes_yolo
    indices_alerta_evento = _localizar_indices_alerta(bounding_boxes_evento, bounding_boxes)

    logger.info(
        "[%s] Pessoas da cena via YOLO: pessoas=%s alvos_evento=%s",
        pacote_id,
        len(bounding_boxes),
        sorted(indices_alerta_evento),
    )
    return bounding_boxes, indices_alerta_evento


def _localizar_indices_alerta(
    bounding_boxes_evento: list[BoundingBox],
    bounding_boxes_yolo: list[BoundingBox],
) -> set[int]:
    """
    Descobre qual pessoa detectada pelo YOLO corresponde à box que
    realmente gerou o evento informado pela câmera, usando a maior
    sobreposição (IoU) entre as caixas.
    """
    indices_alerta_evento: set[int] = set()

    for caixa_evento in bounding_boxes_evento:
        melhor_indice = None
        melhor_iou = -1.0

        for indice_yolo, caixa_yolo in enumerate(bounding_boxes_yolo):
            iou, _ = calcular_metricas_sobreposicao(caixa_evento, caixa_yolo)
            if iou > 0 and iou > melhor_iou:
                melhor_iou = iou
                melhor_indice = indice_yolo

        if melhor_indice is not None:
            indices_alerta_evento.add(melhor_indice)

    return indices_alerta_evento
