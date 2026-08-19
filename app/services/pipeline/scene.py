"""
Análise de contexto da cena e renderização da imagem com as caixas
desenhadas. Wrappers finos em cima de scene_context_analyzer e
scene_renderer, extraídos de CameraEventPipeline.
"""

import asyncio
import logging

from app.domain.models.camera_event import BoundingBox, CameraEvent
from app.services.scene_context_analyzer import (
    analisar_contexto_cena as _analisar_contexto_cena_sync,
)
from app.services.scene_renderer import renderizar_cena_com_boxes

logger = logging.getLogger(__name__)


async def analisar_contexto_cena(
    evento: CameraEvent,
    bounding_boxes: list[BoundingBox],
    pacote_id: str,
):
    try:
        if evento.imagem is None:
            return None

        largura_imagem = evento.imagem.largura
        altura_imagem = evento.imagem.altura
        if not largura_imagem or not altura_imagem:
            logger.info("[%s] Contexto não analisado: dimensões ausentes", pacote_id)
            return None

        contexto = await asyncio.to_thread(
            _analisar_contexto_cena_sync,
            largura_imagem=largura_imagem,
            altura_imagem=altura_imagem,
            bounding_boxes=bounding_boxes,
        )
        logger.info(
            "[%s] Contexto da cena: pessoas=%s esquerda=%s centro=%s "
            "direita=%s muito_proximos=%s proximos=%s separados=%s",
            pacote_id,
            contexto.quantidade_pessoas,
            contexto.pessoas_esquerda,
            contexto.pessoas_centro,
            contexto.pessoas_direita,
            contexto.pares_muito_proximos,
            contexto.pares_proximos,
            contexto.pares_separados,
        )
        logger.info("[%s] Descrição da cena: %s", pacote_id, contexto.descricao)
        return contexto
    except Exception:
        logger.exception("[%s] Não foi possível analisar a cena", pacote_id)
        return None


async def renderizar_cena(
    evento: CameraEvent,
    bounding_boxes: list[BoundingBox],
    pacote_id: str,
):
    try:
        if evento.imagem is None or not evento.imagem.caminho_original:
            return None
        if not bounding_boxes:
            return None

        resultado = await asyncio.to_thread(
            renderizar_cena_com_boxes,
            caminho_imagem=evento.imagem.caminho_original,
            bounding_boxes=bounding_boxes,
        )
        if resultado.quantidade_boxes == 0:
            return None

        logger.info(
            "[%s] Cena renderizada: boxes=%s dimensoes=%sx%s base64=%s caracteres",
            pacote_id,
            resultado.quantidade_boxes,
            resultado.largura_imagem,
            resultado.altura_imagem,
            len(resultado.imagem_base64),
        )
        return resultado
    except Exception:
        logger.exception("[%s] Não foi possível renderizar a cena com boxes", pacote_id)
        return None
