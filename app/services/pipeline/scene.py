"""
Scene context analysis and rendering of the image with the drawn
boxes. Thin wrappers around scene_context_analyzer and scene_renderer,
extracted from CameraEventPipeline.
"""

import asyncio
import logging

from app.domain.models.camera_event import BoundingBox, CameraEvent
from app.services.scene_context_analyzer import (
    analyze_scene_context as _analyze_scene_context_sync,
)
from app.services.scene_renderer import render_scene_with_boxes

logger = logging.getLogger(__name__)


async def analyze_scene_context(
    event: CameraEvent,
    bounding_boxes: list[BoundingBox],
    package_id: str,
):
    try:
        if event.image is None:
            return None

        image_width = event.image.width
        image_height = event.image.height
        if not image_width or not image_height:
            logger.info("[%s] Context not analyzed: missing dimensions", package_id)
            return None

        context = await asyncio.to_thread(
            _analyze_scene_context_sync,
            image_width=image_width,
            image_height=image_height,
            bounding_boxes=bounding_boxes,
        )
        logger.info(
            "[%s] Scene context: people=%s left=%s center=%s "
            "right=%s very_close=%s close=%s separated=%s",
            package_id,
            context.quantidade_pessoas,
            context.pessoas_esquerda,
            context.pessoas_centro,
            context.pessoas_direita,
            context.pares_muito_proximos,
            context.pares_proximos,
            context.pares_separados,
        )
        logger.info("[%s] Scene description: %s", package_id, context.descricao)
        return context
    except Exception:
        logger.exception("[%s] Could not analyze the scene", package_id)
        return None


async def render_scene(
    event: CameraEvent,
    bounding_boxes: list[BoundingBox],
    package_id: str,
):
    try:
        if event.image is None or not event.image.original_path:
            return None
        if not bounding_boxes:
            return None

        result = await asyncio.to_thread(
            render_scene_with_boxes,
            image_path=event.image.original_path,
            bounding_boxes=bounding_boxes,
        )
        if result.box_count == 0:
            return None

        logger.info(
            "[%s] Scene rendered: boxes=%s dimensions=%sx%s base64=%s characters",
            package_id,
            result.box_count,
            result.image_width,
            result.image_height,
            len(result.image_base64),
        )
        return result
    except Exception:
        logger.exception("[%s] Could not render the scene with boxes", package_id)
        return None
