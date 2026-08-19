"""
Background monitor that detects when a tracked person has left the
scene and publishes that event to the panel. Extracted from
CameraEventPipeline.
"""

import asyncio
import logging
from datetime import datetime, timezone

from app.services.appearance_memory import appearance_memory
from app.services.event_hub import event_hub
from app.services.person_movement import person_movement_memory
from app.services.person_tracker import person_tracker

logger = logging.getLogger(__name__)


async def monitor_exits() -> None:
    logger.info("Automatic exit monitor started")

    while True:
        try:
            await asyncio.sleep(1)
            exits = await person_tracker.collect_exits()

            for departure in exits:
                final_appearance = await appearance_memory.finalize(departure.person_id)
                final_movement = await person_movement_memory.finalize(departure.person_id)
                departure_moment = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

                # NOTE: the dict keys below are the WebSocket/panel
                # contract and are intentionally kept in Portuguese,
                # matching the frontend. Only the surrounding code is
                # in English.
                panel_event = {
                    "evento_id": departure.event_id,
                    "pessoa_id": departure.person_id,
                    "status": "exited",
                    "quantidade_deteccoes": departure.detection_count,
                    "camera": departure.camera,
                    "tipo": None,
                    "datetime": departure_moment,
                    "attributes": None,
                    "imagem": None,
                    "aparencia": (
                        final_appearance.to_dict() if final_appearance is not None else None
                    ),
                    "movimento": (
                        final_movement.to_dict() if final_movement is not None else None
                    ),
                    "contexto_cena": None,
                    "imagem_cena": None,
                    "quantidade_boxes_cena": None,
                }
                await event_hub.publish(panel_event)
                logger.info(
                    "[TRACKING] Person left: person=%s camera=%s "
                    "detections=%s visual_samples=%s movement_samples=%s "
                    "observed_time=%ss panels=%s",
                    departure.person_id,
                    departure.camera,
                    departure.detection_count,
                    (
                        final_appearance.quantidade_amostras
                        if final_appearance is not None
                        else 0
                    ),
                    (
                        final_movement.quantidade_amostras
                        if final_movement is not None
                        else 0
                    ),
                    (
                        final_movement.tempo_observado_segundos
                        if final_movement is not None
                        else 0
                    ),
                    event_hub.total_connections,
                )

        except asyncio.CancelledError:
            logger.info("Automatic exit monitor stopped")
            raise
        except Exception:
            logger.exception("Error in the automatic exit monitor")
