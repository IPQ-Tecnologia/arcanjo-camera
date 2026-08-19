"""
Per-person processing: appearance, movement, face crop, building the
alert/face payloads, and publishing to the panel and to Kafka.
Extracted from CameraEventPipeline.
"""

import asyncio
import logging

from app.core.config import settings
from app.domain.models.camera_event import BoundingBox, CameraEvent
from app.messaging.kafka_producer import KafkaPublisher
from app.services.alarm_detection_payload import (
    build_alarm_detection_message,
    crop_detection_base64,
    file_to_base64,
)
from app.services.appearance_memory import appearance_memory
from app.services.event_hub import event_hub
from app.services.face_cropper import FaceCropResult, crop_face
from app.services.person_movement import person_movement_memory
from app.services.person_tracker import TrackingDecision
from app.services.scene_analyzer import analyze_person

logger = logging.getLogger(__name__)


class PersonProcessor:
    """
    Processes each person detected in a normalized event: analyzes
    appearance and movement, generates a face crop, builds the alert
    and face-capture payloads, and publishes everything to the panel
    and to Kafka.
    """

    def __init__(self, publisher: KafkaPublisher, topic: str) -> None:
        self.publisher = publisher
        self.topic = topic

    async def _analyze_appearance(
        self,
        event: CameraEvent,
        bounding_box: BoundingBox,
        person_id: str,
        package_id: str,
    ):
        try:
            if event.image is None or not event.image.original_path:
                logger.info(
                    "[%s] Image has no original path; appearance not analyzed",
                    package_id,
                )
                return None

            box_percentage = float(bounding_box.image_ratio or 0.0) * 100

            # A very small person doesn't have enough clothing pixels
            # for a reliable classification.
            if (
                bounding_box.width < 60
                or bounding_box.height < 120
                or box_percentage < 1.00
            ):
                logger.info(
                    "[%s] Appearance skipped: small box person=%s "
                    "dimensions=%sx%s area=%.2f%%",
                    package_id,
                    person_id,
                    bounding_box.width,
                    bounding_box.height,
                    box_percentage,
                )
                return await appearance_memory.get(person_id)

            visual_analysis = await asyncio.to_thread(
                analyze_person,
                image_path=event.image.original_path,
                x=bounding_box.x,
                y=bounding_box.y,
                width=bounding_box.width,
                height=bounding_box.height,
            )
            # An ambiguous reading doesn't enter the memory's voting.
            # If a reliable appearance already exists, it's preserved.
            if visual_analysis.approximate_clothing_color == "escura-indefinida":
                logger.info(
                    "[%s] Ambiguous dark color: person=%s rgb=%s; "
                    "keeping previous appearance",
                    package_id,
                    person_id,
                    visual_analysis.representative_rgb,
                )
                return await appearance_memory.get(person_id)

            stable_appearance = await appearance_memory.register(
                person_id=person_id,
                analysis=visual_analysis,
            )
            logger.info(
                "[%s] Stable appearance: person=%s color=%s position=%s "
                "size=%s average_area=%s%% samples=%s",
                package_id,
                person_id,
                stable_appearance.cor_roupa_predominante,
                stable_appearance.posicao_atual,
                stable_appearance.tamanho_predominante,
                stable_appearance.percentual_medio_quadro,
                stable_appearance.quantidade_amostras,
            )
            return stable_appearance
        except Exception:
            logger.exception(
                "[%s] Could not analyze the appearance of person=%s",
                package_id,
                person_id,
            )
            return None

    async def _get_face_crop(
        self,
        event: CameraEvent,
        bounding_box: BoundingBox,
        person_id: str,
        package_id: str,
    ) -> FaceCropResult | None:
        if event.image is None or not event.image.original_path:
            return None

        if bounding_box.width < 60 or bounding_box.height < 100:
            logger.info(
                "[%s] Face skipped: small person box person=%s dimensions=%sx%s",
                package_id,
                person_id,
                bounding_box.width,
                bounding_box.height,
            )
            return None

        safe_name = "".join(
            character if (character.isalnum() or character in "-_") else "_"
            for character in person_id
        )

        try:
            result = await asyncio.to_thread(
                crop_face,
                image_path=event.image.original_path,
                bounding_box=bounding_box,
                output_folder="face_crops",
                file_name=f"{package_id}_{safe_name}_face.jpg",
            )

            if result is None:
                logger.info("[%s] Face not found: person=%s", package_id, person_id)
                return None

            logger.info(
                "[%s] Face crop created: person=%s box=%s,%s %sx%s file=%s",
                package_id,
                person_id,
                result.x,
                result.y,
                result.width,
                result.height,
                result.file_path,
            )
            return result

        except Exception:
            logger.exception(
                "[%s] Error generating face crop: person=%s", package_id, person_id
            )
            return None

    async def _publish_partial_update(
        self,
        camera: str,
        decision: TrackingDecision,
        stable_appearance,
        movement,
        scene_context,
        rendered_scene,
        crop_image_base64: str | None,
        scene_index: int,
        scene_total_people: int,
        package_id: str,
    ) -> None:
        # NOTE: the dict keys below are the WebSocket/panel contract
        # and are intentionally kept in Portuguese, matching the
        # frontend. Only the surrounding code is in English.
        partial_update = {
            "evento_id": decision.event_id,
            "pessoa_id": decision.person_id,
            "status": "appearance_updated",
            "quantidade_deteccoes": decision.detection_count,
            "camera": camera,
            "imagem": crop_image_base64,
            "aparencia": stable_appearance.to_dict() if stable_appearance is not None else None,
            "movimento": movement.to_dict() if movement is not None else None,
            "contexto_cena": scene_context.to_dict() if scene_context is not None else None,
            "imagem_cena": (
                rendered_scene.image_base64 if rendered_scene is not None else None
            ),
            "quantidade_boxes_cena": (
                rendered_scene.box_count if rendered_scene is not None else 0
            ),
            "indice_na_cena": scene_index,
            "total_pessoas_cena": scene_total_people,
        }
        await event_hub.publish(partial_update)
        logger.info(
            "[%s] Partial update: person=%s index=%s/%s "
            "detections=%s samples=%s movement=%s/%s speed=%spx_s "
            "boxes=%s panels=%s",
            package_id,
            decision.person_id,
            scene_index,
            scene_total_people,
            decision.detection_count,
            stable_appearance.quantidade_amostras if stable_appearance is not None else 0,
            movement.movimento_horizontal if movement is not None else "-",
            movement.movimento_vertical if movement is not None else "-",
            movement.velocidade_pixels_segundo if movement is not None else 0,
            rendered_scene.box_count if rendered_scene is not None else 0,
            event_hub.total_connections,
        )

    async def process_person(
        self,
        event: CameraEvent,
        camera: str,
        bounding_box: BoundingBox,
        decision: TrackingDecision,
        scene_context,
        rendered_scene,
        scene_index: int,
        scene_total_people: int,
        package_id: str,
        send_event_alert: bool,
        background_image_base64: str | None,
    ) -> str | None:
        stable_appearance = await self._analyze_appearance(
            event=event,
            bounding_box=bounding_box,
            person_id=decision.person_id,
            package_id=package_id,
        )

        movement = await person_movement_memory.register(
            person_id=decision.person_id,
            bbox=decision.bbox,
        )
        logger.info(
            "[%s] Movement: person=%s horizontal=%s vertical=%s "
            "trend=%s speed=%spx_s total_distance=%spx "
            "time=%ss samples=%s",
            package_id,
            decision.person_id,
            movement.movimento_horizontal,
            movement.movimento_vertical,
            movement.tendencia_distancia,
            movement.velocidade_pixels_segundo,
            movement.distancia_total_pixels,
            movement.tempo_observado_segundos,
            movement.quantidade_amostras,
        )

        if not decision.should_process:
            crop_image_base64 = None

            if event.image is not None and event.image.original_path:
                try:
                    crop_image_base64 = await asyncio.to_thread(
                        crop_detection_base64,
                        event.image.original_path,
                        bounding_box,
                    )
                except Exception:
                    logger.exception(
                        "[%s] Could not update the crop for person=%s",
                        package_id,
                        decision.person_id,
                    )

            await self._publish_partial_update(
                camera=camera,
                decision=decision,
                stable_appearance=stable_appearance,
                movement=movement,
                scene_context=scene_context,
                rendered_scene=rendered_scene,
                crop_image_base64=crop_image_base64,
                scene_index=scene_index,
                scene_total_people=scene_total_people,
                package_id=package_id,
            )
            logger.info(
                "[%s] Repeated event suppressed: person=%s index=%s/%s samples=%s",
                package_id,
                decision.person_id,
                scene_index,
                scene_total_people,
                stable_appearance.quantidade_amostras if stable_appearance is not None else 0,
            )
            return background_image_base64

        if event.image is None or not event.image.original_path:
            logger.info("[%s] Person skipped: missing original path", package_id)
            return background_image_base64

        face_result = await self._get_face_crop(
            event=event,
            bounding_box=bounding_box,
            person_id=decision.person_id,
            package_id=package_id,
        )
        face_image_base64 = face_result.image_base64 if face_result is not None else None

        if background_image_base64 is None:
            background_image_base64 = await asyncio.to_thread(
                file_to_base64,
                event.image.original_path,
            )

        try:
            message = await asyncio.to_thread(
                build_alarm_detection_message,
                event,
                bounding_box=bounding_box,
                event_id=decision.event_id,
                background_image_base64=background_image_base64,
            )
        except ValueError as error:
            logger.info(
                "[%s] Person skipped: person=%s error=%s",
                package_id,
                decision.person_id,
                error,
            )
            return background_image_base64

        face_kafka_payload = None

        normalized_event = event.model_dump(mode="json", by_alias=True, exclude_none=True)
        normalized_attributes = normalized_event.get("attributes") or {}

        vendor_event_type = (
            normalized_attributes.get("vendor_event_type")
            or normalized_event.get("event_type")
            or event.event_type
        )

        # The supervisor requested a numeric event.id.
        # Prioritizes the original ID sent by the camera.
        source_event_id = normalized_attributes.get("source_event_id")

        try:
            numeric_event_id = int(source_event_id)
        except (TypeError, ValueError):
            event_id_text = str(decision.event_id)

            # Multi-person IDs can end with suffixes like "-01", "-02",
            # "-03". The base event ID is still the hexadecimal part
            # before that.
            base_event_id = event_id_text.split("-", 1)[0]

            try:
                numeric_event_id = int(base_event_id, 16)
            except (TypeError, ValueError):
                numeric_event_id = int.from_bytes(
                    event_id_text.encode("utf-8")[:8].ljust(8, b"\0"),
                    byteorder="big",
                )

        # ==================================================
        # FACE EVENT
        # ==================================================

        if face_result is not None:
            face_kafka_payload = {
                "camera_name": message.device.name,
                "images": {"face": {"base64": face_result.image_base64}},
            }

            logger.info(
                "[%s] Simple face payload ready: "
                "topic=nelore-face-capture camera=%s quality=%.4f bbox=%s,%s-%s,%s",
                package_id,
                message.device.name,
                face_result.score,
                face_result.x,
                face_result.y,
                face_result.x + face_result.width,
                face_result.y + face_result.height,
            )

        # ==================================================
        # ALERT EVENT
        # ==================================================

        alert_kafka_payload = {
            "schema_version": event.schema_version or "1.0",
            "device": {
                "name": message.device.name,
                "brand": event.manufacturer,
                "ip": event.camera_ip,
                "external_id": event.camera_id,
                "serial_number": None,
                "latitude": None,
                "longitude": None,
            },
            "event": {
                "id": numeric_event_id,
                "name": vendor_event_type,
                "type": event.event_type,
                "timestamp": int(message.event.timestamp),
                "datetime": message.event.datetime,
            },
            "image": {
                "type": "detection",
                "width": event.image.width if event.image is not None else 0,
                "height": event.image.height if event.image is not None else 0,
                "format": (
                    event.image.format
                    if event.image is not None and event.image.format
                    else "jpeg"
                ),
                "original_image_content": message.event.images.background,
                "annotated_image_content": (
                    rendered_scene.image_base64 if rendered_scene is not None else ""
                ),
                "original_image_path": None,
                "annotated_image_path": None,
                "video_path": None,
            },
        }

        if send_event_alert:
            logger.info(
                "[%s] Alert payload ready: topic=%s id=%s name=%s type=%s",
                package_id,
                self.topic,
                numeric_event_id,
                vendor_event_type,
                event.event_type,
            )
        else:
            logger.info(
                "[%s] Person present in the scene with no link to the camera event: "
                "person=%s; alert will not be published",
                package_id,
                decision.person_id,
            )

        # NOTE: the dict keys below are the WebSocket/panel contract
        # and are intentionally kept in Portuguese, matching the
        # frontend. Only the surrounding code is in English.
        panel_event = {
            "evento_id": decision.event_id,
            "pessoa_id": decision.person_id,
            "status": decision.status,
            "quantidade_deteccoes": decision.detection_count,
            "camera": message.device.name,
            "tipo": message.event.type,
            "datetime": message.event.datetime,
            "attributes": (
                message.event.attributes.model_dump(mode="json")
                if message.event.attributes is not None
                else None
            ),
            "imagem": message.event.images.detection,
            "imagem_rosto": face_image_base64,
            "aparencia": stable_appearance.to_dict() if stable_appearance is not None else None,
            "movimento": movement.to_dict(),
            "contexto_cena": scene_context.to_dict() if scene_context is not None else None,
            "imagem_cena": (
                rendered_scene.image_base64 if rendered_scene is not None else None
            ),
            "quantidade_boxes_cena": (
                rendered_scene.box_count if rendered_scene is not None else 0
            ),
            "indice_na_cena": scene_index,
            "total_pessoas_cena": scene_total_people,
        }
        await event_hub.publish(panel_event)
        logger.info(
            "[%s] Person sent to panel: person=%s status=%s index=%s/%s panels=%s",
            package_id,
            decision.person_id,
            decision.status,
            scene_index,
            scene_total_people,
            event_hub.total_connections,
        )

        if not settings.kafka_enabled:
            logger.info(
                "[%s] Payload ready; Kafka disabled: person=%s",
                package_id,
                decision.person_id,
            )
            return background_image_base64

        kafka_publications = []

        if send_event_alert:
            kafka_publications.append(
                {
                    "label": "Alert",
                    "topic": self.topic,
                    "coroutine": self.publisher.publish(
                        topic=self.topic,
                        event_id=decision.event_id,
                        data=alert_kafka_payload,
                    ),
                }
            )

        if face_kafka_payload is not None:
            kafka_publications.append(
                {
                    "label": "Face",
                    "topic": "nelore-face-capture",
                    "coroutine": self.publisher.publish(
                        topic="nelore-face-capture",
                        event_id=decision.event_id,
                        data=face_kafka_payload,
                    ),
                }
            )

        kafka_results = await asyncio.gather(
            *[item["coroutine"] for item in kafka_publications],
            return_exceptions=True,
        )

        for item, kafka_result in zip(kafka_publications, kafka_results, strict=True):
            if isinstance(kafka_result, BaseException):
                logger.error(
                    "[%s] Error publishing %s to %s: %r",
                    package_id,
                    item["label"].lower(),
                    item["topic"],
                    kafka_result,
                )
                continue

            logger.info(
                "[%s] %s published to %s p=%s offset=%s person=%s",
                package_id,
                item["label"],
                kafka_result["topic"],
                kafka_result["partition"],
                kafka_result["offset"],
                decision.person_id,
            )

        return background_image_base64
