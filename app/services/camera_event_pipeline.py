import asyncio
import logging
from datetime import datetime, timezone

from app.adapters.cameras.factory import camera_adapter_factory
from app.core.config import settings
from app.domain.models.camera_event import CameraEvent, RawCameraPackage
from app.messaging.kafka_events import publish_processing_error
from app.messaging.kafka_producer import KafkaPublisher
from app.services.appearance_memory import appearance_memory
from app.services.person_movement import person_movement_memory
from app.services.person_tracker import DetectionBox, person_tracker
from app.services.pipeline import box_matching, scene
from app.services.pipeline.exit_monitor import monitor_exits
from app.services.pipeline.person_processing import PersonProcessor

logger = logging.getLogger(__name__)


class CameraEventPipeline:
    """
    Orchestrates the asynchronous processing of camera events: takes
    raw packages off the queue, normalizes them via the manufacturer's
    adapter, validates people with YOLO, tracks them across frames,
    and hands each detected person to the PersonProcessor
    (appearance/movement/face/Kafka).
    """

    def __init__(
        self,
        publisher: KafkaPublisher,
        maxsize: int = 1000,
        worker_count: int = 4,
    ) -> None:
        self.publisher = publisher
        self.worker_count = worker_count
        self.queue: asyncio.Queue[tuple[RawCameraPackage, bytes]] = asyncio.Queue(
            maxsize=maxsize
        )
        self._person_processor = PersonProcessor(
            publisher=publisher
        )
        self._worker_tasks: list[asyncio.Task] = []
        self._exit_task: asyncio.Task | None = None
        self._started = False

    @property
    def queue_size(self) -> int:
        return self.queue.qsize()

    @property
    def queue_capacity(self) -> int:
        return self.queue.maxsize

    async def start(self) -> None:
        if self._started:
            return

        if settings.kafka_enabled:
            await self.publisher.start()
        else:
            logger.warning("Kafka disabled: events will be processed, but not published.")

        self._worker_tasks = [
            asyncio.create_task(self._worker(number), name=f"camera-worker-{number}")
            for number in range(1, self.worker_count + 1)
        ]
        self._exit_task = asyncio.create_task(monitor_exits(), name="person-exit-monitor")
        self._started = True
        logger.info("Pipeline started with %s workers", self.worker_count)

    async def stop(self) -> None:
        if not self._started:
            return

        try:
            await asyncio.wait_for(self.queue.join(), timeout=10)
        except TimeoutError:
            logger.warning("Shutting down with %s items still in the queue", self.queue.qsize())

        tasks: list[asyncio.Task] = [*self._worker_tasks]
        if self._exit_task is not None:
            tasks.append(self._exit_task)

        for task in tasks:
            task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        if self.publisher.started:
            await self.publisher.stop()

        await appearance_memory.clear()
        await person_movement_memory.clear()
        await person_tracker.clear()

        self._worker_tasks.clear()
        self._exit_task = None
        self._started = False
        logger.info("Pipeline stopped")

    def add(self, package: RawCameraPackage, body: bytes) -> None:
        self.queue.put_nowait((package, body))

    @staticmethod
    def _log_camera_time(event_id: str, event: CameraEvent) -> None:
        """Compares the time reported by the camera with the server's time."""
        camera_time = event.timestamp
        if camera_time.tzinfo is None:
            camera_time = camera_time.replace(tzinfo=timezone.utc)

        camera_time_utc = camera_time.astimezone(timezone.utc)
        camera_time_local = camera_time.astimezone()
        server_time_utc = datetime.now(timezone.utc)
        server_time_local = server_time_utc.astimezone()
        difference_ms = (server_time_utc - camera_time_utc).total_seconds() * 1000

        logger.info(
            "[%s] CAMERA TIME: manufacturer=%s camera=%s sent_utc=%s "
            "sent_local=%s received_local=%s difference_ms=%.0f",
            event_id,
            event.manufacturer,
            event.camera_name or event.camera_id or "unknown",
            camera_time_utc.isoformat(),
            camera_time_local.isoformat(),
            server_time_local.isoformat(),
            difference_ms,
        )

    async def _worker(self, number: int) -> None:
        logger.info("Worker %s started", number)

        while True:
            package, body = await self.queue.get()

            try:
                adapter = camera_adapter_factory.find_adapter(
                    content_type=package.content_type,
                    body=body,
                )
                logger.info(
                    "[%s] Worker %s using %s",
                    package.event_id,
                    number,
                    adapter.__class__.__name__,
                )

                event = await asyncio.to_thread(adapter.normalize, package, body)

                logger.info(
                    "[%s] NORMALIZED OBJECT (%s):\n%s",
                    package.event_id,
                    event.manufacturer,
                    event.model_dump_json(indent=2, by_alias=True, exclude_none=True),
                )

                self._log_camera_time(package.event_id, event)

                if event.image is None:
                    logger.info("[%s] Event skipped: no image", package.event_id)
                    continue

                camera_boxes = box_matching.select_person_boxes(event)
                if not camera_boxes:
                    logger.info(
                        "[%s] Event skipped: no valid boxes provided by the camera",
                        package.event_id,
                    )
                    continue

                bounding_boxes, event_alert_indexes = await box_matching.validate_with_yolo(
                    event=event,
                    camera_boxes=camera_boxes,
                    package_id=package.event_id,
                )

                if not bounding_boxes:
                    logger.info(
                        "[%s] Event skipped: no box was confirmed as a person by YOLO",
                        package.event_id,
                    )
                    continue

                camera = (
                    event.camera_name
                    or event.camera_id
                    or event.camera_ip
                    or "Unknown camera"
                )
                tracking_boxes = [
                    DetectionBox(x=box.x, y=box.y, width=box.width, height=box.height)
                    for box in bounding_boxes
                ]
                decisions = await person_tracker.register_batch(
                    camera=camera,
                    event_id=event.event_id,
                    bboxes=tracking_boxes,
                )

                if len(decisions) != len(bounding_boxes):
                    raise RuntimeError("Number of decisions differs from the number of boxes")

                total_people = len(bounding_boxes)
                logger.info(
                    "[%s] Batch tracking: people=%s ids=%s",
                    package.event_id,
                    total_people,
                    [(decision.person_id, decision.status) for decision in decisions],
                )

                scene_context = await scene.analyze_scene_context(
                    event=event,
                    bounding_boxes=bounding_boxes,
                    package_id=package.event_id,
                )
                rendered_scene = await scene.render_scene(
                    event=event,
                    bounding_boxes=bounding_boxes,
                    package_id=package.event_id,
                )

                background_image_base64: str | None = None
                for index, (bounding_box, decision) in enumerate(
                    zip(bounding_boxes, decisions, strict=True),
                    start=1,
                ):
                    background_image_base64 = await self._person_processor.process_person(
                        event=event,
                        camera=camera,
                        bounding_box=bounding_box,
                        decision=decision,
                        scene_context=scene_context,
                        rendered_scene=rendered_scene,
                        scene_index=index,
                        scene_total_people=total_people,
                        package_id=package.event_id,
                        send_event_alert=index - 1 in event_alert_indexes,
                        background_image_base64=background_image_base64,
                    )

            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception("[%s] Processing error", package.event_id)

                if settings.kafka_enabled and self.publisher.started:
                    error_payload = {
                        "event_id": package.event_id,
                        "type": "camera_processing_error",
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                        "source": {
                            "camera_ip": package.camera_ip,
                            "content_type": package.content_type,
                            "received_at": package.received_at.isoformat(),
                            "package_path": package.package_path,
                        },
                        "worker": number,
                        "occurred_at": datetime.now(timezone.utc).isoformat(),
                    }

                    try:
                        await publish_processing_error(
                            publisher=self.publisher,
                            event_id=package.event_id,
                            data=error_payload,
                        )
                    except Exception:
                        logger.exception(
                            "[%s] Could not publish processing error to Kafka",
                            package.event_id,
                        )
            finally:
                self.queue.task_done()
