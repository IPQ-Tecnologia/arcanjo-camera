"""
Selection and validation of the person boxes coming from the camera
event.

Extracted from CameraEventPipeline: this is where everything that
decides "which boxes are actually people" lives, before moving on to
tracking.
"""

import asyncio
import logging

from app.domain.models.camera_event import BoundingBox, CameraEvent
from app.services.person_detector_yolo import detect_people_yolo
from app.services.person_validation import (
    calculate_overlap_metrics,
    validate_camera_boxes_with_yolo,
)

logger = logging.getLogger(__name__)


def select_person_boxes(event: CameraEvent) -> list[BoundingBox]:
    """Removes invalid boxes, duplicates, and the full-image box."""
    if event.image is None:
        return []

    image_width = event.image.width
    image_height = event.image.height
    if not image_width or not image_height:
        return []

    image_area = image_width * image_height
    seen: set[tuple[int, int, int, int]] = set()
    valid_boxes: list[BoundingBox] = []

    for box in event.bounding_boxes or []:
        if box.width <= 0 or box.height <= 0:
            continue

        x1 = max(0, min(box.x, image_width - 1))
        y1 = max(0, min(box.y, image_height - 1))
        x2 = max(x1 + 1, min(box.x2, image_width))
        y2 = max(y1 + 1, min(box.y2, image_height))
        width = x2 - x1
        height = y2 - y1
        percentage = width * height / image_area * 100

        if percentage >= 60 or percentage < 0.05:
            continue

        signature = (x1, y1, x2, y2)
        if signature in seen:
            continue
        seen.add(signature)

        same_box = x1 == box.x and y1 == box.y and width == box.width and height == box.height
        if same_box:
            adjusted_box = box
        else:
            adjusted_box = box.model_copy(
                update={
                    "x": x1,
                    "y": y1,
                    "width": width,
                    "height": height,
                    "x2": x2,
                    "y2": y2,
                }
            )

        valid_boxes.append(adjusted_box)

    if not valid_boxes and event.selected_bounding_box is not None:
        valid_boxes.append(event.selected_bounding_box)

    valid_boxes.sort(key=lambda box: (box.x + box.width / 2, box.y + box.height / 2))
    return valid_boxes


async def validate_with_yolo(
    event: CameraEvent,
    camera_boxes: list[BoundingBox],
    package_id: str,
) -> tuple[list[BoundingBox], set[int]]:
    """
    Runs YOLO on the image to confirm which camera boxes are actually
    people.

    Returns every person detected by YOLO (for tracker/panel/face) and
    the set of indexes, within that list, that correspond to the box
    that actually triggered the camera's alarm event.

    If YOLO can't run (no saved image or inference failure), uses the
    camera boxes as they are, all of them as alert targets.
    """
    yolo_boxes: list[BoundingBox] = []
    yolo_ran = False
    original_path = event.image.original_path if event.image else None

    if original_path:
        try:
            yolo_boxes = await asyncio.to_thread(detect_people_yolo, original_path)
            yolo_ran = True
        except Exception:
            logger.exception(
                "[%s] Failed to validate people with YOLO; temporarily using the camera boxes",
                package_id,
            )
    else:
        logger.warning(
            "[%s] Image has no original path; YOLO validation could not run",
            package_id,
        )

    if not yolo_ran:
        return camera_boxes, set(range(len(camera_boxes)))

    event_boxes = validate_camera_boxes_with_yolo(
        camera_boxes=camera_boxes,
        yolo_boxes=yolo_boxes,
    )

    rejected_count = len(camera_boxes) - len(event_boxes)
    logger.info(
        "[%s] YOLO validation: camera=%s yolo=%s validated=%s rejected=%s",
        package_id,
        len(camera_boxes),
        len(yolo_boxes),
        len(event_boxes),
        rejected_count,
    )
    if rejected_count > 0:
        logger.info("[%s] Possible false positive rejected by YOLO", package_id)

    if not event_boxes:
        return [], set()

    # Every person detected by YOLO moves on to tracker/panel/face.
    boxes = yolo_boxes
    alert_indexes = _find_alert_indexes(event_boxes, boxes)

    logger.info(
        "[%s] People in the scene via YOLO: people=%s event_targets=%s",
        package_id,
        len(boxes),
        sorted(alert_indexes),
    )
    return boxes, alert_indexes


def _find_alert_indexes(
    event_boxes: list[BoundingBox],
    yolo_boxes: list[BoundingBox],
) -> set[int]:
    """
    Figures out which person detected by YOLO matches the box that
    actually triggered the event reported by the camera, using the
    highest overlap (IoU) between boxes.
    """
    alert_indexes: set[int] = set()

    for event_box in event_boxes:
        best_index = None
        best_iou = -1.0

        for yolo_index, yolo_box in enumerate(yolo_boxes):
            iou, _ = calculate_overlap_metrics(event_box, yolo_box)
            if iou > 0 and iou > best_iou:
                best_iou = iou
                best_index = yolo_index

        if best_index is not None:
            alert_indexes.add(best_index)

    return alert_indexes
