from __future__ import annotations

from typing import Any

from app.domain.models.camera_event import BoundingBox


def _get_bounds(box: Any) -> tuple[int, int, int, int]:
    x1 = int(box.x)
    y1 = int(box.y)
    x2 = x1 + int(box.width)
    y2 = y1 + int(box.height)

    return x1, y1, x2, y2


def calculate_overlap_metrics(box_a: Any, box_b: Any) -> tuple[float, float]:
    """
    Returns:

    - IoU between the boxes;
    - coverage of the smaller box by the intersection.
    """
    ax1, ay1, ax2, ay2 = _get_bounds(box_a)
    bx1, by1, bx2, by2 = _get_bounds(box_b)

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union_area = area_a + area_b - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0.0
    smaller_coverage = intersection_area / min(area_a, area_b)

    return iou, smaller_coverage


def validate_camera_boxes_with_yolo(
    camera_boxes: list[BoundingBox],
    yolo_boxes: list[BoundingBox],
    min_iou: float = 0.12,
    min_coverage: float = 0.50,
) -> list[BoundingBox]:
    """
    Keeps only the camera boxes that have a spatial match with a
    person detected by YOLO.

    The camera's original box is preserved so the rest of the
    pipeline keeps working without changes.
    """
    if not camera_boxes:
        return []

    if not yolo_boxes:
        return []

    validated_boxes: list[BoundingBox] = []

    for camera_box in camera_boxes:
        confirmed = False

        for yolo_box in yolo_boxes:
            iou, coverage = calculate_overlap_metrics(camera_box, yolo_box)
            if iou >= min_iou or coverage >= min_coverage:
                confirmed = True
                break

        if confirmed:
            validated_boxes.append(camera_box)

    return validated_boxes
