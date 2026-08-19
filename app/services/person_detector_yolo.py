from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO

from app.domain.models.camera_event import BoundingBox


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Segmentation model: provides boxes and masks.
MODEL_PATH = PROJECT_ROOT / "yolo11n-seg.pt"

_model: YOLO | None = None

_model_lock = Lock()
_predict_lock = Lock()
_cache_lock = Lock()

# Keeps recent results so the same image isn't run through YOLO again
# when analyzed a second time.
_MAX_CACHE = 4

_results_cache: OrderedDict[tuple[str, int, int, float, float, int], Any] = OrderedDict()


def _get_model() -> YOLO:
    global _model

    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

            logger.info("Loading segmentation model: %s", MODEL_PATH)
            _model = YOLO(str(MODEL_PATH))

    return _model


def _get_person_classes(model: YOLO) -> list[int]:
    classes = [
        index for index, name in model.names.items() if str(name).strip().lower() == "person"
    ]

    if not classes:
        raise RuntimeError("The person class was not found in the YOLO model.")

    return classes


def _build_cache_key(
    path: Path,
    min_confidence: float,
    iou: float,
    image_size: int,
) -> tuple[str, int, int, float, float, int]:
    stats = path.stat()

    return (
        str(path.resolve()),
        stats.st_mtime_ns,
        stats.st_size,
        round(min_confidence, 4),
        round(iou, 4),
        image_size,
    )


def _get_yolo_result(
    path: Path,
    min_confidence: float,
    iou: float,
    image_size: int,
):
    key = _build_cache_key(
        path=path,
        min_confidence=min_confidence,
        iou=iou,
        image_size=image_size,
    )

    with _cache_lock:
        cached_result = _results_cache.get(key)
        if cached_result is not None:
            _results_cache.move_to_end(key)
            return cached_result

    model = _get_model()
    person_classes = _get_person_classes(model)

    # Prevents several workers from running the model at the same
    # time and overloading the CPU.
    with _predict_lock:
        # Checks again because another thread may have finished the
        # prediction while we were waiting.
        with _cache_lock:
            cached_result = _results_cache.get(key)
            if cached_result is not None:
                _results_cache.move_to_end(key)
                return cached_result

        results = model.predict(
            source=str(path),
            classes=person_classes,
            conf=min_confidence,
            iou=iou,
            imgsz=image_size,
            device="cpu",
            verbose=False,
        )

    if not results:
        return None

    result = results[0]

    with _cache_lock:
        _results_cache[key] = result
        _results_cache.move_to_end(key)

        while len(_results_cache) > _MAX_CACHE:
            _results_cache.popitem(last=False)

    return result


def _clamp_coordinates(
    x1_float: float,
    y1_float: float,
    x2_float: float,
    y2_float: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1 = max(0, min(round(x1_float), image_width - 1))
    y1 = max(0, min(round(y1_float), image_height - 1))
    x2 = max(x1 + 1, min(round(x2_float), image_width))
    y2 = max(y1 + 1, min(round(y2_float), image_height))

    return x1, y1, x2, y2


def detect_people_yolo(
    image_path: str | Path,
    min_confidence: float = 0.50,
    iou: float = 0.30,
    image_size: int = 960,
) -> list[BoundingBox]:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    result = _get_yolo_result(
        path=path,
        min_confidence=min_confidence,
        iou=iou,
        image_size=image_size,
    )
    if result is None:
        return []

    result_boxes = result.boxes
    if result_boxes is None or len(result_boxes) == 0:
        return []

    image_height, image_width = result.orig_shape
    image_area = image_width * image_height

    boxes: list[BoundingBox] = []

    for box in result_boxes:
        x1_float, y1_float, x2_float, y2_float = box.xyxy[0].tolist()

        x1, y1, x2, y2 = _clamp_coordinates(
            x1_float=x1_float,
            y1_float=y1_float,
            x2_float=x2_float,
            y2_float=y2_float,
            image_width=image_width,
            image_height=image_height,
        )

        width = x2 - x1
        height = y2 - y1
        area = width * height
        percentage = area / image_area * 100
        confidence = float(box.conf[0].item())

        logger.info(
            "YOLO person: conf=%.4f bbox=%s,%s-%s,%s area=%.3f%%",
            confidence,
            x1,
            y1,
            x2,
            y2,
            percentage,
        )

        if percentage < 0.05 or percentage >= 60:
            continue

        boxes.append(
            BoundingBox(
                source="yolo_person_seg",
                x=x1,
                y=y1,
                width=width,
                height=height,
                x2=x2,
                y2=y2,
                image_ratio=area / image_area,
            )
        )

    boxes.sort(key=lambda box: (box.x + box.width / 2, box.y + box.height / 2))

    return boxes


def _calculate_overlap(
    reference: tuple[int, int, int, int],
    detected: tuple[int, int, int, int],
) -> tuple[float, float, float]:
    ref_x1, ref_y1, ref_x2, ref_y2 = reference
    det_x1, det_y1, det_x2, det_y2 = detected

    inter_x1 = max(ref_x1, det_x1)
    inter_y1 = max(ref_y1, det_y1)
    inter_x2 = min(ref_x2, det_x2)
    inter_y2 = min(ref_y2, det_y2)

    intersection_width = max(0, inter_x2 - inter_x1)
    intersection_height = max(0, inter_y2 - inter_y1)
    intersection_area = intersection_width * intersection_height

    reference_area = max(0, ref_x2 - ref_x1) * max(0, ref_y2 - ref_y1)
    detected_area = max(0, det_x2 - det_x1) * max(0, det_y2 - det_y1)

    if reference_area <= 0 or detected_area <= 0 or intersection_area <= 0:
        return 0.0, 0.0, 0.0

    union_area = reference_area + detected_area - intersection_area
    iou_value = intersection_area / union_area if union_area > 0 else 0.0
    reference_coverage = intersection_area / reference_area
    detected_coverage = intersection_area / detected_area

    return iou_value, reference_coverage, detected_coverage


def get_segmented_clothing_rgb(
    image_path: str | Path,
    x: int,
    y: int,
    width: int,
    height: int,
    min_confidence: float = 0.35,
    iou: float = 0.30,
    image_size: int = 960,
) -> tuple[int, int, int] | None:
    """
    Gets the clothing color using the person's mask.

    The bounding box received is used to pick the matching person
    when there are several people in the same image.

    Returns None when segmentation doesn't find a matching person. In
    that case, scene_analyzer falls back to the old method.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    if width <= 0 or height <= 0:
        return None

    result = _get_yolo_result(
        path=path,
        min_confidence=min_confidence,
        iou=iou,
        image_size=image_size,
    )
    if (
        result is None
        or result.boxes is None
        or result.masks is None
        or len(result.boxes) == 0
    ):
        return None

    image_height, image_width = result.orig_shape

    reference_x1 = max(0, min(int(x), image_width - 1))
    reference_y1 = max(0, min(int(y), image_height - 1))
    reference_x2 = max(reference_x1 + 1, min(int(x + width), image_width))
    reference_y2 = max(reference_y1 + 1, min(int(y + height), image_height))
    reference = (reference_x1, reference_y1, reference_x2, reference_y2)

    numpy_boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    best_index: int | None = None
    best_box: tuple[int, int, int, int] | None = None
    best_score = (-1.0, -1.0, -1.0, -1.0)

    for index, numpy_box in enumerate(numpy_boxes):
        detected = _clamp_coordinates(
            x1_float=float(numpy_box[0]),
            y1_float=float(numpy_box[1]),
            x2_float=float(numpy_box[2]),
            y2_float=float(numpy_box[3]),
            image_width=image_width,
            image_height=image_height,
        )

        iou_value, reference_coverage, detected_coverage = _calculate_overlap(
            reference=reference,
            detected=detected,
        )

        matches = iou_value >= 0.15 or reference_coverage >= 0.45 or detected_coverage >= 0.45
        if not matches:
            continue

        score = (
            max(iou_value, reference_coverage, detected_coverage),
            reference_coverage,
            iou_value,
            float(confidences[index]),
        )

        if score > best_score:
            best_score = score
            best_index = index
            best_box = detected

    if best_index is None or best_box is None:
        return None

    mask = result.masks.data[best_index].cpu().numpy()

    if mask.shape != (image_height, image_width):
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_image = mask_image.resize((image_width, image_height), Image.Resampling.NEAREST)
        mask = np.asarray(mask_image) > 127
    else:
        mask = mask > 0.5

    person_x1, person_y1, person_x2, person_y2 = best_box
    person_width = person_x2 - person_x1
    person_height = person_y2 - person_y1

    # Same region that worked in the real test: the central and lower
    # part of the shirt.
    torso_x1 = int(person_x1 + person_width * 0.18)
    torso_x2 = int(person_x1 + person_width * 0.82)
    torso_y1 = int(person_y1 + person_height * 0.42)
    torso_y2 = int(person_y1 + person_height * 0.72)

    torso_x1 = max(0, min(torso_x1, image_width - 1))
    torso_y1 = max(0, min(torso_y1, image_height - 1))
    torso_x2 = max(torso_x1 + 1, min(torso_x2, image_width))
    torso_y2 = max(torso_y1 + 1, min(torso_y2, image_height))

    valid_region = np.zeros((image_height, image_width), dtype=bool)
    valid_region[torso_y1:torso_y2, torso_x1:torso_x2] = True
    valid_region &= mask

    with Image.open(path) as opened_image:
        image_rgb = np.asarray(opened_image.convert("RGB"))

    pixels = image_rgb[valid_region]
    if len(pixels) < 50:
        return None

    float_pixels = pixels.astype(np.float32)
    luminance = (
        float_pixels[:, 0] * 0.2126 + float_pixels[:, 1] * 0.7152 + float_pixels[:, 2] * 0.0722
    )
    pixels = pixels[(luminance >= 20) & (luminance <= 235)]

    if len(pixels) < 50:
        return None

    rgb = tuple(int(round(value)) for value in np.median(pixels, axis=0))

    logger.debug("Segmented color: image=%s rgb=%s pixels=%s", path, rgb, len(pixels))

    return rgb


def _area(box: BoundingBox) -> int:
    return max(0, box.width) * max(0, box.height)


def _intersection_area(first: BoundingBox, second: BoundingBox) -> int:
    x1 = max(first.x, second.x)
    y1 = max(first.y, second.y)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    return width * height


def _represent_same_person(first: BoundingBox, second: BoundingBox) -> bool:
    first_area = _area(first)
    second_area = _area(second)

    if first_area <= 0 or second_area <= 0:
        return False

    intersection = _intersection_area(first, second)
    if intersection <= 0:
        return False

    union = first_area + second_area - intersection
    iou_value = intersection / union if union > 0 else 0
    smaller_coverage = intersection / min(first_area, second_area)

    return iou_value >= 0.30 or smaller_coverage >= 0.60


def combine_bounding_boxes(
    camera_boxes: list[BoundingBox],
    yolo_boxes: list[BoundingBox],
) -> list[BoundingBox]:
    if not yolo_boxes:
        return list(camera_boxes)

    result = list(yolo_boxes)

    for camera_box in camera_boxes:
        duplicate = any(
            _represent_same_person(camera_box, yolo_box) for yolo_box in yolo_boxes
        )
        if not duplicate:
            result.append(camera_box)

    result.sort(key=lambda box: (box.x + box.width / 2, box.y + box.height / 2))

    return result
