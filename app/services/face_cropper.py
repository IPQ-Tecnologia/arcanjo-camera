from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceCropResult:
    image_base64: str
    file_path: str
    x: int
    y: int
    width: int
    height: int
    score: float

    def to_dict(self) -> dict:
        return {
            "image_base64": self.image_base64,
            "file_path": self.file_path,
            "score": self.score,
            "bounding_box": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
        }


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODEL_PATH = (
    _PROJECT_ROOT / "models" / "face_detection" / "face_detection_yunet_2023mar.onnx"
)
_DETECTOR_LOCK = Lock()


def _create_detector():
    if not _MODEL_PATH.exists():
        raise RuntimeError(f"Missing YuNet model: {_MODEL_PATH}")

    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("This OpenCV version doesn't have cv2.FaceDetectorYN")

    return cv2.FaceDetectorYN.create(str(_MODEL_PATH), "", (320, 320), 0.70, 0.30, 5000)


_DETECTOR = _create_detector()


def _clamp_box(
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1 = max(0, min(x, image_width - 1))
    y1 = max(0, min(y, image_height - 1))
    x2 = max(x1 + 1, min(x + width, image_width))
    y2 = max(y1 + 1, min(y + height, image_height))

    return x1, y1, x2, y2


def _detect_faces_yunet(
    person_crop: np.ndarray,
) -> list[tuple[int, int, int, int, float]]:
    person_height, person_width = person_crop.shape[:2]

    if person_width < 35 or person_height < 70:
        return []

    # The face should be mostly in the upper part of the person's box.
    region_height = max(1, int(person_height * 0.68))
    head_region = person_crop[0:region_height, 0:person_width]
    region_height, region_width = head_region.shape[:2]

    # Upscales small crops to help detect distant faces.
    scale = min(3.0, max(1.0, 240 / max(1, region_width), 240 / max(1, region_height)))

    if scale > 1.0:
        analysis_image = cv2.resize(
            head_region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    else:
        analysis_image = head_region

    analysis_height, analysis_width = analysis_image.shape[:2]

    # The pipeline has several workers. Protects the shared detector
    # because setInputSize changes its internal state.
    with _DETECTOR_LOCK:
        _DETECTOR.setInputSize((analysis_width, analysis_height))
        _, detections = _DETECTOR.detect(analysis_image)

    if detections is None:
        return []

    results: list[tuple[int, int, int, int, float]] = []

    for detection in detections:
        score = float(detection[-1])
        if score < 0.70:
            continue

        x = int(round(float(detection[0]) / scale))
        y = int(round(float(detection[1]) / scale))
        face_width = int(round(float(detection[2]) / scale))
        face_height = int(round(float(detection[3]) / scale))

        x1, y1, x2, y2 = _clamp_box(
            x=x,
            y=y,
            width=face_width,
            height=face_height,
            image_width=region_width,
            image_height=region_height,
        )
        face_width = x2 - x1
        face_height = y2 - y1

        if face_width < 12 or face_height < 12:
            continue

        ratio = face_width / max(1, face_height)
        width_ratio = face_width / max(1, person_width)
        height_ratio = face_height / max(1, person_height)
        center_y = (y1 + face_height / 2) / max(1, person_height)

        # Additional filters against random room objects.
        if not 0.60 <= ratio <= 1.55:
            continue

        if not 0.07 <= width_ratio <= 0.90:
            continue

        if not 0.04 <= height_ratio <= 0.50:
            continue

        if center_y > 0.67:
            continue

        results.append((x1, y1, face_width, face_height, score))

    results.sort(key=lambda item: (item[4], item[2] * item[3]), reverse=True)

    return results


def crop_face(
    image_path: str,
    bounding_box: Any | None = None,
    output_folder: str = "face_crops",
    file_name: str | None = None,
) -> FaceCropResult | None:
    # Never detects a face on the full scene. The confirmed person box
    # is required.
    if bounding_box is None:
        return None

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Invalid image: {image_path}")

    image_height, image_width = image.shape[:2]

    person_x = int(bounding_box.x)
    person_y = int(bounding_box.y)
    person_width = int(bounding_box.width)
    person_height = int(bounding_box.height)

    # Adds a small margin around the person. Some boxes start below
    # the hair or cut off part of the head at the edges.
    margin_x = int(person_width * 0.10)
    top_margin = int(person_height * 0.16)
    bottom_margin = int(person_height * 0.03)

    x1, y1, x2, y2 = _clamp_box(
        x=person_x - margin_x,
        y=person_y - top_margin,
        width=person_width + margin_x * 2,
        height=person_height + top_margin + bottom_margin,
        image_width=image_width,
        image_height=image_height,
    )

    person_crop = image[y1:y2, x1:x2]
    if person_crop.size == 0:
        return None

    faces = _detect_faces_yunet(person_crop)
    if not faces:
        return None

    # Position of the person's original bounding box within the
    # upscaled crop.
    local_person_x = person_x - x1
    local_person_y = person_y - y1

    # The expected face should be close to the horizontal center and
    # the upper region of this person.
    expected_center_x = local_person_x + person_width / 2
    expected_center_y = local_person_y + person_height * 0.18

    # Accepts a small lateral tolerance because some bounding boxes
    # cut off part of the head.
    x_tolerance = max(12, int(person_width * 0.15))
    min_x_limit = local_person_x - x_tolerance
    max_x_limit = local_person_x + person_width + x_tolerance

    face_candidates = []

    for face in faces:
        fx, fy, fw, fh, fscore = face

        face_center_x = fx + fw / 2
        face_center_y = fy + fh / 2

        if not (min_x_limit <= face_center_x <= max_x_limit):
            continue

        # Avoids matching a face far below this person's expected
        # head region.
        if face_center_y > local_person_y + person_height * 0.48:
            continue

        distance_x = abs(face_center_x - expected_center_x) / max(1, person_width)
        distance_y = abs(face_center_y - expected_center_y) / max(1, person_height)
        distance = distance_x * 1.5 + distance_y

        face_candidates.append((distance, -fscore, face))

    if not face_candidates:
        return None

    face_candidates.sort(key=lambda item: (item[0], item[1]))
    face_x, face_y, face_width, face_height, _score = face_candidates[0][2]

    face_margin_x = int(face_width * 0.32)
    face_top_margin = int(face_height * 0.36)
    face_bottom_margin = int(face_height * 0.42)

    rx1 = max(0, face_x - face_margin_x)
    ry1 = max(0, face_y - face_top_margin)
    rx2 = min(person_crop.shape[1], face_x + face_width + face_margin_x)
    ry2 = min(person_crop.shape[0], face_y + face_height + face_bottom_margin)

    face_crop = person_crop[ry1:ry2, rx1:rx2]
    if face_crop.size == 0:
        return None

    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)

    if not file_name:
        file_name = Path(image_path).stem + "_face.jpg"

    if not file_name.lower().endswith((".jpg", ".jpeg")):
        file_name += ".jpg"

    # Upscales small face crops before saving and converting to
    # Base64.
    crop_height, crop_width = face_crop.shape[:2]
    min_width = 300

    if crop_width < min_width:
        scale_factor = min_width / crop_width
        new_width = min_width
        new_height = max(1, round(crop_height * scale_factor))

        face_crop = cv2.resize(
            face_crop, (new_width, new_height), interpolation=cv2.INTER_CUBIC
        )

    output_path = folder / file_name

    saved = cv2.imwrite(str(output_path), face_crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not saved:
        raise RuntimeError("Could not save the face crop")

    encoded, buffer = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not encoded:
        raise RuntimeError("Could not encode the face crop")

    image_base64 = base64.b64encode(buffer.tobytes()).decode("ascii")

    return FaceCropResult(
        image_base64=image_base64,
        file_path=str(output_path),
        score=round(float(_score), 6),
        x=x1 + rx1,
        y=y1 + ry1,
        width=rx2 - rx1,
        height=ry2 - ry1,
    )
