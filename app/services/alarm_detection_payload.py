import base64
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

from app.domain.models.alarm_detection_message import (
    AlarmDetectionMessage,
    AlarmDevice,
    AlarmEventData,
    AlarmImages,
)
from app.domain.models.camera_event import BoundingBox, CameraEvent


def file_to_base64(path: str) -> str:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Image not found: {file}")

    content = file.read_bytes()

    return base64.b64encode(content).decode("ascii")


def normalize_utc_date(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    return timestamp.astimezone(timezone.utc)


def date_to_milliseconds(timestamp: datetime) -> int:
    utc_date = normalize_utc_date(timestamp)

    return int(utc_date.timestamp() * 1000)


def date_to_iso8601(timestamp: datetime) -> str:
    utc_date = normalize_utc_date(timestamp)

    return utc_date.isoformat().replace("+00:00", "Z")


def crop_detection_base64(image_path: str, bounding_box: BoundingBox) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as image:
        image = image.convert("RGB")
        image_width, image_height = image.size

        x1 = max(0, min(bounding_box.x, image_width - 1))
        y1 = max(0, min(bounding_box.y, image_height - 1))
        x2 = max(x1 + 1, min(bounding_box.x2, image_width))
        y2 = max(y1 + 1, min(bounding_box.y2, image_height))

        crop = image.crop((x1, y1, x2, y2))

        buffer = BytesIO()
        crop.save(buffer, format="JPEG", quality=90, optimize=True)

        return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_alarm_detection_message(
    event: CameraEvent,
    bounding_box: BoundingBox | None = None,
    event_id: str | None = None,
    background_image_base64: str | None = None,
) -> AlarmDetectionMessage:
    """
    Builds the alarm payload.

    The optional parameters allow generating a separate payload for
    each person in the same scene, without changing the format
    required by Kafka.
    """
    if event.image is None:
        raise ValueError("The event has no image")

    original_path = event.image.original_path
    if not original_path:
        raise ValueError("The event has no original image path")

    selected_box = bounding_box or event.selected_bounding_box
    if selected_box is None:
        raise ValueError("The event has no selected bounding box")

    if background_image_base64 is None:
        background_image_base64 = file_to_base64(original_path)

    detection_image = crop_detection_base64(
        image_path=original_path,
        bounding_box=selected_box,
    )

    camera_name = event.camera_name or event.camera_id or "camera-desconhecida"
    final_id = event_id or event.event_id

    return AlarmDetectionMessage(
        device=AlarmDevice(name=camera_name),
        event=AlarmEventData(
            id=final_id,
            type="".join(part.capitalize() for part in event.event_type.split("_")),
            timestamp=date_to_milliseconds(event.timestamp),
            datetime=date_to_iso8601(event.timestamp),
            attributes=event.attributes,
            images=AlarmImages(
                background=background_image_base64,
                detection=detection_image,
            ),
        ),
    )
