from __future__ import annotations

import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from app.adapters.cameras.base import CameraAdapter
from app.domain.models.camera_event import (
    BoundingBox,
    CameraEvent,
    EventAttributes,
    EventPoint,
    ImageData,
    RawCameraPackage,
)


IMAGES_FOLDER = Path("event_images")
DAHUA_NORMALIZED_SCALE = 8191.0


def get_boundary(content_type: str, body: bytes) -> str | None:
    match = re.search(
        r'boundary\s*=\s*["\']?([^;"\'\s]+)',
        content_type or "",
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    first_line = body.split(b"\r\n", 1)[0].strip()
    if first_line.startswith(b"--"):
        return first_line[2:].decode("utf-8", errors="ignore").strip()

    return None


def load_json(content: bytes) -> dict[str, Any]:
    text = content.decode("utf-8-sig", errors="ignore").strip()
    if not text:
        raise ValueError("Empty Dahua JSON")

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Dahua JSON is not an object")

    return data


def extract_jpeg(content: bytes) -> bytes | None:
    start = content.find(b"\xff\xd8\xff")
    if start == -1:
        return None

    end = content.find(b"\xff\xd9", start)
    if end == -1:
        return content[start:]

    return content[start : end + 2]


def extract_multipart(
    content_type: str,
    body: bytes,
) -> tuple[dict[str, Any], bytes | None]:
    boundary = get_boundary(content_type, body)
    if not boundary:
        raise ValueError("Dahua boundary not found")

    delimiter = b"--" + boundary.encode("utf-8", errors="ignore")

    json_payload: dict[str, Any] | None = None
    image: bytes | None = None

    for part in body.split(delimiter):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue

        if b"\r\n\r\n" not in part:
            continue

        header, content = part.split(b"\r\n\r\n", 1)
        lower_header = header.lower()

        if b"application/json" in lower_header or b"text/plain" in lower_header:
            try:
                json_payload = load_json(content)
            except (json.JSONDecodeError, ValueError):
                continue

        if b"image/jpeg" in lower_header:
            image = extract_jpeg(content)

    if json_payload is None:
        raise ValueError("JSON metadata not found in Dahua multipart")

    return json_payload, image


def select_event(payload: dict[str, Any]) -> dict[str, Any]:
    events = payload.get("Events")
    if isinstance(events, list) and events:
        first = events[0]
        if isinstance(first, dict):
            return first

    return payload


def convert_date_time(payload: dict[str, Any], event: dict[str, Any]) -> datetime:
    data = event.get("Data")
    if not isinstance(data, dict):
        data = {}

    for key in ("RealUTC", "UTC"):
        value = data.get(key)
        if value is None:
            continue

        try:
            timestamp = float(value)
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            continue

    text = payload.get("Time")
    if isinstance(text, str) and text.strip():
        try:
            date = datetime.strptime(text.strip(), "%Y-%m-%d %H:%M:%S")
            return date.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    return datetime.now(timezone.utc)


def sanitize_name(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())

    return result.strip("_") or "evento"


def build_base_name(date_time: datetime, event_type: str, event_id: str) -> str:
    time = date_time.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    type_ = sanitize_name(event_type)

    return f"{time}_{type_}_{event_id}"


def get_raw_bbox(obj: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = obj.get("BoundingBox")

    if isinstance(bbox, list) and len(bbox) >= 4:
        try:
            return tuple(float(value) for value in bbox[:4])
        except (TypeError, ValueError):
            return None

    if not isinstance(bbox, dict):
        return None

    bound_keys = (
        ("Left", "Top", "Right", "Bottom"),
        ("left", "top", "right", "bottom"),
        ("X1", "Y1", "X2", "Y2"),
        ("x1", "y1", "x2", "y2"),
    )

    for keys in bound_keys:
        if all(key in bbox for key in keys):
            try:
                return tuple(float(bbox[key]) for key in keys)
            except (TypeError, ValueError):
                return None

    size_keys = (
        ("X", "Y", "Width", "Height"),
        ("x", "y", "width", "height"),
    )

    for x_key, y_key, w_key, h_key in size_keys:
        if all(key in bbox for key in (x_key, y_key, w_key, h_key)):
            try:
                x = float(bbox[x_key])
                y = float(bbox[y_key])
                width = float(bbox[w_key])
                height = float(bbox[h_key])

                return (x, y, x + width, y + height)
            except (TypeError, ValueError):
                return None

    return None


def detect_normalized_scale(
    data: dict[str, Any],
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> bool:
    region = data.get("DetectRegion")
    region_coordinates: list[float] = []

    if isinstance(region, list):
        for point in region:
            if not isinstance(point, list):
                continue

            for value in point:
                try:
                    region_coordinates.append(float(value))
                except (TypeError, ValueError):
                    continue

    if region_coordinates:
        return max(region_coordinates) > max(image_width, image_height)

    x1, y1, x2, y2 = bbox

    return x1 > image_width or x2 > image_width or y1 > image_height or y2 > image_height


def convert_bbox_to_pixels(
    bbox: tuple[float, float, float, float],
    data: dict[str, Any],
    image_width: int,
    image_height: int,
) -> BoundingBox | None:
    x1, y1, x2, y2 = bbox

    maximum = max(abs(x1), abs(y1), abs(x2), abs(y2))

    if maximum <= 1.0:
        x1 *= image_width
        x2 *= image_width
        y1 *= image_height
        y2 *= image_height
    elif detect_normalized_scale(data, bbox, image_width, image_height):
        x1 = x1 / DAHUA_NORMALIZED_SCALE * image_width
        x2 = x2 / DAHUA_NORMALIZED_SCALE * image_width
        y1 = y1 / DAHUA_NORMALIZED_SCALE * image_height
        y2 = y2 / DAHUA_NORMALIZED_SCALE * image_height

    x1_int = max(0, min(image_width - 1, round(x1)))
    y1_int = max(0, min(image_height - 1, round(y1)))
    x2_int = max(0, min(image_width - 1, round(x2)))
    y2_int = max(0, min(image_height - 1, round(y2)))

    if x2_int <= x1_int or y2_int <= y1_int:
        return None

    width = x2_int - x1_int
    height = y2_int - y1_int
    ratio = width * height / (image_width * image_height)

    return BoundingBox(
        source="dahua_object_bounding_box",
        x=x1_int,
        y=y1_int,
        width=width,
        height=height,
        x2=x2_int,
        y2=y2_int,
        image_ratio=ratio,
    )


def save_images(
    image: bytes,
    base_name: str,
    bbox: BoundingBox | None,
) -> tuple[str, str, int, int]:
    IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)

    original_path = IMAGES_FOLDER / f"{base_name}_original.jpg"
    annotated_path = IMAGES_FOLDER / f"{base_name}_marcada.jpg"

    original_path.write_bytes(image)

    with Image.open(io.BytesIO(image)) as pil_image:
        rgb_image = pil_image.convert("RGB")
        image_width, image_height = rgb_image.size

        if bbox is not None:
            draw = ImageDraw.Draw(rgb_image)
            thickness = max(2, min(image_width, image_height) // 300)
            draw.rectangle(
                [bbox.x, bbox.y, bbox.x2, bbox.y2],
                outline="red",
                width=thickness,
            )

        rgb_image.save(annotated_path, format="JPEG", quality=95)

    return str(original_path), str(annotated_path), image_width, image_height


def _number_dahua_attributes(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def _text_dahua_attributes(value) -> str | None:
    if value is None:
        return None

    if isinstance(value, (dict, list, tuple)):
        return None

    text = str(value).strip()

    return text or None


def _normalize_dahua_coordinate(value) -> float | None:
    number = _number_dahua_attributes(value)
    if number is None:
        return None

    # The region points observed on Dahua use a 0-to-8191 scale.
    if number > 1:
        number = number / 8191

    return max(0.0, min(1.0, number))


def _extract_dahua_geometry(value) -> list[EventPoint]:
    points: list[EventPoint] = []

    def walk(item) -> None:
        if isinstance(item, dict):
            keys = {str(key).lower(): content for key, content in item.items()}

            x = None
            y = None

            for name in ("x", "positionx", "left"):
                if name in keys:
                    x = _normalize_dahua_coordinate(keys[name])
                    break

            for name in ("y", "positiony", "top"):
                if name in keys:
                    y = _normalize_dahua_coordinate(keys[name])
                    break

            if x is not None and y is not None:
                points.append(EventPoint(x=x, y=y))
                return

            for content in item.values():
                walk(content)

            return

        if isinstance(item, (list, tuple)):
            if len(item) >= 2:
                x = _normalize_dahua_coordinate(item[0])
                y = _normalize_dahua_coordinate(item[1])

                if x is not None and y is not None:
                    points.append(EventPoint(x=x, y=y))
                    return

            for content in item:
                walk(content)

    walk(value)

    # Removes duplicate points while keeping the order.
    result: list[EventPoint] = []
    seen: set[tuple[float, float]] = set()

    for point in points:
        signature = (round(point.x, 6), round(point.y, 6))
        if signature in seen:
            continue

        seen.add(signature)
        result.append(point)

    return result


def _normalize_dahua_target_type(target_detected: str | None) -> str | None:
    if not target_detected:
        return None

    target = str(target_detected).strip().lower()
    mapping = {
        "human": "person",
        "person": "person",
        "vehicle": "vehicle",
        "car": "vehicle",
        "automobile": "vehicle",
    }

    return mapping.get(target, target.replace(" ", "_").replace("-", "_"))


def _normalize_dahua_event_type(event_type: str, target_type: str | None) -> str:
    if target_type == "person":
        return "person_detection"

    if target_type == "vehicle":
        return "vehicle_detection"

    type_ = str(event_type or "").strip().lower()
    mapping = {
        "crossregiondetection": "intrusion_detection",
        "tripwiredetection": "line_crossing_detection",
        "leftdetection": "object_left_detection",
        "motiondetect": "motion_detection",
        "videoloss": "video_loss",
    }

    return mapping.get(type_, type_.replace(" ", "_").replace("-", "_") or "unknown_event")


def _dahua_event_category(event_type: str) -> str:
    normalized_type = event_type.strip().lower()

    categories = {
        "crossregiondetection": "intrusion",
        "leftdetection": "object_left",
        "tripwiredetection": "line_crossing",
        "motiondetect": "motion",
        "videoloss": "video_loss",
    }

    return categories.get(normalized_type, normalized_type)


def _build_dahua_attributes(
    payload: dict,
    data: dict,
    obj: dict,
    event_type: str,
    state: str | None,
    target_detected: str | None,
    original_action,
) -> EventAttributes:
    payload = payload if isinstance(payload, dict) else {}
    data = data if isinstance(data, dict) else {}
    obj = obj if isinstance(obj, dict) else {}

    geometry = _extract_dahua_geometry(
        data.get("DetectRegion") or data.get("Region") or data.get("Polygon")
    )

    if len(geometry) == 2:
        geometry_type = "line"
    elif len(geometry) >= 3:
        geometry_type = "region"
    else:
        geometry_type = None

    action = data.get("Action") or obj.get("Action") or original_action
    direction = data.get("Direction") or obj.get("Direction") or obj.get(
        "humanTripLineDirection"
    )
    normalized_direction = _text_dahua_attributes(direction)

    if normalized_direction and normalized_direction.lower() in {
        "0",
        "0.0",
        "unknown",
        "none",
    }:
        normalized_direction = None

    confidence = _number_dahua_attributes(obj.get("Confidence"))

    # In the captures analyzed, zero means the confidence wasn't made
    # available.
    if confidence is not None and confidence <= 0:
        confidence = None

    raw_bbox = get_raw_bbox(obj)

    return EventAttributes(
        manufacturer="dahua",
        vendor_event_type=event_type,
        category=_dahua_event_category(event_type),
        target_type=target_detected,
        state=state,
        action=_text_dahua_attributes(action),
        source_event_id=_text_dahua_attributes(data.get("EventID") or payload.get("EventID")),
        rule_id=_text_dahua_attributes(data.get("RuleID")),
        rule_name=_text_dahua_attributes(data.get("Name")),
        group_id=_text_dahua_attributes(data.get("GroupID")),
        object_id=_text_dahua_attributes(obj.get("ObjectID")),
        sensitivity=None,
        direction=normalized_direction,
        confidence=confidence,
        geometry_type=geometry_type,
        geometry=geometry,
        raw_bounding_box=(
            [float(value) for value in raw_bbox] if raw_bbox is not None else None
        ),
        vendor_data={
            key: value
            for key, value in {
                "vendor_target_type": _text_dahua_attributes(obj.get("ObjectType")),
                "vendor_state": state,
                "event_action": _text_dahua_attributes(original_action),
            }.items()
            if value is not None
        },
    )


class DahuaAdapter(CameraAdapter):
    manufacturer = "dahua"

    def can_handle(self, content_type: str, body: bytes) -> bool:
        type_ = (content_type or "").lower()
        content = body.lower()

        has_dahua_structure = (
            b'"code"' in content
            and b'"action"' in content
            and (
                b'"objecttype"' in content
                or b'"detectregion"' in content
                or b'"events"' in content
                or b'"picturetype"' in content
            )
        )

        return has_dahua_structure and (
            "application/json" in type_
            or "multipart/x-mixed-replace" in type_
            or content.startswith(b"{")
            or content.startswith(b"--")
        )

    def normalize(self, package: RawCameraPackage, body: bytes) -> CameraEvent:
        content_type = package.content_type or ""
        lower_content_type = content_type.lower()

        if "multipart/x-mixed-replace" in lower_content_type:
            package_format = "multipart"
            payload, image = extract_multipart(content_type, body)
        else:
            package_format = "direct_json"
            payload = load_json(body)
            image = None

        event = select_event(payload)

        data = event.get("Data")
        if not isinstance(data, dict):
            data = {}

        obj = data.get("Object")
        if not isinstance(obj, dict):
            obj = {}

        event_type = str(event.get("Code") or "desconhecido")
        original_action = str(event.get("Action") or data.get("Action") or "").strip()

        states = {
            "start": "active",
            "appear": "active",
            "pulse": "active",
            "stop": "inactive",
            "disappear": "inactive",
        }
        state = states.get(original_action.lower(), original_action.lower() or None)

        date_time = convert_date_time(payload, event)

        channel = payload.get("Channel")
        if channel is None:
            channel = event.get("Index")

        camera_id = str(channel) if channel is not None else None
        rule_name = data.get("Name")
        camera_name = f"Dahua Channel {camera_id}" if camera_id is not None else "Dahua"

        target = obj.get("ObjectType")
        target_detected = str(target).lower() if target is not None else None

        normalized_target = _normalize_dahua_target_type(target_detected)
        normalized_event_type = _normalize_dahua_event_type(event_type, normalized_target)

        image_data: ImageData | None = None
        bounding_box: BoundingBox | None = None

        original_path = None
        annotated_path = None
        image_width = None
        image_height = None

        raw_bbox = get_raw_bbox(obj)

        if image is not None:
            with Image.open(io.BytesIO(image)) as pil_image:
                image_width, image_height = pil_image.size

            if raw_bbox is not None:
                bounding_box = convert_bbox_to_pixels(raw_bbox, data, image_width, image_height)

            base_name = build_base_name(date_time, event_type, package.event_id)
            original_path, annotated_path, image_width, image_height = save_images(
                image, base_name, bounding_box
            )

            image_data = ImageData(
                width=image_width,
                height=image_height,
                format="jpeg",
                original_path=original_path,
                annotated_path=annotated_path,
            )

        bounding_boxes = [bounding_box] if bounding_box is not None else []

        return CameraEvent(
            event_id=package.event_id,
            manufacturer=self.manufacturer,
            camera_model=None,
            camera_id=camera_id,
            camera_name=camera_name,
            camera_ip=package.camera_ip,
            event_type=normalized_event_type,
            state=state,
            timestamp=date_time,
            target_type=normalized_target,
            attributes=_build_dahua_attributes(
                payload=payload,
                data=data,
                obj=obj,
                event_type=event_type,
                state=state,
                target_detected=target_detected,
                original_action=original_action,
            ),
            bounding_boxes=bounding_boxes,
            selected_bounding_box=bounding_box,
            image=image_data,
            extra_data={
                "package_format": package_format,
                "content_type": content_type,
                "original_action": original_action,
                "rule_name": rule_name,
                "dahua_event_id": data.get("EventID"),
                "group_id": data.get("GroupID"),
                "rule_id": data.get("RuleID"),
                "object_id": obj.get("ObjectID"),
                "confidence": obj.get("Confidence"),
                "raw_bounding_box": raw_bbox,
                "image_received": image is not None,
                "payload_resolution": payload.get("Resolution"),
            },
        )
