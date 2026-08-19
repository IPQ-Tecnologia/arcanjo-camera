import io
import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

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


logger = logging.getLogger(__name__)


COORDINATE_SCALE = None
FULL_IMAGE_BOX_LIMIT = 0.90
IMAGES_FOLDER = Path("event_images")
HIKVISION_NAMESPACE = {"ns": "http://www.hikvision.com/ver20/XMLSchema"}


def get_boundary(content_type: str, body: bytes) -> str | None:
    match = re.search(
        r"boundary\s*=\s*[\"']?([^;\"'\s]+)",
        content_type or "",
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    first_line = body.split(b"\r\n", 1)[0].strip()
    if first_line.startswith(b"--") and len(first_line) > 2:
        return first_line[2:].decode("utf-8", errors="ignore").strip()

    return None


def extract_image(body: bytes, boundary: str) -> bytes | None:
    for part in body.split(boundary.encode()):
        if b"Content-Type: image/jpeg" not in part:
            continue

        start = part.find(b"\xff\xd8\xff")
        if start == -1:
            continue

        end = part.find(b"\xff\xd9", start)
        if end != -1:
            return part[start : end + 2]

        return part[start:]

    return None


def extract_xml(body: bytes, boundary: str) -> str | None:
    for part in body.split(boundary.encode()):
        is_xml = b"Content-Type: application/xml" in part or b"Content-Type: text/xml" in part
        if not is_xml:
            continue

        start = part.find(b"\r\n\r\n")
        if start == -1:
            continue

        xml_bytes = part[start + 4 :]
        end = xml_bytes.rfind(b">")
        if end != -1:
            xml_bytes = xml_bytes[: end + 1]

        return xml_bytes.decode("utf-8", errors="ignore").strip()

    return None


def _part_name(part: bytes) -> str | None:
    # Matches the Content-Disposition `name="..."` attribute without being
    # fooled by the `filename="..."` attribute that often follows it.
    match = re.search(rb';\s*name="([^"]+)"', part, flags=re.IGNORECASE)
    if match:
        return match.group(1).decode("utf-8", errors="ignore")

    return None


def extract_jpeg_from_part(part: bytes | None) -> bytes | None:
    if not part:
        return None

    start = part.find(b"\xff\xd8\xff")
    if start == -1:
        return None

    end = part.find(b"\xff\xd9", start)
    if end != -1:
        return part[start : end + 2]

    return part[start:]


def extract_json_from_part(part: bytes | None) -> dict | None:
    if not part:
        return None

    start = part.find(b"\r\n\r\n")
    if start == -1:
        return None

    payload = part[start + 4 :].strip(b"\r\n")
    if payload.endswith(b"--"):
        payload = payload[:-2].strip(b"\r\n")

    try:
        return json.loads(payload.decode("utf-8", errors="ignore"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def is_face_capture_payload(body_lower: bytes) -> bool:
    return b'"eventtype"' in body_lower and (
        b'"facecapture"' in body_lower or b'"facesnap"' in body_lower
    )


def _is_face_capture_json(candidate: dict) -> bool:
    return str(candidate.get("eventType", "")).strip().lower() in ("facecapture", "facesnap")


def extract_face_capture_multipart(
    body: bytes, boundary: str
) -> tuple[dict, bytes | None, bytes | None] | None:
    """
    Splits a Hikvision `faceCapture` multipart package into its JSON
    metadata and its two JPEGs.

    Devices don't agree on the file part names: some send the face crop
    as `name="faceImage"`, others reuse `name="faceCapture"` for both the
    JSON metadata part *and* the image part (differentiated only by
    Content-Type/filename). So parts are classified by content
    (JSON vs. JPEG magic bytes), not only by name.
    """
    payload: dict | None = None
    jpeg_parts: list[tuple[str | None, bytes]] = []

    for part in body.split(boundary.encode()):
        if payload is None:
            candidate = extract_json_from_part(part)
            if candidate is not None and _is_face_capture_json(candidate):
                payload = candidate
                continue

        jpeg_bytes = extract_jpeg_from_part(part)
        if jpeg_bytes:
            jpeg_parts.append((_part_name(part), jpeg_bytes))

    if payload is None:
        return None

    background_image: bytes | None = None
    face_image: bytes | None = None
    leftover: list[tuple[str | None, bytes]] = []

    for name, data in jpeg_parts:
        normalized_name = (name or "").strip().lower()
        if normalized_name == "backgroundimage" and background_image is None:
            background_image = data
        elif normalized_name in ("faceimage", "facecapture") and face_image is None:
            face_image = data
        else:
            leftover.append((name, data))

    # Fallback for whatever is still missing: the background image is
    # always the larger JPEG (full scene vs. a small face crop).
    if leftover and (background_image is None or face_image is None):
        leftover.sort(key=lambda item: len(item[1]))
        if face_image is None:
            face_image = leftover.pop(0)[1]
        if background_image is None and leftover:
            background_image = leftover.pop()[1]

    return payload, face_image, background_image


def clean_direct_xml(body: bytes) -> str:
    xml = body.decode("utf-8-sig", errors="ignore").strip()

    start = xml.find("<")
    end = xml.rfind(">")
    if start == -1 or end == -1:
        raise ValueError("Invalid XML body")

    return xml[start : end + 1]


def convert_utc_date(date_time: str | None) -> datetime:
    if not date_time:
        utc_date = datetime.now(timezone.utc)
        logger.info("[HIKVISION DATE CONVERSION] date_time empty; using=%s", utc_date.isoformat())
        return utc_date

    logger.info("[HIKVISION DATE CONVERSION] original_date_time=%r", date_time)

    text = date_time.strip()
    logger.info("[HIKVISION DATE CONVERSION] text_after_strip=%r", text)

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    logger.info("[HIKVISION DATE CONVERSION] text_for_fromisoformat=%r", text)

    date = datetime.fromisoformat(text)
    logger.info(
        "[HIKVISION DATE CONVERSION] date_fromisoformat=%s tzinfo=%s",
        date.isoformat(),
        date.tzinfo,
    )

    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)

    utc_date = date.astimezone(timezone.utc)
    logger.info("[HIKVISION DATE CONVERSION] date_astimezone_utc=%s", utc_date.isoformat())

    return utc_date


def get_text(root: ET.Element, path: str, namespace: dict[str, str]) -> str | None:
    element = root.find(path, namespace)
    if element is not None and element.text:
        return element.text.strip()

    return None


def get_text_by_local_name(root: ET.Element, names: tuple[str, ...]) -> str | None:
    normalized_names = {name.lower() for name in names}

    for element in root.iter():
        if local_name(element.tag) not in normalized_names:
            continue

        if element.text and element.text.strip():
            return element.text.strip()

    return None


def local_name(tag: str) -> str:
    return tag.split("}")[-1].strip().lower()


def to_number(value) -> float | None:
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def first_value(values: dict[str, str], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name not in values:
            continue

        value = to_number(values[name])
        if value is not None:
            return value

    return None


def build_box(element: ET.Element) -> dict | None:
    values: dict[str, str] = {}

    for child in element.iter():
        if child is element or not child.text:
            continue

        key = local_name(child.tag)
        values.setdefault(key, child.text.strip())

    x = first_value(values, ("x", "positionx", "left", "xmin", "startx"))
    y = first_value(values, ("y", "positiony", "top", "ymin", "starty"))
    width = first_value(values, ("width", "largura", "boxwidth", "targetwidth"))
    height = first_value(values, ("height", "altura", "boxheight", "targetheight"))
    right = first_value(values, ("right", "xmax", "endx", "x2"))
    bottom = first_value(values, ("bottom", "ymax", "endy", "y2"))

    if width is None and x is not None and right is not None:
        width = right - x

    if height is None and y is not None and bottom is not None:
        height = bottom - y

    if any(value is None for value in (x, y, width, height)):
        return None

    if width <= 0 or height <= 0:
        return None

    return {
        "xml_source": local_name(element.tag),
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }


def extract_bounding_boxes(root: ET.Element) -> list[dict]:
    """
    Extracts only boxes that can represent the detected target.

    Hikvision also sends rectangles for the configured rule region.
    Those regions must not be treated as people.
    """
    priority_groups = (
        ("targetrect", "targetrectangle", "objectrect"),
        ("boundingbox", "bounding_box", "bounding-box"),
        ("detectionrect",),
    )

    for names in priority_groups:
        boxes: list[dict] = []
        seen: set[tuple[float, float, float, float]] = set()

        for element in root.iter():
            tag = local_name(element.tag)
            if tag not in names:
                continue

            box = build_box(element)
            if box is None:
                continue

            signature = tuple(round(box[key], 6) for key in ("x", "y", "width", "height"))
            if signature in seen:
                continue

            seen.add(signature)
            boxes.append(box)

        # Uses the first group that actually found boxes. targetRect
        # takes priority over generic regions.
        if boxes:
            return boxes

    return []


def get_xml_scale(root: ET.Element) -> tuple[float, float] | None:
    width = None
    height = None

    width_names = {"normalizedscreenwidth", "coordinatewidth", "referencewidth"}
    height_names = {"normalizedscreenheight", "coordinateheight", "referenceheight"}

    for element in root.iter():
        tag = local_name(element.tag)
        if tag in width_names and element.text:
            width = to_number(element.text)

        if tag in height_names and element.text:
            height = to_number(element.text)

    if width and height:
        return width, height

    return None


def determine_scale(
    boxes: list[dict],
    root: ET.Element,
    image_width: int,
    image_height: int,
) -> tuple[float, float]:
    if COORDINATE_SCALE == "pixels":
        return float(image_width), float(image_height)

    if isinstance(COORDINATE_SCALE, (int, float)):
        scale = float(COORDINATE_SCALE)
        return scale, scale

    xml_scale = get_xml_scale(root)
    if xml_scale:
        return xml_scale

    max_value = max(max(box["x"] + box["width"], box["y"] + box["height"]) for box in boxes)
    if max_value <= 1.5:
        return 1.0, 1.0

    exceeds_image = any(
        box["x"] + box["width"] > image_width or box["y"] + box["height"] > image_height
        for box in boxes
    )
    if exceeds_image and max_value <= 1100:
        return 1000.0, 1000.0

    return float(image_width), float(image_height)


def convert_boxes_to_pixels(
    boxes: list[dict],
    root: ET.Element,
    image_width: int,
    image_height: int,
) -> list[dict]:
    if not boxes:
        return []

    scale_x, scale_y = determine_scale(boxes, root, image_width, image_height)

    result: list[dict] = []

    for box in boxes:
        x = round(box["x"] * image_width / scale_x)
        y = round(box["y"] * image_height / scale_y)
        width = round(box["width"] * image_width / scale_x)
        height = round(box["height"] * image_height / scale_y)

        x = max(0, min(x, image_width - 1))
        y = max(0, min(y, image_height - 1))
        width = max(1, min(width, image_width - x))
        height = max(1, min(height, image_height - y))

        ratio = (width * height) / (image_width * image_height)

        result.append(
            {
                "xml_source": box["xml_source"],
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "x2": x + width,
                "y2": y + height,
                "image_ratio": round(ratio, 4),
            }
        )

    return result


def choose_bounding_box(pixel_boxes: list[dict]) -> dict | None:
    if not pixel_boxes:
        return None

    specific = [box for box in pixel_boxes if box["image_ratio"] < FULL_IMAGE_BOX_LIMIT]
    candidates = specific or pixel_boxes

    return min(candidates, key=lambda box: box["width"] * box["height"])


def sanitize_name(value, default: str) -> str:
    text = str(value or default)
    return "".join(
        character if character.isalnum() or character in "-_" else "_" for character in text
    )


def build_base_name(date_time: datetime, event_type: str | None, event_id: str) -> str:
    date_name = date_time.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    type_name = sanitize_name(event_type, "evento")

    return f"{date_name}_{type_name}_{event_id}"


def save_original_image(image: bytes, base_name: str) -> str:
    IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
    path = IMAGES_FOLDER / f"{base_name}_original.jpg"
    path.write_bytes(image)

    return str(path)


def save_annotated_image(image: bytes, base_name: str, box: dict | None) -> str | None:
    if box is None:
        return None

    IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
    path = IMAGES_FOLDER / f"{base_name}_marcada.jpg"

    with Image.open(io.BytesIO(image)) as pil_image:
        pil_image = pil_image.convert("RGB")
        draw = ImageDraw.Draw(pil_image)
        draw.rectangle(
            [box["x"], box["y"], box["x2"], box["y2"]],
            outline="red",
            width=5,
        )
        pil_image.save(path, "JPEG", quality=95)

    return str(path)


def convert_box_to_model(box: dict | None) -> BoundingBox | None:
    if box is None:
        return None

    return BoundingBox(
        source=box["xml_source"],
        x=box["x"],
        y=box["y"],
        width=box["width"],
        height=box["height"],
        x2=box["x2"],
        y2=box["y2"],
        image_ratio=box.get("image_ratio"),
    )


def _local_name_hikvision_attributes(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _first_text_hikvision_attributes(root: ET.Element, names: tuple[str, ...]) -> str | None:
    normalized_names = {name.lower() for name in names}

    for element in root.iter():
        name = _local_name_hikvision_attributes(element.tag)
        if name not in normalized_names:
            continue

        if element.text and element.text.strip():
            return element.text.strip()

    return None


def _number_hikvision_attributes(value) -> float | None:
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def _normalize_hikvision_coordinate(value) -> float | None:
    number = _number_hikvision_attributes(value)
    if number is None:
        return None

    # Hikvision region coordinates normally use a 0-to-1000 scale.
    if number > 1:
        number = number / 1000

    return max(0.0, min(1.0, number))


def _extract_hikvision_geometry(root: ET.Element) -> list[EventPoint]:
    points: list[EventPoint] = []

    for element in root.iter():
        if _local_name_hikvision_attributes(element.tag) != "regioncoordinates":
            continue

        x = None
        y = None

        for child in element.iter():
            name = _local_name_hikvision_attributes(child.tag)
            if not child.text:
                continue

            if name == "positionx":
                x = _normalize_hikvision_coordinate(child.text)
            elif name == "positiony":
                y = _normalize_hikvision_coordinate(child.text)

        if x is not None and y is not None:
            points.append(EventPoint(x=x, y=y))

    return points


def _normalize_hikvision_target_type(
    target_detected: str | None,
    event_type: str | None = None,
) -> str | None:
    if target_detected:
        target = str(target_detected).strip().lower()
        mapping = {
            "human": "person",
            "person": "person",
            "vehicle": "vehicle",
            "car": "vehicle",
            "face": "face",
        }
        return mapping.get(target, target.replace(" ", "_").replace("-", "_"))

    type_ = str(event_type or "").strip().lower()
    if type_ == "facedetection":
        return "face"

    return None


def _normalize_hikvision_event_type(event_type: str, target_type: str | None) -> str:
    if target_type == "person":
        return "person_detection"

    if target_type == "vehicle":
        return "vehicle_detection"

    if target_type == "face":
        return "face_detection"

    type_ = str(event_type or "").strip().lower()
    mapping = {
        "fielddetection": "intrusion_detection",
        "linedetection": "line_crossing_detection",
        "vmd": "motion_detection",
        "videoloss": "video_loss",
        "facedetection": "face_detection",
        "duration": "duration",
    }

    return mapping.get(type_, type_.replace(" ", "_").replace("-", "_") or "unknown_event")


def _hikvision_event_category(event_type: str) -> str:
    categories = {
        "fielddetection": "intrusion",
        "linedetection": "line_crossing",
        "vmd": "motion",
        "videoloss": "video_loss",
    }
    normalized_type = event_type.strip().lower()

    return categories.get(normalized_type, normalized_type)


def _build_hikvision_attributes(
    root: ET.Element,
    event_type: str,
    state: str | None,
    target_detected: str | None,
) -> EventAttributes:
    geometry = _extract_hikvision_geometry(root)
    normalized_type = event_type.strip().lower()

    if normalized_type == "linedetection":
        geometry_type = "line"
    elif normalized_type == "fielddetection":
        geometry_type = "region"
    elif len(geometry) == 2:
        geometry_type = "line"
    elif len(geometry) >= 3:
        geometry_type = "region"
    else:
        geometry_type = None

    source_event_id = None
    for name, value in root.attrib.items():
        if _local_name_hikvision_attributes(name) == "id":
            source_event_id = str(value)
            break

    sensitivity = _number_hikvision_attributes(
        _first_text_hikvision_attributes(root, ("sensitivityLevel",))
    )

    return EventAttributes(
        manufacturer="hikvision",
        vendor_event_type=event_type,
        category=_hikvision_event_category(event_type),
        target_type=target_detected,
        state=state,
        action=None,
        source_event_id=source_event_id,
        rule_id=_first_text_hikvision_attributes(root, ("regionID", "ruleID")),
        rule_name=_first_text_hikvision_attributes(
            root, ("ruleName", "regionName", "lineName")
        ),
        object_id=None,
        sensitivity=sensitivity,
        direction=_first_text_hikvision_attributes(
            root, ("direction", "targetDirection", "lineCrossingDirection")
        ),
        confidence=None,
        geometry_type=geometry_type,
        geometry=geometry,
        vendor_data={
            key: value
            for key, value in {
                "vendor_target_type": target_detected,
                "vendor_state": state,
            }.items()
            if value is not None
        },
    )


def _first_face_capture_item(payload: dict) -> dict:
    items = payload.get("faceCapture")

    if isinstance(items, list) and items:
        return items[0]

    if isinstance(items, dict):
        return items

    return {}


def _first_face(item: dict) -> dict:
    faces = item.get("faces")
    if isinstance(faces, list) and faces:
        return faces[0]

    return {}


def _face_rect_to_pixel_box(
    face_rect: dict,
    image_width: int,
    image_height: int,
) -> dict | None:
    x = to_number(face_rect.get("x"))
    y = to_number(face_rect.get("y"))
    width = to_number(face_rect.get("width"))
    height = to_number(face_rect.get("height"))

    if any(value is None for value in (x, y, width, height)):
        return None

    if width <= 0 or height <= 0:
        return None

    pixel_x = round(x * image_width)
    pixel_y = round(y * image_height)
    pixel_width = round(width * image_width)
    pixel_height = round(height * image_height)

    pixel_x = max(0, min(pixel_x, image_width - 1))
    pixel_y = max(0, min(pixel_y, image_height - 1))
    pixel_width = max(1, min(pixel_width, image_width - pixel_x))
    pixel_height = max(1, min(pixel_height, image_height - pixel_y))

    ratio = (pixel_width * pixel_height) / (image_width * image_height)

    return {
        "xml_source": "faceRect",
        "x": pixel_x,
        "y": pixel_y,
        "width": pixel_width,
        "height": pixel_height,
        "x2": pixel_x + pixel_width,
        "y2": pixel_y + pixel_height,
        "image_ratio": round(ratio, 4),
    }


def save_face_crop_image(image: bytes, base_name: str) -> str:
    IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
    path = IMAGES_FOLDER / f"{base_name}_facecrop.jpg"
    path.write_bytes(image)

    return str(path)


def _build_hikvision_face_attributes(
    state: str | None,
    source_event_id: str | None,
    face_id,
    face_score: float | None,
    ip_address: str | None,
    mac_address: str | None,
    device_id: str | None,
    stay_duration: float | None,
) -> EventAttributes:
    confidence = None
    if face_score is not None:
        confidence = round(face_score / 100, 4) if face_score > 1 else face_score

    vendor_data = {
        key: value
        for key, value in {
            "vendor_ip_address": ip_address,
            "vendor_mac_address": mac_address,
            "vendor_device_id": device_id,
            "face_score": face_score,
            "stay_duration_ms": stay_duration,
        }.items()
        if value is not None
    }

    return EventAttributes(
        manufacturer="hikvision",
        vendor_event_type="faceCapture",
        category="face_capture",
        target_type="face",
        state=state,
        action=None,
        source_event_id=source_event_id,
        rule_id=None,
        rule_name=None,
        object_id=str(face_id) if face_id is not None else None,
        sensitivity=None,
        direction=None,
        confidence=confidence,
        geometry_type=None,
        geometry=[],
        vendor_data=vendor_data,
    )


def build_face_capture_event(
    package: RawCameraPackage,
    payload: dict,
    face_image: bytes | None,
    background_image: bytes | None,
    package_format: str,
    content_type: str,
) -> CameraEvent:
    """
    Normalizes a Hikvision `faceCapture` event (JSON payload, either as a
    standalone body or as the `faceCapture` part of a multipart package
    that also carries `faceImage`/`backgroundImage` JPEGs).
    """
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    event_type = payload.get("eventType") or "faceCapture"
    state = payload.get("eventState")
    date_time = convert_utc_date(payload.get("dateTime"))
    camera_name = payload.get("channelName")
    channel_id = payload.get("channelID")
    camera_id = str(channel_id) if channel_id is not None else None
    ip_address = payload.get("ipAddress")
    mac_address = payload.get("macAddress")

    capture_item = _first_face_capture_item(payload)
    target_attrs = capture_item.get("targetAttrs") or {}
    face = _first_face(capture_item)
    face_rect = face.get("faceRect") or {}

    face_id = face.get("faceId")
    face_score = to_number(face.get("faceScore"))
    stay_duration = to_number(face.get("stayDuration"))
    device_id = target_attrs.get("deviceId")
    source_event_id = target_attrs.get("pId") or capture_item.get("uid")

    image_width = None
    image_height = None
    original_path = None
    annotated_path = None
    face_crop_path = None
    image_data = None
    pixel_box = None

    if background_image is not None:
        with Image.open(io.BytesIO(background_image)) as pil_image:
            image_width, image_height = pil_image.size

        pixel_box = _face_rect_to_pixel_box(face_rect, image_width, image_height)

        base_name = build_base_name(date_time, event_type, package.event_id)
        original_path = save_original_image(background_image, base_name)
        annotated_path = save_annotated_image(background_image, base_name, pixel_box)

        if face_image is not None:
            face_crop_path = save_face_crop_image(face_image, base_name)

        image_data = ImageData(
            width=image_width,
            height=image_height,
            format="jpeg",
            original_path=original_path,
            annotated_path=annotated_path,
            face_crop_path=face_crop_path,
        )

    model_box = convert_box_to_model(pixel_box)

    return CameraEvent(
        event_id=package.event_id,
        manufacturer="hikvision",
        camera_model=None,
        camera_id=camera_id,
        camera_name=camera_name,
        camera_ip=package.camera_ip,
        event_type="face_detection",
        state=state,
        timestamp=date_time,
        target_type="face",
        attributes=_build_hikvision_face_attributes(
            state=state,
            source_event_id=source_event_id,
            face_id=face_id,
            face_score=face_score,
            ip_address=ip_address,
            mac_address=mac_address,
            device_id=device_id,
            stay_duration=stay_duration,
        ),
        bounding_boxes=[model_box] if model_box is not None else [],
        selected_bounding_box=model_box,
        image=image_data,
        extra_data={
            "package_format": package_format,
            "image_received": background_image is not None,
            "bounding_box_count": 1 if model_box is not None else 0,
            "content_type": content_type,
            "face_crop_path": face_crop_path,
            "face_crop_received": face_image is not None,
        },
    )


class HikvisionAdapter(CameraAdapter):
    manufacturer = "hikvision"

    def can_handle(self, content_type: str, body: bytes) -> bool:
        body_lower = body.lower()

        return (
            b"hikvision.com" in body_lower
            or b"eventnotificationalert" in body_lower
            or b"<eventtype>" in body_lower
            or is_face_capture_payload(body_lower)
        )

    def normalize(self, package: RawCameraPackage, body: bytes) -> CameraEvent:
        content_type = package.content_type or ""
        lower_content_type = content_type.lower()
        direct_xml = "application/xml" in lower_content_type or "text/xml" in lower_content_type
        direct_json = not direct_xml and "application/json" in lower_content_type

        image: bytes | None = None

        if direct_json:
            payload = json.loads(body.decode("utf-8", errors="ignore"))
            if is_face_capture_payload(body.lower()):
                return build_face_capture_event(
                    package=package,
                    payload=payload,
                    face_image=None,
                    background_image=None,
                    package_format="direct_json",
                    content_type=content_type,
                )

        if direct_xml:
            package_format = "direct_xml"
            xml = clean_direct_xml(body)
        else:
            package_format = "multipart"
            boundary = get_boundary(content_type, body)
            if not boundary:
                raise ValueError("Boundary not found")

            face_capture_result = extract_face_capture_multipart(body, boundary)
            if face_capture_result is not None:
                payload, face_image, background_image = face_capture_result
                return build_face_capture_event(
                    package=package,
                    payload=payload,
                    face_image=face_image,
                    background_image=background_image,
                    package_format="multipart_json",
                    content_type=content_type,
                )

            xml = extract_xml(body, boundary)
            image = extract_image(body, boundary)
            if xml is None:
                raise ValueError("XML not found in multipart")

        root = ET.fromstring(xml)

        event_type = get_text(root, "ns:eventType", HIKVISION_NAMESPACE) or "desconhecido"
        date_time_text = get_text(root, "ns:dateTime", HIKVISION_NAMESPACE)
        date_time = convert_utc_date(date_time_text)
        state = get_text(root, "ns:eventState", HIKVISION_NAMESPACE)
        camera_name = get_text(root, "ns:channelName", HIKVISION_NAMESPACE)
        camera_id = get_text(root, "ns:channelID", HIKVISION_NAMESPACE)
        camera_model = get_text_by_local_name(root, ("deviceModel", "model", "modelName"))
        target_detected = get_text(root, ".//ns:detectionTarget", HIKVISION_NAMESPACE)

        normalized_target = _normalize_hikvision_target_type(target_detected, event_type)
        normalized_event_type = _normalize_hikvision_event_type(event_type, normalized_target)

        xml_boxes = extract_bounding_boxes(root)
        pixel_boxes: list[dict] = []
        chosen_box: dict | None = None

        image_width = None
        image_height = None
        original_path = None
        annotated_path = None
        image_data = None

        if image is not None:
            with Image.open(io.BytesIO(image)) as pil_image:
                image_width, image_height = pil_image.size

            pixel_boxes = convert_boxes_to_pixels(xml_boxes, root, image_width, image_height)
            chosen_box = choose_bounding_box(pixel_boxes)

            base_name = build_base_name(date_time, event_type, package.event_id)
            original_path = save_original_image(image, base_name)
            annotated_path = save_annotated_image(image, base_name, chosen_box)

            image_data = ImageData(
                width=image_width,
                height=image_height,
                format="jpeg",
                original_path=original_path,
                annotated_path=annotated_path,
            )

        model_boxes = [convert_box_to_model(box) for box in pixel_boxes]

        return CameraEvent(
            event_id=package.event_id,
            manufacturer=self.manufacturer,
            camera_model=camera_model,
            camera_id=camera_id,
            camera_name=camera_name,
            camera_ip=package.camera_ip,
            event_type=normalized_event_type,
            state=state,
            timestamp=date_time,
            target_type=normalized_target,
            attributes=_build_hikvision_attributes(
                root=root,
                event_type=event_type,
                state=state,
                target_detected=target_detected,
            ),
            bounding_boxes=[box for box in model_boxes if box is not None],
            selected_bounding_box=convert_box_to_model(chosen_box),
            image=image_data,
            extra_data={
                "package_format": package_format,
                "image_received": image is not None,
                "bounding_box_count": len(pixel_boxes),
                "xml_bounding_boxes": xml_boxes,
                "content_type": content_type,
            },
        )
