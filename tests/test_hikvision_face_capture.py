import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from app.adapters.cameras import hikvision
from app.adapters.cameras.hikvision import HikvisionAdapter
from app.domain.models.camera_event import RawCameraPackage


FACE_CAPTURE_PAYLOAD = {
    "ipAddress": "192.168.101.215",
    "portNo": 8855,
    "protocol": "HTTP",
    "macAddress": "e0:ca:3c:ba:52:a0",
    "channelID": 1,
    "dateTime": "2026-08-19T23:34:19+08:00",
    "activePostCount": 1,
    "isDataRetransmission": False,
    "eventType": "faceCapture",
    "eventState": "active",
    "eventDescription": "faceCapture",
    "channelName": "Camera 01",
    "faceCapture": [
        {
            "targetAttrs": {
                "deviceName": "IP CAMERA",
                "deviceChannel": 1,
                "deviceId": "c4634000-12fd-11b3-8138-e0ca3cba52a0",
                "faceTime": "2026-08-19T23:34:19+08:00",
                "contentID": "backgroundImage",
                "pId": "20260819233422551006SaRwF7S6igl4",
            },
            "faces": [
                {
                    "faceId": 1960,
                    "faceRect": {
                        "height": 0.194,
                        "width": 0.087,
                        "x": 0.575,
                        "y": 0.4,
                    },
                    "contentID": "faceImage",
                    "pId": "2026081923342255100gyuEgvVVi0cBT",
                    "stayDuration": 25000,
                    "faceScore": 50,
                    "captureEndMark": False,
                    "FacePictureRect": {
                        "height": 0.071,
                        "width": 0.04,
                        "x": 0.6,
                        "y": 0.476,
                    },
                }
            ],
            "uid": "2026081923342255200ch70DGElR1XkQEvlDGPFri9ualogF3SBGo8x5PTIHRoq",
        }
    ],
}

BACKGROUND_SIZE = (1280, 720)
FACE_CROP_SIZE = (56, 140)


def _jpeg_bytes(size: tuple[int, int]) -> bytes:
    image = Image.new("RGB", size, color=(10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=80)
    return buffer.getvalue()


def _build_multipart_body(payload: dict, face_image: bytes, background_image: bytes) -> bytes:
    payload_bytes = json.dumps(payload).encode("utf-8")

    parts = [
        b"--boundary\r\n"
        b'Content-Disposition: form-data; name="faceCapture"\r\n'
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(payload_bytes)).encode() + b"\r\n\r\n"
        + payload_bytes + b"\r\n",
        b"--boundary\r\n"
        b'Content-Disposition: form-data; name="faceImage"; filename="face.jpg"\r\n'
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: " + str(len(face_image)).encode() + b"\r\n"
        b"Content-ID: face\r\n\r\n"
        + face_image + b"\r\n",
        b"--boundary\r\n"
        b'Content-Disposition: form-data; name="backgroundImage"; filename="bg.jpg"\r\n'
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: " + str(len(background_image)).encode() + b"\r\n"
        b"Content-ID: bg\r\n\r\n"
        + background_image + b"\r\n",
        b"--boundary--\r\n",
    ]
    return b"".join(parts)


def _build_multipart_body_face_named_like_json(
    payload: dict, face_image: bytes, background_image: bytes
) -> bytes:
    """
    Mirrors what webhook.site showed for the real device: the face crop
    is sent as a *file* part that reuses `name="faceCapture"`, the same
    name as the JSON metadata part (differing only by Content-Type and
    the `filename` attribute).
    """
    payload_bytes = json.dumps(payload).encode("utf-8")

    parts = [
        b"--boundary\r\n"
        b'Content-Disposition: form-data; name="faceCapture"\r\n'
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(payload_bytes)).encode() + b"\r\n\r\n"
        + payload_bytes + b"\r\n",
        b"--boundary\r\n"
        b'Content-Disposition: form-data; name="faceCapture"; filename="face.jpg"\r\n'
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: " + str(len(face_image)).encode() + b"\r\n\r\n"
        + face_image + b"\r\n",
        b"--boundary\r\n"
        b'Content-Disposition: form-data; name="backgroundImage"; filename="bg.jpg"\r\n'
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: " + str(len(background_image)).encode() + b"\r\n\r\n"
        + background_image + b"\r\n",
        b"--boundary--\r\n",
    ]
    return b"".join(parts)


@pytest.fixture
def redirect_images_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(hikvision, "IMAGES_FOLDER", tmp_path)
    return tmp_path


def test_can_handle_face_capture_multipart():
    adapter = HikvisionAdapter()
    body = _build_multipart_body(FACE_CAPTURE_PAYLOAD, b"\xff\xd8\xff\xd9", b"\xff\xd8\xff\xd9")

    assert adapter.can_handle("multipart/form-data; boundary=boundary", body) is True


def test_can_handle_face_capture_direct_json():
    adapter = HikvisionAdapter()
    body = json.dumps(FACE_CAPTURE_PAYLOAD).encode("utf-8")

    assert adapter.can_handle("application/json", body) is True


def test_normalize_face_capture_multipart_with_images(redirect_images_folder):
    adapter = HikvisionAdapter()
    face_image = _jpeg_bytes(FACE_CROP_SIZE)
    background_image = _jpeg_bytes(BACKGROUND_SIZE)
    body = _build_multipart_body(FACE_CAPTURE_PAYLOAD, face_image, background_image)

    package = RawCameraPackage(
        event_id="testface01",
        received_at=datetime.now(timezone.utc),
        content_type="multipart/form-data; boundary=boundary",
        camera_ip="192.168.101.215",
        package_path="memory://testface01",
    )

    event = adapter.normalize(package, body)

    assert event.manufacturer == "hikvision"
    assert event.event_type == "face_detection"
    assert event.target_type == "face"
    assert event.camera_id == "1"
    assert event.camera_name == "Camera 01"
    assert event.camera_ip == "192.168.101.215"
    assert event.timestamp == datetime.fromisoformat("2026-08-19T15:34:19+00:00")

    assert event.image is not None
    assert (event.image.width, event.image.height) == BACKGROUND_SIZE

    width, height = BACKGROUND_SIZE
    expected_x = round(0.575 * width)
    expected_y = round(0.4 * height)
    expected_width = round(0.087 * width)
    expected_height = round(0.194 * height)

    box = event.selected_bounding_box
    assert box is not None
    assert box.source == "faceRect"
    assert box.x == expected_x
    assert box.y == expected_y
    assert box.width == expected_width
    assert box.height == expected_height
    assert event.bounding_boxes == [box]

    assert event.attributes is not None
    assert event.attributes.vendor_event_type == "faceCapture"
    assert event.attributes.confidence == 0.5
    assert event.attributes.object_id == "1960"
    assert event.attributes.source_event_id == "20260819233422551006SaRwF7S6igl4"
    assert event.attributes.vendor_data["face_score"] == 50

    assert event.extra_data["face_crop_received"] is True
    face_crop_path = Path(event.extra_data["face_crop_path"])
    assert face_crop_path.is_file()
    assert Path(event.image.original_path).is_file()
    assert Path(event.image.annotated_path).is_file()


def test_normalize_face_capture_multipart_face_part_named_like_json(redirect_images_folder):
    """Real device format seen via webhook.site: the face image part also
    uses name="faceCapture", same as the JSON metadata part."""
    adapter = HikvisionAdapter()
    face_image = _jpeg_bytes(FACE_CROP_SIZE)
    background_image = _jpeg_bytes(BACKGROUND_SIZE)
    body = _build_multipart_body_face_named_like_json(
        FACE_CAPTURE_PAYLOAD, face_image, background_image
    )

    package = RawCameraPackage(
        event_id="testface03",
        received_at=datetime.now(timezone.utc),
        content_type="multipart/form-data; boundary=boundary",
        camera_ip="192.168.101.215",
        package_path="memory://testface03",
    )

    event = adapter.normalize(package, body)

    assert event.image is not None
    assert (event.image.width, event.image.height) == BACKGROUND_SIZE
    assert event.selected_bounding_box is not None

    assert event.extra_data["face_crop_received"] is True
    face_crop_path = Path(event.extra_data["face_crop_path"])
    assert face_crop_path.is_file()
    assert face_crop_path.read_bytes() == face_image
    assert Path(event.image.original_path).read_bytes() == background_image


def test_normalize_face_capture_direct_json_without_images():
    adapter = HikvisionAdapter()
    body = json.dumps(FACE_CAPTURE_PAYLOAD).encode("utf-8")

    package = RawCameraPackage(
        event_id="testface02",
        received_at=datetime.now(timezone.utc),
        content_type="application/json",
        camera_ip="192.168.101.215",
        package_path="memory://testface02",
    )

    event = adapter.normalize(package, body)

    assert event.image is None
    assert event.bounding_boxes == []
    assert event.selected_bounding_box is None
    assert event.extra_data["package_format"] == "direct_json"
