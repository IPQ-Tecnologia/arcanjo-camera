import json
from datetime import datetime
from pathlib import Path

from app.domain.models.camera_event import BoundingBox, CameraEvent, ImageData
from app.services.alarm_detection_payload import build_alarm_detection_message


ORIGINAL_PATH = "event_images/20260717T175437Z_linedetection_288470f11d7a_original.jpg"


event = CameraEvent(
    event_id="288470f11d7a",
    manufacturer="hikvision",
    camera_model=None,
    camera_id="1",
    camera_name="Camera 01",
    camera_ip="192.168.101.214",
    event_type="linedetection",
    state="active",
    timestamp=datetime.fromisoformat("2026-07-17T17:54:37+00:00"),
    target_type="human",
    bounding_boxes=[
        BoundingBox(
            source="targetrect",
            x=604,
            y=262,
            width=105,
            height=190,
            x2=709,
            y2=452,
            image_ratio=0.0216,
        )
    ],
    selected_bounding_box=BoundingBox(
        source="targetrect",
        x=604,
        y=262,
        width=105,
        height=190,
        x2=709,
        y2=452,
        image_ratio=0.0216,
    ),
    image=ImageData(
        width=1280,
        height=720,
        format="jpeg",
        original_path=ORIGINAL_PATH,
        annotated_path="event_images/20260717T175437Z_linedetection_288470f11d7a_marcada.jpg",
    ),
)


payload = build_alarm_detection_message(event)

data = payload.model_dump(mode="json")

output_path = Path("payload_alarm_teste.json")

output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

print("===== PAYLOAD CREATED =====")
print("Device:", payload.device.name)
print("Event ID:", payload.event.id)
print("Event type:", payload.event.type)
print("Timestamp:", payload.event.timestamp)
print("Datetime:", payload.event.datetime)

print("Background Base64 size:", len(payload.event.images.background), "characters")
print("Detection Base64 size:", len(payload.event.images.detection), "characters")

print("File saved:", output_path)
print("==========================")
