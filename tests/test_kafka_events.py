import pytest

from app.core.config import settings
from app.messaging.kafka_events import (
    publish_alarm_detection,
    publish_face_capture,
    publish_processing_error,
)


class FakePublisher:
    def __init__(self) -> None:
        self.calls = []

    async def publish(
        self,
        topic: str,
        event_id: str,
        data: dict,
    ) -> dict:
        self.calls.append(
            {
                "topic": topic,
                "event_id": event_id,
                "data": data,
            }
        )

        return {
            "topic": topic,
            "partition": 0,
            "offset": 1,
            "timestamp": None,
        }


@pytest.mark.asyncio
async def test_publish_alarm_detection_uses_alarm_topic():
    publisher = FakePublisher()
    payload = {"type": "alarm"}

    result = await publish_alarm_detection(
        publisher=publisher,
        event_id="event-alarm-1",
        data=payload,
    )

    assert len(publisher.calls) == 1
    assert publisher.calls[0]["topic"] == settings.kafka_topic_alarm_detection
    assert publisher.calls[0]["event_id"] == "event-alarm-1"
    assert publisher.calls[0]["data"] == payload
    assert result["topic"] == settings.kafka_topic_alarm_detection


@pytest.mark.asyncio
async def test_publish_face_capture_uses_face_topic():
    publisher = FakePublisher()
    payload = {"type": "face"}

    result = await publish_face_capture(
        publisher=publisher,
        event_id="event-face-1",
        data=payload,
    )

    assert len(publisher.calls) == 1
    assert publisher.calls[0]["topic"] == settings.kafka_topic_face_capture
    assert publisher.calls[0]["event_id"] == "event-face-1"
    assert publisher.calls[0]["data"] == payload
    assert result["topic"] == settings.kafka_topic_face_capture


@pytest.mark.asyncio
async def test_publish_processing_error_uses_error_topic():
    publisher = FakePublisher()
    payload = {
        "type": "camera_processing_error",
        "error": {
            "type": "RuntimeError",
            "message": "teste",
        },
    }

    result = await publish_processing_error(
        publisher=publisher,
        event_id="event-error-1",
        data=payload,
    )

    assert len(publisher.calls) == 1
    assert publisher.calls[0]["topic"] == settings.kafka_topic_errors
    assert publisher.calls[0]["event_id"] == "event-error-1"
    assert publisher.calls[0]["data"] == payload
    assert result["topic"] == settings.kafka_topic_errors
