from typing import Any

from app.core.config import settings
from app.messaging.kafka_producer import KafkaPublisher


async def publish_alarm_detection(
    publisher: KafkaPublisher,
    event_id: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Publishes an alarm detection event."""
    return await publisher.publish(
        topic=settings.kafka_topic_alarm_detection,
        event_id=event_id,
        data=data,
    )


async def publish_face_capture(
    publisher: KafkaPublisher,
    event_id: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Publishes a face capture event."""
    return await publisher.publish(
        topic=settings.kafka_topic_face_capture,
        event_id=event_id,
        data=data,
    )


async def publish_processing_error(
    publisher: KafkaPublisher,
    event_id: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Publishes a camera package processing error."""
    return await publisher.publish(
        topic=settings.kafka_topic_errors,
        event_id=event_id,
        data=data,
    )

