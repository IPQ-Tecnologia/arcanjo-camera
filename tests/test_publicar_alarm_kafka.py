import asyncio
import json
from pathlib import Path

from app.core.config import settings
from app.messaging.kafka_producer import kafka_publisher


PAYLOAD_FILE = Path("payload_alarm_teste.json")


async def main() -> None:
    if not PAYLOAD_FILE.exists():
        raise FileNotFoundError("File payload_alarm_teste.json not found")

    data = json.loads(PAYLOAD_FILE.read_text(encoding="utf-8"))

    event_id = data["event"]["id"]
    topic = settings.kafka_topic_normalized

    message_size = len(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    print("===== KAFKA PUBLICATION =====")
    print("Topic:", topic)
    print("Event ID:", event_id)
    print("Size:", message_size, "bytes")

    try:
        await kafka_publisher.start()

        result = await kafka_publisher.publish(
            topic=topic,
            event_id=event_id,
            data=data,
        )

        print("Message published successfully.")
        print("Metadata:", result)

    finally:
        await kafka_publisher.stop()

    print("===== END OF PUBLICATION =====")


if __name__ == "__main__":
    asyncio.run(main())
