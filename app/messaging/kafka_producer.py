import json
import ssl
from typing import Any

from aiokafka import AIOKafkaProducer
from aiokafka.helpers import create_ssl_context

from app.core.config import settings


class KafkaPublisher:
    """
    Reusable Kafka producer.

    The connection is opened once with start() and closed with
    stop().
    """

    def __init__(self) -> None:
        self._producer: AIOKafkaProducer | None = None
        self._started: bool = False

    @property
    def started(self) -> bool:
        """Reports whether the producer has been started."""
        return self._started

    @staticmethod
    def _get_servers() -> list[str]:
        servers = [
            server.strip()
            for server in settings.kafka_bootstrap_servers.split(",")
            if server.strip()
        ]

        if not servers:
            raise ValueError("KAFKA_BOOTSTRAP_SERVERS was not configured")

        return servers

    @staticmethod
    def _create_ssl_context() -> ssl.SSLContext:
        if settings.kafka_ssl_cafile:
            return create_ssl_context(cafile=settings.kafka_ssl_cafile)

        return create_ssl_context()

    def _build_config(self) -> dict[str, Any]:
        protocol = settings.kafka_security_protocol.strip().upper()

        valid_protocols = {"PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"}

        if protocol not in valid_protocols:
            raise ValueError(f"Invalid Kafka protocol: {protocol}")

        config: dict[str, Any] = {
            "bootstrap_servers": self._get_servers(),
            "client_id": settings.kafka_client_id,
            "security_protocol": protocol,
            "request_timeout_ms": settings.kafka_request_timeout_ms,
            "max_request_size": settings.kafka_max_request_size,
            "acks": "all",
        }

        if protocol in {"SASL_PLAINTEXT", "SASL_SSL"}:
            if not settings.kafka_sasl_mechanism:
                raise ValueError("KAFKA_SASL_MECHANISM not configured")

            if not settings.kafka_sasl_username:
                raise ValueError("KAFKA_SASL_USERNAME not configured")

            if not settings.kafka_sasl_password:
                raise ValueError("KAFKA_SASL_PASSWORD not configured")

            config.update(
                {
                    "sasl_mechanism": settings.kafka_sasl_mechanism,
                    "sasl_plain_username": settings.kafka_sasl_username,
                    "sasl_plain_password": settings.kafka_sasl_password,
                }
            )

        if protocol in {"SSL", "SASL_SSL"}:
            config["ssl_context"] = self._create_ssl_context()

        return config

    async def start(self) -> None:
        """Opens the connection to Kafka. If already started, doesn't create another connection."""
        if self._started:
            print("[KafkaPublisher] Producer is already started")
            return

        config = self._build_config()

        print("[KafkaPublisher] Connecting to:", config["bootstrap_servers"])

        self._producer = AIOKafkaProducer(**config)

        try:
            await self._producer.start()
        except Exception:
            self._producer = None
            self._started = False
            raise

        self._started = True

        print("[KafkaPublisher] Connection established successfully")

    async def stop(self) -> None:
        """Closes the connection to Kafka."""
        if self._producer is None:
            self._started = False

            print("[KafkaPublisher] Producer is already stopped")
            return

        try:
            await self._producer.stop()
        finally:
            self._producer = None
            self._started = False

        print("[KafkaPublisher] Connection closed successfully")

    async def publish(
        self,
        topic: str,
        event_id: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Publishes a JSON event to Kafka.

        This function will be used in the next steps. Don't run it
        yet without confirming the topic.
        """
        if not self._started:
            raise RuntimeError("The Kafka producer has not been started")

        if self._producer is None:
            raise RuntimeError("Producer instance unavailable")

        if not topic.strip():
            raise ValueError("The Kafka topic was not provided")

        message = json.dumps(
            data,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")

        key = event_id.encode("utf-8")

        metadata = await self._producer.send_and_wait(
            topic=topic,
            key=key,
            value=message,
        )

        result = {
            "topic": metadata.topic,
            "partition": metadata.partition,
            "offset": metadata.offset,
            "timestamp": metadata.timestamp,
        }

        print(
            f"[KafkaPublisher] Event {event_id} published to topic {metadata.topic}, "
            f"partition {metadata.partition}, offset {metadata.offset}"
        )

        return result


kafka_publisher = KafkaPublisher()
