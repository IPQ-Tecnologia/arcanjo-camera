import asyncio
import ssl

from aiokafka import AIOKafkaProducer
from aiokafka.helpers import create_ssl_context

from app.core.config import settings


def get_servers() -> list[str]:
    servers = [
        server.strip()
        for server in settings.kafka_bootstrap_servers.split(",")
        if server.strip()
    ]

    if not servers:
        raise ValueError("KAFKA_BOOTSTRAP_SERVERS was not configured")

    return servers


def build_producer_config() -> dict:
    protocol = settings.kafka_security_protocol.upper()

    valid_protocols = {"PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"}

    if protocol not in valid_protocols:
        raise ValueError(f"Invalid Kafka protocol: {protocol}")

    config = {
        "bootstrap_servers": get_servers(),
        "client_id": settings.kafka_client_id,
        "security_protocol": protocol,
        "request_timeout_ms": settings.kafka_request_timeout_ms,
    }

    if protocol in {"SASL_PLAINTEXT", "SASL_SSL"}:
        if not settings.kafka_sasl_mechanism:
            raise ValueError("KAFKA_SASL_MECHANISM was not configured")

        if not settings.kafka_sasl_username:
            raise ValueError("KAFKA_SASL_USERNAME was not configured")

        if not settings.kafka_sasl_password:
            raise ValueError("KAFKA_SASL_PASSWORD was not configured")

        config.update(
            {
                "sasl_mechanism": settings.kafka_sasl_mechanism,
                "sasl_plain_username": settings.kafka_sasl_username,
                "sasl_plain_password": settings.kafka_sasl_password,
            }
        )

    if protocol in {"SSL", "SASL_SSL"}:
        ssl_context: ssl.SSLContext

        if settings.kafka_ssl_cafile:
            ssl_context = create_ssl_context(cafile=settings.kafka_ssl_cafile)
        else:
            ssl_context = create_ssl_context()

        config["ssl_context"] = ssl_context

    return config


async def test_connection() -> None:
    config = build_producer_config()

    print("===== KAFKA CONNECTION TEST =====")
    print("Servers:", config["bootstrap_servers"])
    print("Protocol:", config["security_protocol"])
    print("Client ID:", config["client_id"])

    producer = AIOKafkaProducer(**config)
    started = False

    try:
        print("Trying to connect...")

        await producer.start()
        started = True

        print("Connection to Kafka established successfully.")

    except Exception as error:
        print("Could not connect to Kafka.")
        print("Error type:", type(error).__name__)
        print("Details:", str(error))

        raise

    finally:
        if started:
            await producer.stop()
            print("Connection closed successfully.")

        print("===== END OF TEST =====")


if __name__ == "__main__":
    asyncio.run(test_connection())
