import asyncio

from app.messaging.kafka_producer import kafka_publisher


async def main() -> None:
    print("===== KAFKA PUBLISHER TEST =====")
    print("Started before:", kafka_publisher.started)

    try:
        await kafka_publisher.start()

        print("Started after start:", kafka_publisher.started)

        print("Reusable producer working.")

    finally:
        await kafka_publisher.stop()

        print("Started after stop:", kafka_publisher.started)

    print("===== END OF TEST =====")


if __name__ == "__main__":
    asyncio.run(main())
