import asyncio

from app.messaging.kafka_producer import kafka_publisher


async def main() -> None:
    print("===== TESTE DO KAFKA PUBLISHER =====")
    print("Iniciado antes:", kafka_publisher.iniciado)

    try:
        await kafka_publisher.start()

        print(
            "Iniciado depois do start:",
            kafka_publisher.iniciado
        )

        print("Produtor reutilizável funcionando.")

    finally:
        await kafka_publisher.stop()

        print(
            "Iniciado depois do stop:",
            kafka_publisher.iniciado
        )

    print("===== FIM DO TESTE =====")


if __name__ == "__main__":
    asyncio.run(main())