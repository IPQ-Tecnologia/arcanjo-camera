import asyncio
import json
from pathlib import Path

from app.core.config import settings
from app.messaging.kafka_producer import kafka_publisher


ARQUIVO_PAYLOAD = Path("payload_alarm_teste.json")


async def main() -> None:
    if not ARQUIVO_PAYLOAD.exists():
        raise FileNotFoundError(
            "Arquivo payload_alarm_teste.json não encontrado"
        )

    dados = json.loads(
        ARQUIVO_PAYLOAD.read_text(encoding="utf-8")
    )

    evento_id = dados["event"]["id"]
    topico = settings.kafka_topic_normalized

    tamanho_mensagem = len(
        json.dumps(
            dados,
            ensure_ascii=False,
            separators=(",", ":")
        ).encode("utf-8")
    )

    print("===== PUBLICAÇÃO KAFKA =====")
    print("Tópico:", topico)
    print("Evento ID:", evento_id)
    print("Tamanho:", tamanho_mensagem, "bytes")

    try:
        await kafka_publisher.start()

        resultado = await kafka_publisher.publicar(
            topico=topico,
            evento_id=evento_id,
            dados=dados
        )

        print("Mensagem publicada com sucesso.")
        print("Metadados:", resultado)

    finally:
        await kafka_publisher.stop()

    print("===== FIM DA PUBLICAÇÃO =====")


if __name__ == "__main__":
    asyncio.run(main())