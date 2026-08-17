import asyncio
import ssl

from aiokafka import AIOKafkaProducer
from aiokafka.helpers import create_ssl_context

from app.core.config import settings


def obter_servidores() -> list[str]:
    servidores = [
        servidor.strip()
        for servidor in settings.kafka_bootstrap_servers.split(",")
        if servidor.strip()
    ]

    if not servidores:
        raise ValueError(
            "KAFKA_BOOTSTRAP_SERVERS não foi configurado"
        )

    return servidores


def criar_configuracao_produtor() -> dict:
    protocolo = settings.kafka_security_protocol.upper()

    protocolos_validos = {
        "PLAINTEXT",
        "SSL",
        "SASL_PLAINTEXT",
        "SASL_SSL"
    }

    if protocolo not in protocolos_validos:
        raise ValueError(
            f"Protocolo Kafka inválido: {protocolo}"
        )

    configuracao = {
        "bootstrap_servers": obter_servidores(),
        "client_id": settings.kafka_client_id,
        "security_protocol": protocolo,
        "request_timeout_ms": settings.kafka_request_timeout_ms
    }

    if protocolo in {"SASL_PLAINTEXT", "SASL_SSL"}:
        if not settings.kafka_sasl_mechanism:
            raise ValueError(
                "KAFKA_SASL_MECHANISM não foi configurado"
            )

        if not settings.kafka_sasl_username:
            raise ValueError(
                "KAFKA_SASL_USERNAME não foi configurado"
            )

        if not settings.kafka_sasl_password:
            raise ValueError(
                "KAFKA_SASL_PASSWORD não foi configurado"
            )

        configuracao.update({
            "sasl_mechanism": settings.kafka_sasl_mechanism,
            "sasl_plain_username": settings.kafka_sasl_username,
            "sasl_plain_password": settings.kafka_sasl_password
        })

    if protocolo in {"SSL", "SASL_SSL"}:
        contexto_ssl: ssl.SSLContext

        if settings.kafka_ssl_cafile:
            contexto_ssl = create_ssl_context(
                cafile=settings.kafka_ssl_cafile
            )
        else:
            contexto_ssl = create_ssl_context()

        configuracao["ssl_context"] = contexto_ssl

    return configuracao


async def testar_conexao() -> None:
    configuracao = criar_configuracao_produtor()

    print("===== TESTE DE CONEXÃO KAFKA =====")
    print("Servidores:", configuracao["bootstrap_servers"])
    print("Protocolo:", configuracao["security_protocol"])
    print("Client ID:", configuracao["client_id"])

    produtor = AIOKafkaProducer(**configuracao)
    iniciado = False

    try:
        print("Tentando conectar...")

        await produtor.start()
        iniciado = True

        print("Conexão com o Kafka realizada com sucesso.")

    except Exception as erro:
        print("Não foi possível conectar ao Kafka.")
        print("Tipo do erro:", type(erro).__name__)
        print("Detalhes:", str(erro))

        raise

    finally:
        if iniciado:
            await produtor.stop()
            print("Conexão encerrada corretamente.")

        print("===== FIM DO TESTE =====")


if __name__ == "__main__":
    asyncio.run(testar_conexao())