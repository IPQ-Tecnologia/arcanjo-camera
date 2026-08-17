import json
import ssl
from typing import Any

from aiokafka import AIOKafkaProducer
from aiokafka.helpers import create_ssl_context

from app.core.config import settings


class KafkaPublisher:
    """
    Produtor Kafka reutilizável.

    A conexão é aberta uma única vez com start()
    e encerrada com stop().
    """

    def __init__(self) -> None:
        self._producer: AIOKafkaProducer | None = None
        self._iniciado: bool = False

    @property
    def iniciado(self) -> bool:
        """
        Informa se o produtor foi iniciado.
        """

        return self._iniciado

    @staticmethod
    def _obter_servidores() -> list[str]:
        servidores = [
            servidor.strip()
            for servidor in (
                settings.kafka_bootstrap_servers.split(",")
            )
            if servidor.strip()
        ]

        if not servidores:
            raise ValueError(
                "KAFKA_BOOTSTRAP_SERVERS não foi configurado"
            )

        return servidores

    @staticmethod
    def _criar_contexto_ssl() -> ssl.SSLContext:
        if settings.kafka_ssl_cafile:
            return create_ssl_context(
                cafile=settings.kafka_ssl_cafile
            )

        return create_ssl_context()

    def _criar_configuracao(self) -> dict[str, Any]:
        protocolo = (
            settings.kafka_security_protocol
            .strip()
            .upper()
        )

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

        configuracao: dict[str, Any] = {
            "bootstrap_servers": self._obter_servidores(),
            "client_id": (
                f"{settings.kafka_client_id}-melhorado"
            ),
            "security_protocol": protocolo,
            "request_timeout_ms": (
                settings.kafka_request_timeout_ms
            ),
            "acks": "all",
            "enable_idempotence": True,
            "retry_backoff_ms": 500
        }

        if protocolo in {
            "SASL_PLAINTEXT",
            "SASL_SSL"
        }:
            if not settings.kafka_sasl_mechanism:
                raise ValueError(
                    "KAFKA_SASL_MECHANISM não configurado"
                )

            if not settings.kafka_sasl_username:
                raise ValueError(
                    "KAFKA_SASL_USERNAME não configurado"
                )

            if not settings.kafka_sasl_password:
                raise ValueError(
                    "KAFKA_SASL_PASSWORD não configurado"
                )

            configuracao.update({
                "sasl_mechanism": (
                    settings.kafka_sasl_mechanism
                ),
                "sasl_plain_username": (
                    settings.kafka_sasl_username
                ),
                "sasl_plain_password": (
                    settings.kafka_sasl_password
                )
            })

        if protocolo in {"SSL", "SASL_SSL"}:
            configuracao["ssl_context"] = (
                self._criar_contexto_ssl()
            )

        return configuracao

    async def start(self) -> None:
        """
        Abre a conexão com o Kafka.

        Se já estiver iniciado, não cria outra conexão.
        """

        if self._iniciado:
            print(
                "[KafkaPublisher] Produtor já está iniciado"
            )
            return

        configuracao = self._criar_configuracao()

        print(
            "[KafkaPublisher] Conectando em:",
            configuracao["bootstrap_servers"]
        )

        self._producer = AIOKafkaProducer(
            **configuracao
        )

        try:
            await self._producer.start()

        except Exception:
            self._producer = None
            self._iniciado = False
            raise

        self._iniciado = True

        print(
            "[KafkaPublisher] Conexão realizada com sucesso"
        )

    async def stop(self) -> None:
        """
        Encerra a conexão com o Kafka.
        """

        if self._producer is None:
            self._iniciado = False

            print(
                "[KafkaPublisher] Produtor já está parado"
            )
            return

        try:
            await self._producer.stop()

        finally:
            self._producer = None
            self._iniciado = False

        print(
            "[KafkaPublisher] Conexão encerrada corretamente"
        )

    async def publicar(
        self,
        topico: str,
        evento_id: str,
        dados: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Publica um evento JSON no Kafka.

        Esta função será utilizada nas próximas etapas.
        Não execute ainda sem confirmar o tópico.
        """

        if not self._iniciado:
            raise RuntimeError(
                "O produtor Kafka não foi iniciado"
            )

        if self._producer is None:
            raise RuntimeError(
                "Instância do produtor indisponível"
            )

        if not topico.strip():
            raise ValueError(
                "O tópico Kafka não foi informado"
            )

        mensagem = json.dumps(
            dados,
            ensure_ascii=False,
            default=str,
            separators=(",", ":")
        ).encode("utf-8")

        chave = evento_id.encode("utf-8")

        metadata = await self._producer.send_and_wait(
            topic=topico,
            key=chave,
            value=mensagem
        )

        resultado = {
            "topico": metadata.topic,
            "particao": metadata.partition,
            "offset": metadata.offset,
            "timestamp": metadata.timestamp
        }

        print(
            f"[KafkaPublisher] Evento {evento_id} "
            f"publicado no tópico {metadata.topic}, "
            f"partição {metadata.partition}, "
            f"offset {metadata.offset}"
        )

        return resultado


kafka_publisher = KafkaPublisher()