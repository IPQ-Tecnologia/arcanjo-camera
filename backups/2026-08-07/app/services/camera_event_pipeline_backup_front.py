import asyncio
import logging

from app.adapters.cameras.factory import camera_adapter_factory
from app.core.config import settings
from app.domain.models.camera_event import RawCameraPackage
from app.messaging.kafka_producer import KafkaPublisher
from app.services.alarm_detection_payload import (
    montar_alarm_detection_message,
)
from app.services.event_hub import event_hub


logger = logging.getLogger(__name__)


class CameraEventPipeline:
    def __init__(
        self,
        publisher: KafkaPublisher,
        topic: str,
        maxsize: int = 1000,
        worker_count: int = 4,
    ) -> None:
        self.publisher = publisher
        self.topic = topic
        self.worker_count = worker_count

        self.queue: asyncio.Queue[
            tuple[RawCameraPackage, bytes]
        ] = asyncio.Queue(maxsize=maxsize)

        self._worker_tasks: list[asyncio.Task] = []
        self._iniciado = False

    @property
    def tamanho_fila(self) -> int:
        return self.queue.qsize()

    @property
    def capacidade_fila(self) -> int:
        return self.queue.maxsize

    async def start(self) -> None:
        if self._iniciado:
            return

        if settings.kafka_enabled:
            await self.publisher.start()
        else:
            logger.warning(
                "Kafka desativado: eventos serão processados, "
                "mas não publicados."
            )

        self._worker_tasks = [
            asyncio.create_task(
                self._worker(numero),
                name=f"camera-worker-{numero}",
            )
            for numero in range(
                1,
                self.worker_count + 1,
            )
        ]

        self._iniciado = True

        logger.info(
            "Pipeline iniciado com %s workers",
            self.worker_count,
        )

    async def stop(self) -> None:
        if not self._iniciado:
            return

        try:
            await asyncio.wait_for(
                self.queue.join(),
                timeout=10,
            )

        except TimeoutError:
            logger.warning(
                "Encerrando com %s itens na fila",
                self.queue.qsize(),
            )

        for task in self._worker_tasks:
            task.cancel()

        await asyncio.gather(
            *self._worker_tasks,
            return_exceptions=True,
        )

        if self.publisher.iniciado:
            await self.publisher.stop()

        self._worker_tasks.clear()
        self._iniciado = False

        logger.info("Pipeline encerrado")

    def adicionar(
        self,
        pacote: RawCameraPackage,
        body: bytes,
    ) -> None:
        self.queue.put_nowait(
            (pacote, body)
        )

    async def _worker(
        self,
        numero: int,
    ) -> None:
        logger.info(
            "Worker %s iniciado",
            numero,
        )

        while True:
            pacote, body = await self.queue.get()

            try:
                adapter = (
                    camera_adapter_factory.encontrar_adapter(
                        content_type=pacote.content_type,
                        body=body,
                    )
                )

                logger.info(
                    "[%s] Worker %s usando %s",
                    pacote.evento_id,
                    numero,
                    adapter.__class__.__name__,
                )

                evento = await asyncio.to_thread(
                    adapter.normalizar,
                    pacote,
                    body,
                )

                try:
                    mensagem = await asyncio.to_thread(
                        montar_alarm_detection_message,
                        evento,
                    )

                except ValueError as erro:
                    logger.info(
                        "[%s] Evento ignorado: %s",
                        pacote.evento_id,
                        erro,
                    )
                    continue

                # Dados simplificados enviados para o painel.
                evento_painel = {
                    "evento_id": evento.evento_id,
                    "camera": mensagem.device.name,
                    "tipo": mensagem.event.type,
                    "datetime": mensagem.event.datetime,
                    "imagem": (
                        mensagem.event.images.detection
                    ),
                }

                await event_hub.publicar(
                    evento_painel
                )

                logger.info(
                    "[%s] Evento enviado para "
                    "%s painel(is)",
                    pacote.evento_id,
                    event_hub.total_conexoes,
                )

                # O painel funciona mesmo com o Kafka desligado.
                if not settings.kafka_enabled:
                    logger.info(
                        "[%s] Payload pronto; "
                        "Kafka desativado",
                        pacote.evento_id,
                    )
                    continue

                resultado = (
                    await self.publisher.publicar(
                        topico=self.topic,
                        evento_id=evento.evento_id,
                        dados=mensagem.model_dump(
                            mode="json"
                        ),
                    )
                )

                logger.info(
                    "[%s] Publicado em %s "
                    "p=%s offset=%s",
                    pacote.evento_id,
                    resultado["topico"],
                    resultado["particao"],
                    resultado["offset"],
                )

            except Exception:
                logger.exception(
                    "[%s] Erro no processamento",
                    pacote.evento_id,
                )

            finally:
                self.queue.task_done()