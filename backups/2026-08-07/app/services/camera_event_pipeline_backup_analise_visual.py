import asyncio
import logging
from datetime import datetime, timezone

from app.adapters.cameras.factory import camera_adapter_factory
from app.core.config import settings
from app.domain.models.camera_event import RawCameraPackage
from app.messaging.kafka_producer import KafkaPublisher
from app.services.alarm_detection_payload import (
    montar_alarm_detection_message,
)
from app.services.event_hub import event_hub
from app.services.person_tracker import (
    DetectionBox,
    person_tracker,
)


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
        self._exit_task: asyncio.Task | None = None
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

        self._exit_task = asyncio.create_task(
            self._monitorar_saidas(),
            name="person-exit-monitor",
        )

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

        tarefas: list[asyncio.Task] = [
            *self._worker_tasks,
        ]

        if self._exit_task is not None:
            tarefas.append(self._exit_task)

        for tarefa in tarefas:
            tarefa.cancel()

        if tarefas:
            await asyncio.gather(
                *tarefas,
                return_exceptions=True,
            )

        if self.publisher.iniciado:
            await self.publisher.stop()

        self._worker_tasks.clear()
        self._exit_task = None
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

    async def _monitorar_saidas(self) -> None:
        """
        Verifica a cada segundo quais pessoas estão há mais
        de oito segundos sem receber uma nova detecção.
        """

        logger.info(
            "Monitor automático de saídas iniciado"
        )

        while True:
            try:
                await asyncio.sleep(1)

                saidas = (
                    await person_tracker.coletar_saidas()
                )

                for saida in saidas:
                    momento_saida = (
                        datetime.now(timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z")
                    )

                    evento_painel = {
                        "evento_id": saida.evento_id,
                        "pessoa_id": saida.pessoa_id,
                        "status": "exited",
                        "quantidade_deteccoes": (
                            saida.quantidade_deteccoes
                        ),
                        "camera": saida.camera,
                        "tipo": None,
                        "datetime": momento_saida,
                        "imagem": None,
                    }

                    await event_hub.publicar(
                        evento_painel
                    )

                    logger.info(
                        "[RASTREAMENTO] Pessoa saiu: "
                        "pessoa=%s camera=%s "
                        "deteccoes=%s paineis=%s",
                        saida.pessoa_id,
                        saida.camera,
                        saida.quantidade_deteccoes,
                        event_hub.total_conexoes,
                    )

            except asyncio.CancelledError:
                logger.info(
                    "Monitor automático de saídas encerrado"
                )
                raise

            except Exception:
                logger.exception(
                    "Erro no monitor automático de saídas"
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

                if evento.imagem is None:
                    logger.info(
                        "[%s] Evento ignorado: "
                        "não possui imagem",
                        pacote.evento_id,
                    )
                    continue

                bounding_box = (
                    evento.bounding_box_escolhida
                )

                if bounding_box is None:
                    logger.info(
                        "[%s] Evento ignorado: "
                        "não possui bounding box",
                        pacote.evento_id,
                    )
                    continue

                camera = (
                    evento.nome_camera
                    or evento.camera_id
                    or evento.ip_camera
                    or "Camera desconhecida"
                )

                caixa_rastreamento = DetectionBox(
                    x=bounding_box.x,
                    y=bounding_box.y,
                    largura=bounding_box.largura,
                    altura=bounding_box.altura,
                )

                decisao = await person_tracker.registrar(
                    camera=camera,
                    evento_id=evento.evento_id,
                    bbox=caixa_rastreamento,
                )

                logger.info(
                    "[%s] Rastreamento: "
                    "pessoa=%s status=%s "
                    "deteccoes=%s",
                    pacote.evento_id,
                    decisao.pessoa_id,
                    decisao.status,
                    decisao.quantidade_deteccoes,
                )

                # Eventos repetidos em menos de cinco segundos
                # não geram Base64, painel ou Kafka.
                if not decisao.deve_processar:
                    logger.info(
                        "[%s] Evento repetido suprimido: "
                        "pessoa=%s",
                        pacote.evento_id,
                        decisao.pessoa_id,
                    )
                    continue

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

                evento_painel = {
                    "evento_id": evento.evento_id,
                    "pessoa_id": decisao.pessoa_id,
                    "status": decisao.status,
                    "quantidade_deteccoes": (
                        decisao.quantidade_deteccoes
                    ),
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

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception(
                    "[%s] Erro no processamento",
                    pacote.evento_id,
                )

            finally:
                self.queue.task_done()