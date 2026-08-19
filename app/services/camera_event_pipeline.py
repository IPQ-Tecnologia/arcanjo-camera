import asyncio
import logging
from datetime import datetime, timezone

from app.adapters.cameras.factory import camera_adapter_factory
from app.core.config import settings
from app.domain.models.camera_event import CameraEvent, RawCameraPackage
from app.messaging.kafka_producer import KafkaPublisher
from app.services.appearance_memory import appearance_memory
from app.services.person_movement import person_movement_memory
from app.services.person_tracker import DetectionBox, person_tracker
from app.services.pipeline import box_matching, scene
from app.services.pipeline.exit_monitor import monitorar_saidas
from app.services.pipeline.person_processing import PersonProcessor

logger = logging.getLogger(__name__)


class CameraEventPipeline:
    """
    Orquestra o processamento assíncrono dos eventos de câmera: recebe
    pacotes brutos na fila, normaliza via adapter do fabricante, valida
    as pessoas com YOLO, rastreia entre frames e delega a cada pessoa
    detectada para o PersonProcessor (aparência/movimento/face/Kafka).
    """

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
        self.queue: asyncio.Queue[tuple[RawCameraPackage, bytes]] = asyncio.Queue(
            maxsize=maxsize
        )
        self._person_processor = PersonProcessor(publisher=publisher, topic=topic)
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
                "Kafka desativado: eventos serão processados, mas não publicados."
            )

        self._worker_tasks = [
            asyncio.create_task(self._worker(numero), name=f"camera-worker-{numero}")
            for numero in range(1, self.worker_count + 1)
        ]
        self._exit_task = asyncio.create_task(monitorar_saidas(), name="person-exit-monitor")
        self._iniciado = True
        logger.info("Pipeline iniciado com %s workers", self.worker_count)

    async def stop(self) -> None:
        if not self._iniciado:
            return

        try:
            await asyncio.wait_for(self.queue.join(), timeout=10)
        except TimeoutError:
            logger.warning("Encerrando com %s itens na fila", self.queue.qsize())

        tarefas: list[asyncio.Task] = [*self._worker_tasks]
        if self._exit_task is not None:
            tarefas.append(self._exit_task)

        for tarefa in tarefas:
            tarefa.cancel()

        if tarefas:
            await asyncio.gather(*tarefas, return_exceptions=True)

        if self.publisher.iniciado:
            await self.publisher.stop()

        await appearance_memory.limpar()
        await person_movement_memory.limpar()
        await person_tracker.limpar()

        self._worker_tasks.clear()
        self._exit_task = None
        self._iniciado = False
        logger.info("Pipeline encerrado")

    def adicionar(self, pacote: RawCameraPackage, body: bytes) -> None:
        self.queue.put_nowait((pacote, body))

    @staticmethod
    def _logar_horario_camera(evento_id: str, evento: CameraEvent) -> None:
        """Compara o horário informado pela câmera com o horário do servidor."""
        hora_camera = evento.data_hora
        if hora_camera.tzinfo is None:
            hora_camera = hora_camera.replace(tzinfo=timezone.utc)

        hora_camera_utc = hora_camera.astimezone(timezone.utc)
        hora_camera_local = hora_camera.astimezone()
        hora_servidor_utc = datetime.now(timezone.utc)
        hora_servidor_local = hora_servidor_utc.astimezone()
        diferenca_ms = (hora_servidor_utc - hora_camera_utc).total_seconds() * 1000

        logger.info(
            "[%s] HORÁRIO DA CÂMERA: fabricante=%s camera=%s enviado_utc=%s "
            "enviado_local=%s recebido_local=%s diferenca_ms=%.0f",
            evento_id,
            evento.fabricante,
            evento.nome_camera or evento.camera_id or "desconhecida",
            hora_camera_utc.isoformat(),
            hora_camera_local.isoformat(),
            hora_servidor_local.isoformat(),
            diferenca_ms,
        )

    async def _worker(self, numero: int) -> None:
        logger.info("Worker %s iniciado", numero)

        while True:
            pacote, body = await self.queue.get()

            try:
                adapter = camera_adapter_factory.encontrar_adapter(
                    content_type=pacote.content_type,
                    body=body,
                )
                logger.info(
                    "[%s] Worker %s usando %s",
                    pacote.evento_id,
                    numero,
                    adapter.__class__.__name__,
                )

                evento = await asyncio.to_thread(adapter.normalizar, pacote, body)

                logger.info(
                    "[%s] OBJETO NORMALIZADO (%s):\n%s",
                    pacote.evento_id,
                    evento.fabricante,
                    evento.model_dump_json(indent=2, by_alias=True, exclude_none=True),
                )

                self._logar_horario_camera(pacote.evento_id, evento)

                if evento.imagem is None:
                    logger.info("[%s] Evento ignorado: não possui imagem", pacote.evento_id)
                    continue

                bounding_boxes_camera = box_matching.selecionar_boxes_pessoas(evento)
                if not bounding_boxes_camera:
                    logger.info(
                        "[%s] Evento ignorado: não possui boxes válidas fornecidas pela câmera",
                        pacote.evento_id,
                    )
                    continue

                bounding_boxes, indices_alerta_evento = await box_matching.validar_com_yolo(
                    evento=evento,
                    bounding_boxes_camera=bounding_boxes_camera,
                    pacote_id=pacote.evento_id,
                )

                if not bounding_boxes:
                    logger.info(
                        "[%s] Evento ignorado: nenhuma caixa foi confirmada "
                        "como pessoa pelo YOLO",
                        pacote.evento_id,
                    )
                    continue

                camera = (
                    evento.nome_camera
                    or evento.camera_id
                    or evento.ip_camera
                    or "Camera desconhecida"
                )
                caixas_rastreamento = [
                    DetectionBox(
                        x=caixa.x, y=caixa.y, largura=caixa.largura, altura=caixa.altura
                    )
                    for caixa in bounding_boxes
                ]
                decisoes = await person_tracker.registrar_lote(
                    camera=camera,
                    evento_id=evento.evento_id,
                    bboxes=caixas_rastreamento,
                )

                if len(decisoes) != len(bounding_boxes):
                    raise RuntimeError(
                        "Quantidade de decisões diferente da quantidade de boxes"
                    )

                total_pessoas = len(bounding_boxes)
                logger.info(
                    "[%s] Rastreamento em lote: pessoas=%s ids=%s",
                    pacote.evento_id,
                    total_pessoas,
                    [(decisao.pessoa_id, decisao.status) for decisao in decisoes],
                )

                contexto_cena = await scene.analisar_contexto_cena(
                    evento=evento,
                    bounding_boxes=bounding_boxes,
                    pacote_id=pacote.evento_id,
                )
                cena_renderizada = await scene.renderizar_cena(
                    evento=evento,
                    bounding_boxes=bounding_boxes,
                    pacote_id=pacote.evento_id,
                )

                imagem_background_base64: str | None = None
                for indice, (bounding_box, decisao) in enumerate(
                    zip(bounding_boxes, decisoes, strict=True),
                    start=1,
                ):
                    imagem_background_base64 = await self._person_processor.processar_pessoa(
                        evento=evento,
                        camera=camera,
                        bounding_box=bounding_box,
                        decisao=decisao,
                        contexto_cena=contexto_cena,
                        cena_renderizada=cena_renderizada,
                        indice_na_cena=indice,
                        total_pessoas_cena=total_pessoas,
                        pacote_id=pacote.evento_id,
                        enviar_alerta_evento=indice - 1 in indices_alerta_evento,
                        imagem_background_base64=imagem_background_base64,
                    )

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[%s] Erro no processamento", pacote.evento_id)
            finally:
                self.queue.task_done()
