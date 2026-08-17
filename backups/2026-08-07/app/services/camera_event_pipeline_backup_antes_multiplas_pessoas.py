import asyncio
import logging
from datetime import datetime, timezone

from app.adapters.cameras.factory import (
    camera_adapter_factory,
)
from app.core.config import settings
from app.domain.models.camera_event import (
    RawCameraPackage,
)
from app.messaging.kafka_producer import (
    KafkaPublisher,
)
from app.services.alarm_detection_payload import (
    montar_alarm_detection_message,
)
from app.services.appearance_memory import (
    appearance_memory,
)
from app.services.event_hub import event_hub
from app.services.person_tracker import (
    DetectionBox,
    person_tracker,
)
from app.services.scene_analyzer import (
    analisar_pessoa,
)
from app.services.scene_context_analyzer import (
    analisar_contexto_cena,
)
from app.services.scene_renderer import (
    renderizar_cena_com_boxes,
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
                "Kafka desativado: eventos serão "
                "processados, mas não publicados."
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
            tarefas.append(
                self._exit_task
            )

        for tarefa in tarefas:
            tarefa.cancel()

        if tarefas:
            await asyncio.gather(
                *tarefas,
                return_exceptions=True,
            )

        if self.publisher.iniciado:
            await self.publisher.stop()

        await appearance_memory.limpar()

        self._worker_tasks.clear()
        self._exit_task = None
        self._iniciado = False

        logger.info(
            "Pipeline encerrado"
        )

    def adicionar(
        self,
        pacote: RawCameraPackage,
        body: bytes,
    ) -> None:
        self.queue.put_nowait(
            (pacote, body)
        )

    async def _monitorar_saidas(
        self,
    ) -> None:
        logger.info(
            "Monitor automático de saídas iniciado"
        )

        while True:
            try:
                await asyncio.sleep(1)

                saidas = (
                    await person_tracker
                    .coletar_saidas()
                )

                for saida in saidas:
                    aparencia_final = (
                        await appearance_memory
                        .finalizar(
                            saida.pessoa_id
                        )
                    )

                    momento_saida = (
                        datetime.now(
                            timezone.utc
                        )
                        .isoformat()
                        .replace(
                            "+00:00",
                            "Z",
                        )
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
                        "aparencia": (
                            aparencia_final.to_dict()
                            if aparencia_final is not None
                            else None
                        ),
                        "contexto_cena": None,

                        # O painel manterá a última
                        # cena recebida no cartão.
                        "imagem_cena": None,
                        "quantidade_boxes_cena": None,
                    }

                    await event_hub.publicar(
                        evento_painel
                    )

                    logger.info(
                        "[RASTREAMENTO] Pessoa saiu: "
                        "pessoa=%s camera=%s "
                        "deteccoes=%s amostras=%s "
                        "paineis=%s",
                        saida.pessoa_id,
                        saida.camera,
                        saida.quantidade_deteccoes,
                        (
                            aparencia_final
                            .quantidade_amostras
                            if aparencia_final is not None
                            else 0
                        ),
                        event_hub.total_conexoes,
                    )

            except asyncio.CancelledError:
                logger.info(
                    "Monitor automático de "
                    "saídas encerrado"
                )
                raise

            except Exception:
                logger.exception(
                    "Erro no monitor automático "
                    "de saídas"
                )

    async def _analisar_aparencia(
        self,
        evento,
        bounding_box,
        pessoa_id: str,
        pacote_id: str,
    ):
        try:
            if evento.imagem is None:
                return None

            caminho_original = (
                evento.imagem.caminho_original
            )

            if not caminho_original:
                logger.info(
                    "[%s] Imagem sem caminho "
                    "original; aparência não analisada",
                    pacote_id,
                )
                return None

            analise_visual = await asyncio.to_thread(
                analisar_pessoa,
                caminho_imagem=caminho_original,
                x=bounding_box.x,
                y=bounding_box.y,
                largura=bounding_box.largura,
                altura=bounding_box.altura,
            )

            aparencia_estavel = (
                await appearance_memory.registrar(
                    pessoa_id=pessoa_id,
                    analise=analise_visual,
                )
            )

            logger.info(
                "[%s] Aparência estável: "
                "pessoa=%s cor=%s posição=%s "
                "tamanho=%s área_média=%s%% "
                "amostras=%s",
                pacote_id,
                pessoa_id,
                (
                    aparencia_estavel
                    .cor_roupa_predominante
                ),
                aparencia_estavel.posicao_atual,
                (
                    aparencia_estavel
                    .tamanho_predominante
                ),
                (
                    aparencia_estavel
                    .percentual_medio_quadro
                ),
                (
                    aparencia_estavel
                    .quantidade_amostras
                ),
            )

            return aparencia_estavel

        except Exception:
            logger.exception(
                "[%s] Não foi possível analisar "
                "ou estabilizar a aparência",
                pacote_id,
            )

            return None

    async def _analisar_contexto_cena(
        self,
        evento,
        pacote_id: str,
    ):
        try:
            if evento.imagem is None:
                return None

            largura_imagem = (
                evento.imagem.largura
            )

            altura_imagem = (
                evento.imagem.altura
            )

            if (
                not largura_imagem
                or not altura_imagem
            ):
                logger.info(
                    "[%s] Contexto não analisado: "
                    "dimensões da imagem ausentes",
                    pacote_id,
                )
                return None

            bounding_boxes = (
                evento.bounding_boxes
                or []
            )

            contexto = await asyncio.to_thread(
                analisar_contexto_cena,
                largura_imagem=largura_imagem,
                altura_imagem=altura_imagem,
                bounding_boxes=bounding_boxes,
            )

            logger.info(
                "[%s] Contexto da cena: "
                "pessoas=%s esquerda=%s "
                "centro=%s direita=%s "
                "muito_proximos=%s "
                "proximos=%s separados=%s",
                pacote_id,
                contexto.quantidade_pessoas,
                contexto.pessoas_esquerda,
                contexto.pessoas_centro,
                contexto.pessoas_direita,
                contexto.pares_muito_proximos,
                contexto.pares_proximos,
                contexto.pares_separados,
            )

            logger.info(
                "[%s] Descrição da cena: %s",
                pacote_id,
                contexto.descricao,
            )

            return contexto

        except Exception:
            logger.exception(
                "[%s] Não foi possível analisar "
                "o contexto da cena",
                pacote_id,
            )

            return None

    async def _renderizar_cena(
        self,
        evento,
        pacote_id: str,
    ):
        """
        Desenha as bounding boxes reais da câmera
        sobre a imagem original.

        A imagem resultante é enviada somente ao
        painel e não altera o payload do Kafka.
        """

        try:
            if evento.imagem is None:
                return None

            caminho_original = (
                evento.imagem.caminho_original
            )

            if not caminho_original:
                logger.info(
                    "[%s] Cena não renderizada: "
                    "caminho original ausente",
                    pacote_id,
                )
                return None

            bounding_boxes = (
                evento.bounding_boxes
                or []
            )

            if not bounding_boxes:
                logger.info(
                    "[%s] Cena não renderizada: "
                    "nenhuma bounding box recebida",
                    pacote_id,
                )
                return None

            resultado = await asyncio.to_thread(
                renderizar_cena_com_boxes,
                caminho_imagem=caminho_original,
                bounding_boxes=bounding_boxes,
            )

            if resultado.quantidade_boxes == 0:
                logger.info(
                    "[%s] Cena não renderizada: "
                    "nenhuma bounding box válida",
                    pacote_id,
                )
                return None

            logger.info(
                "[%s] Cena renderizada: "
                "boxes=%s dimensoes=%sx%s "
                "base64=%s caracteres",
                pacote_id,
                resultado.quantidade_boxes,
                resultado.largura_imagem,
                resultado.altura_imagem,
                len(resultado.imagem_base64),
            )

            return resultado

        except Exception:
            logger.exception(
                "[%s] Não foi possível renderizar "
                "a cena com bounding boxes",
                pacote_id,
            )

            return None

    async def _worker(
        self,
        numero: int,
    ) -> None:
        logger.info(
            "Worker %s iniciado",
            numero,
        )

        while True:
            pacote, body = (
                await self.queue.get()
            )

            try:
                adapter = (
                    camera_adapter_factory
                    .encontrar_adapter(
                        content_type=(
                            pacote.content_type
                        ),
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

                aparencia_estavel = (
                    await self._analisar_aparencia(
                        evento=evento,
                        bounding_box=bounding_box,
                        pessoa_id=decisao.pessoa_id,
                        pacote_id=pacote.evento_id,
                    )
                )

                contexto_cena = (
                    await self._analisar_contexto_cena(
                        evento=evento,
                        pacote_id=pacote.evento_id,
                    )
                )

                cena_renderizada = (
                    await self._renderizar_cena(
                        evento=evento,
                        pacote_id=pacote.evento_id,
                    )
                )

                # Evento repetido:
                # atualiza o cartão, mas não vai
                # novamente para o Kafka.
                if not decisao.deve_processar:
                    atualizacao_parcial = {
                        "evento_id": evento.evento_id,
                        "pessoa_id": decisao.pessoa_id,
                        "status": "appearance_updated",
                        "quantidade_deteccoes": (
                            decisao.quantidade_deteccoes
                        ),
                        "camera": camera,
                        "aparencia": (
                            aparencia_estavel.to_dict()
                            if aparencia_estavel is not None
                            else None
                        ),
                        "contexto_cena": (
                            contexto_cena.to_dict()
                            if contexto_cena is not None
                            else None
                        ),
                        "imagem_cena": (
                            cena_renderizada.imagem_base64
                            if cena_renderizada is not None
                            else None
                        ),
                        "quantidade_boxes_cena": (
                            cena_renderizada.quantidade_boxes
                            if cena_renderizada is not None
                            else 0
                        ),
                    }

                    await event_hub.publicar(
                        atualizacao_parcial
                    )

                    logger.info(
                        "[%s] Aparência, contexto e "
                        "cena atualizados em "
                        "%s painel(is): pessoa=%s "
                        "deteccoes=%s amostras=%s "
                        "boxes=%s",
                        pacote.evento_id,
                        event_hub.total_conexoes,
                        decisao.pessoa_id,
                        decisao.quantidade_deteccoes,
                        (
                            aparencia_estavel
                            .quantidade_amostras
                            if aparencia_estavel is not None
                            else 0
                        ),
                        (
                            cena_renderizada
                            .quantidade_boxes
                            if cena_renderizada is not None
                            else 0
                        ),
                    )

                    logger.info(
                        "[%s] Evento repetido "
                        "suprimido: pessoa=%s "
                        "amostras=%s",
                        pacote.evento_id,
                        decisao.pessoa_id,
                        (
                            aparencia_estavel
                            .quantidade_amostras
                            if aparencia_estavel is not None
                            else 0
                        ),
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

                    # Recorte individual da pessoa.
                    "imagem": (
                        mensagem
                        .event
                        .images
                        .detection
                    ),

                    "aparencia": (
                        aparencia_estavel.to_dict()
                        if aparencia_estavel is not None
                        else None
                    ),

                    "contexto_cena": (
                        contexto_cena.to_dict()
                        if contexto_cena is not None
                        else None
                    ),

                    # Imagem completa com todas as
                    # bounding boxes válidas.
                    "imagem_cena": (
                        cena_renderizada.imagem_base64
                        if cena_renderizada is not None
                        else None
                    ),

                    "quantidade_boxes_cena": (
                        cena_renderizada.quantidade_boxes
                        if cena_renderizada is not None
                        else 0
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

                # O formato do Kafka permanece igual.
                # imagem_cena e contexto_cena ficam
                # apenas no painel/WebSocket.
                resultado = await self.publisher.publicar(
                    topico=self.topic,
                    evento_id=evento.evento_id,
                    dados=mensagem.model_dump(
                        mode="json"
                    ),
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