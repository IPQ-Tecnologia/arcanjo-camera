"""
Processamento por pessoa detectada: aparência, movimento, recorte
facial, montagem dos payloads de alerta/face e publicação no painel e
no Kafka. Extraído de CameraEventPipeline.
"""

import asyncio
import logging

from app.core.config import settings
from app.domain.models.camera_event import BoundingBox, CameraEvent
from app.messaging.kafka_producer import KafkaPublisher
from app.services.alarm_detection_payload import (
    arquivo_para_base64,
    montar_alarm_detection_message,
    recortar_deteccao_base64,
)
from app.services.appearance_memory import appearance_memory
from app.services.event_hub import event_hub
from app.services.face_cropper import FaceCropResult, recortar_rosto
from app.services.person_movement import person_movement_memory
from app.services.person_tracker import TrackingDecision
from app.services.scene_analyzer import analisar_pessoa

logger = logging.getLogger(__name__)


class PersonProcessor:
    """
    Processa cada pessoa detectada em um evento normalizado: analisa
    aparência e movimento, gera recorte facial, monta os payloads de
    alerta e de captura facial, e publica tudo no painel e no Kafka.
    """

    def __init__(self, publisher: KafkaPublisher, topic: str) -> None:
        self.publisher = publisher
        self.topic = topic

    async def _analisar_aparencia(
        self,
        evento: CameraEvent,
        bounding_box: BoundingBox,
        pessoa_id: str,
        pacote_id: str,
    ):
        try:
            if evento.imagem is None or not evento.imagem.caminho_original:
                logger.info(
                    "[%s] Imagem sem caminho original; aparência não analisada",
                    pacote_id,
                )
                return None

            percentual_caixa = float(bounding_box.proporcao_imagem or 0.0) * 100

            # Uma pessoa muito pequena não possui pixels de roupa
            # suficientes para uma classificação confiável.
            if (
                bounding_box.largura < 60
                or bounding_box.altura < 120
                or percentual_caixa < 1.00
            ):
                logger.info(
                    "[%s] Aparência ignorada: caixa pequena pessoa=%s "
                    "dimensoes=%sx%s area=%.2f%%",
                    pacote_id,
                    pessoa_id,
                    bounding_box.largura,
                    bounding_box.altura,
                    percentual_caixa,
                )
                return await appearance_memory.obter(pessoa_id)

            analise_visual = await asyncio.to_thread(
                analisar_pessoa,
                caminho_imagem=evento.imagem.caminho_original,
                x=bounding_box.x,
                y=bounding_box.y,
                largura=bounding_box.largura,
                altura=bounding_box.altura,
            )
            # Uma leitura ambígua não entra na votação da memória.
            # Caso já exista uma aparência confiável, ela é preservada.
            if analise_visual.cor_roupa_aproximada == "escura-indefinida":
                logger.info(
                    "[%s] Cor escura indefinida: pessoa=%s rgb=%s; "
                    "mantendo aparência anterior",
                    pacote_id,
                    pessoa_id,
                    analise_visual.rgb_representativo,
                )
                return await appearance_memory.obter(pessoa_id)

            aparencia_estavel = await appearance_memory.registrar(
                pessoa_id=pessoa_id,
                analise=analise_visual,
            )
            logger.info(
                "[%s] Aparência estável: pessoa=%s cor=%s posição=%s "
                "tamanho=%s área_média=%s%% amostras=%s",
                pacote_id,
                pessoa_id,
                aparencia_estavel.cor_roupa_predominante,
                aparencia_estavel.posicao_atual,
                aparencia_estavel.tamanho_predominante,
                aparencia_estavel.percentual_medio_quadro,
                aparencia_estavel.quantidade_amostras,
            )
            return aparencia_estavel
        except Exception:
            logger.exception(
                "[%s] Não foi possível analisar a aparência da pessoa=%s",
                pacote_id,
                pessoa_id,
            )
            return None

    async def _obter_recorte_facial(
        self,
        evento: CameraEvent,
        bounding_box: BoundingBox,
        pessoa_id: str,
        pacote_id: str,
    ) -> FaceCropResult | None:
        if evento.imagem is None or not evento.imagem.caminho_original:
            return None

        if bounding_box.largura < 60 or bounding_box.altura < 100:
            logger.info(
                "[%s] Face ignorada: caixa da pessoa pequena pessoa=%s dimensoes=%sx%s",
                pacote_id,
                pessoa_id,
                bounding_box.largura,
                bounding_box.altura,
            )
            return None

        nome_seguro = "".join(
            caractere if (caractere.isalnum() or caractere in "-_") else "_"
            for caractere in pessoa_id
        )

        try:
            resultado = await asyncio.to_thread(
                recortar_rosto,
                caminho_imagem=evento.imagem.caminho_original,
                bounding_box=bounding_box,
                pasta_saida="recortes_faciais",
                nome_arquivo=f"{pacote_id}_{nome_seguro}_face.jpg",
            )

            if resultado is None:
                logger.info("[%s] Rosto não encontrado: pessoa=%s", pacote_id, pessoa_id)
                return None

            logger.info(
                "[%s] Recorte facial criado: pessoa=%s caixa=%s,%s %sx%s arquivo=%s",
                pacote_id,
                pessoa_id,
                resultado.x,
                resultado.y,
                resultado.largura,
                resultado.altura,
                resultado.caminho_arquivo,
            )
            return resultado

        except Exception:
            logger.exception(
                "[%s] Erro ao gerar recorte facial: pessoa=%s", pacote_id, pessoa_id
            )
            return None

    async def _publicar_atualizacao_parcial(
        self,
        camera: str,
        decisao: TrackingDecision,
        aparencia_estavel,
        movimento,
        contexto_cena,
        cena_renderizada,
        imagem_recorte_base64: str | None,
        indice_na_cena: int,
        total_pessoas_cena: int,
        pacote_id: str,
    ) -> None:
        atualizacao_parcial = {
            "evento_id": decisao.evento_id,
            "pessoa_id": decisao.pessoa_id,
            "status": "appearance_updated",
            "quantidade_deteccoes": decisao.quantidade_deteccoes,
            "camera": camera,
            "imagem": imagem_recorte_base64,
            "aparencia": aparencia_estavel.to_dict() if aparencia_estavel is not None else None,
            "movimento": movimento.to_dict() if movimento is not None else None,
            "contexto_cena": contexto_cena.to_dict() if contexto_cena is not None else None,
            "imagem_cena": (
                cena_renderizada.imagem_base64 if cena_renderizada is not None else None
            ),
            "quantidade_boxes_cena": (
                cena_renderizada.quantidade_boxes if cena_renderizada is not None else 0
            ),
            "indice_na_cena": indice_na_cena,
            "total_pessoas_cena": total_pessoas_cena,
        }
        await event_hub.publicar(atualizacao_parcial)
        logger.info(
            "[%s] Atualização parcial: pessoa=%s indice=%s/%s "
            "deteccoes=%s amostras=%s movimento=%s/%s velocidade=%spx_s "
            "boxes=%s paineis=%s",
            pacote_id,
            decisao.pessoa_id,
            indice_na_cena,
            total_pessoas_cena,
            decisao.quantidade_deteccoes,
            aparencia_estavel.quantidade_amostras if aparencia_estavel is not None else 0,
            movimento.movimento_horizontal if movimento is not None else "-",
            movimento.movimento_vertical if movimento is not None else "-",
            movimento.velocidade_pixels_segundo if movimento is not None else 0,
            cena_renderizada.quantidade_boxes if cena_renderizada is not None else 0,
            event_hub.total_conexoes,
        )

    async def processar_pessoa(
        self,
        evento: CameraEvent,
        camera: str,
        bounding_box: BoundingBox,
        decisao: TrackingDecision,
        contexto_cena,
        cena_renderizada,
        indice_na_cena: int,
        total_pessoas_cena: int,
        pacote_id: str,
        enviar_alerta_evento: bool,
        imagem_background_base64: str | None,
    ) -> str | None:
        aparencia_estavel = await self._analisar_aparencia(
            evento=evento,
            bounding_box=bounding_box,
            pessoa_id=decisao.pessoa_id,
            pacote_id=pacote_id,
        )

        movimento = await person_movement_memory.registrar(
            pessoa_id=decisao.pessoa_id,
            bbox=decisao.bbox,
        )
        logger.info(
            "[%s] Movimento: pessoa=%s horizontal=%s vertical=%s "
            "tendencia=%s velocidade=%spx_s distancia_total=%spx "
            "tempo=%ss amostras=%s",
            pacote_id,
            decisao.pessoa_id,
            movimento.movimento_horizontal,
            movimento.movimento_vertical,
            movimento.tendencia_distancia,
            movimento.velocidade_pixels_segundo,
            movimento.distancia_total_pixels,
            movimento.tempo_observado_segundos,
            movimento.quantidade_amostras,
        )

        if not decisao.deve_processar:
            imagem_recorte_base64 = None

            if evento.imagem is not None and evento.imagem.caminho_original:
                try:
                    imagem_recorte_base64 = await asyncio.to_thread(
                        recortar_deteccao_base64,
                        evento.imagem.caminho_original,
                        bounding_box,
                    )
                except Exception:
                    logger.exception(
                        "[%s] Não foi possível atualizar o recorte da pessoa=%s",
                        pacote_id,
                        decisao.pessoa_id,
                    )

            await self._publicar_atualizacao_parcial(
                camera=camera,
                decisao=decisao,
                aparencia_estavel=aparencia_estavel,
                movimento=movimento,
                contexto_cena=contexto_cena,
                cena_renderizada=cena_renderizada,
                imagem_recorte_base64=imagem_recorte_base64,
                indice_na_cena=indice_na_cena,
                total_pessoas_cena=total_pessoas_cena,
                pacote_id=pacote_id,
            )
            logger.info(
                "[%s] Evento repetido suprimido: pessoa=%s indice=%s/%s amostras=%s",
                pacote_id,
                decisao.pessoa_id,
                indice_na_cena,
                total_pessoas_cena,
                aparencia_estavel.quantidade_amostras if aparencia_estavel is not None else 0,
            )
            return imagem_background_base64

        if evento.imagem is None or not evento.imagem.caminho_original:
            logger.info("[%s] Pessoa ignorada: caminho original ausente", pacote_id)
            return imagem_background_base64

        resultado_face = await self._obter_recorte_facial(
            evento=evento,
            bounding_box=bounding_box,
            pessoa_id=decisao.pessoa_id,
            pacote_id=pacote_id,
        )
        imagem_rosto_base64 = (
            resultado_face.imagem_base64 if resultado_face is not None else None
        )

        if imagem_background_base64 is None:
            imagem_background_base64 = await asyncio.to_thread(
                arquivo_para_base64,
                evento.imagem.caminho_original,
            )

        try:
            mensagem = await asyncio.to_thread(
                montar_alarm_detection_message,
                evento,
                bounding_box=bounding_box,
                evento_id=decisao.evento_id,
                imagem_background_base64=imagem_background_base64,
            )
        except ValueError as erro:
            logger.info(
                "[%s] Pessoa ignorada: pessoa=%s erro=%s",
                pacote_id,
                decisao.pessoa_id,
                erro,
            )
            return imagem_background_base64

        payload_face_kafka = None

        evento_normalizado = evento.model_dump(mode="json", by_alias=True, exclude_none=True)
        atributos_normalizados = evento_normalizado.get("attributes") or {}

        vendor_event_type = (
            atributos_normalizados.get("vendor_event_type")
            or evento_normalizado.get("event_type")
            or evento.tipo_evento
        )

        # O supervisor solicitou event.id numérico.
        # Prioriza o ID original enviado pela câmera.
        source_event_id = atributos_normalizados.get("source_event_id")

        try:
            event_id_numerico = int(source_event_id)
        except (TypeError, ValueError):
            texto_evento_id = str(decisao.evento_id)

            # IDs multi-pessoa podem terminar em sufixos como "-01",
            # "-02", "-03". O ID base do evento continua sendo a
            # parte hexadecimal anterior.
            evento_id_base = texto_evento_id.split("-", 1)[0]

            try:
                event_id_numerico = int(evento_id_base, 16)
            except (TypeError, ValueError):
                event_id_numerico = int.from_bytes(
                    texto_evento_id.encode("utf-8")[:8].ljust(8, b"\0"),
                    byteorder="big",
                )

        # ==================================================
        # EVENTO FACIAL
        # ==================================================

        if resultado_face is not None:
            payload_face_kafka = {
                "camera_name": mensagem.device.name,
                "images": {"face": {"base64": resultado_face.imagem_base64}},
            }

            logger.info(
                "[%s] Payload facial simples pronto: "
                "topico=nelore-face-capture camera=%s qualidade=%.4f bbox=%s,%s-%s,%s",
                pacote_id,
                mensagem.device.name,
                resultado_face.score,
                resultado_face.x,
                resultado_face.y,
                resultado_face.x + resultado_face.largura,
                resultado_face.y + resultado_face.altura,
            )

        # ==================================================
        # EVENTO DE ALERTA
        # ==================================================

        payload_alerta_kafka = {
            "schema_version": evento.schema_version or "1.0",
            "device": {
                "name": mensagem.device.name,
                "brand": evento.fabricante,
                "ip": evento.ip_camera,
                "external_id": evento.camera_id,
                "serial_number": None,
                "latitude": None,
                "longitude": None,
            },
            "event": {
                "id": event_id_numerico,
                "name": vendor_event_type,
                "type": evento.tipo_evento,
                "timestamp": int(mensagem.event.timestamp),
                "datetime": mensagem.event.datetime,
            },
            "image": {
                "type": "detection",
                "width": evento.imagem.largura if evento.imagem is not None else 0,
                "height": evento.imagem.altura if evento.imagem is not None else 0,
                "format": (
                    evento.imagem.formato
                    if evento.imagem is not None and evento.imagem.formato
                    else "jpeg"
                ),
                "original_image_content": mensagem.event.images.background,
                "annotated_image_content": (
                    cena_renderizada.imagem_base64 if cena_renderizada is not None else ""
                ),
                "original_image_path": None,
                "annotated_image_path": None,
                "video_path": None,
            },
        }

        if enviar_alerta_evento:
            logger.info(
                "[%s] Payload de alerta pronto: topico=%s id=%s name=%s type=%s",
                pacote_id,
                self.topic,
                event_id_numerico,
                vendor_event_type,
                evento.tipo_evento,
            )
        else:
            logger.info(
                "[%s] Pessoa presente na cena sem vínculo com o evento da câmera: "
                "pessoa=%s; alerta não será publicado",
                pacote_id,
                decisao.pessoa_id,
            )

        evento_painel = {
            "evento_id": decisao.evento_id,
            "pessoa_id": decisao.pessoa_id,
            "status": decisao.status,
            "quantidade_deteccoes": decisao.quantidade_deteccoes,
            "camera": mensagem.device.name,
            "tipo": mensagem.event.type,
            "datetime": mensagem.event.datetime,
            "attributes": (
                mensagem.event.attributes.model_dump(mode="json")
                if mensagem.event.attributes is not None
                else None
            ),
            "imagem": mensagem.event.images.detection,
            "imagem_rosto": imagem_rosto_base64,
            "aparencia": aparencia_estavel.to_dict() if aparencia_estavel is not None else None,
            "movimento": movimento.to_dict(),
            "contexto_cena": contexto_cena.to_dict() if contexto_cena is not None else None,
            "imagem_cena": (
                cena_renderizada.imagem_base64 if cena_renderizada is not None else None
            ),
            "quantidade_boxes_cena": (
                cena_renderizada.quantidade_boxes if cena_renderizada is not None else 0
            ),
            "indice_na_cena": indice_na_cena,
            "total_pessoas_cena": total_pessoas_cena,
        }
        await event_hub.publicar(evento_painel)
        logger.info(
            "[%s] Pessoa enviada ao painel: pessoa=%s status=%s indice=%s/%s paineis=%s",
            pacote_id,
            decisao.pessoa_id,
            decisao.status,
            indice_na_cena,
            total_pessoas_cena,
            event_hub.total_conexoes,
        )

        if not settings.kafka_enabled:
            logger.info(
                "[%s] Payload pronto; Kafka desativado: pessoa=%s",
                pacote_id,
                decisao.pessoa_id,
            )
            return imagem_background_base64

        publicacoes_kafka = []

        if enviar_alerta_evento:
            publicacoes_kafka.append(
                {
                    "rotulo": "Alerta",
                    "topico": self.topic,
                    "corrotina": self.publisher.publicar(
                        topico=self.topic,
                        evento_id=decisao.evento_id,
                        dados=payload_alerta_kafka,
                    ),
                }
            )

        if payload_face_kafka is not None:
            publicacoes_kafka.append(
                {
                    "rotulo": "Face",
                    "topico": "nelore-face-capture",
                    "corrotina": self.publisher.publicar(
                        topico="nelore-face-capture",
                        evento_id=decisao.evento_id,
                        dados=payload_face_kafka,
                    ),
                }
            )

        resultados_kafka = await asyncio.gather(
            *[item["corrotina"] for item in publicacoes_kafka],
            return_exceptions=True,
        )

        for item, resultado_kafka in zip(publicacoes_kafka, resultados_kafka, strict=True):
            if isinstance(resultado_kafka, BaseException):
                logger.error(
                    "[%s] Erro ao publicar %s em %s: %r",
                    pacote_id,
                    item["rotulo"].lower(),
                    item["topico"],
                    resultado_kafka,
                )
                continue

            logger.info(
                "[%s] %s publicado em %s p=%s offset=%s pessoa=%s",
                pacote_id,
                item["rotulo"],
                resultado_kafka["topico"],
                resultado_kafka["particao"],
                resultado_kafka["offset"],
                decisao.pessoa_id,
            )

        return imagem_background_base64
