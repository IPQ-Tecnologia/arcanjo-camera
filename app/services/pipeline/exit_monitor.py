"""
Monitor em background que detecta quando uma pessoa rastreada saiu de
cena e publica esse evento no painel. Extraído de CameraEventPipeline.
"""

import asyncio
import logging
from datetime import datetime, timezone

from app.services.appearance_memory import appearance_memory
from app.services.event_hub import event_hub
from app.services.person_movement import person_movement_memory
from app.services.person_tracker import person_tracker

logger = logging.getLogger(__name__)


async def monitorar_saidas() -> None:
    logger.info("Monitor automático de saídas iniciado")

    while True:
        try:
            await asyncio.sleep(1)
            saidas = await person_tracker.coletar_saidas()

            for saida in saidas:
                aparencia_final = await appearance_memory.finalizar(saida.pessoa_id)
                movimento_final = await person_movement_memory.finalizar(saida.pessoa_id)
                momento_saida = (
                    datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                )

                evento_painel = {
                    "evento_id": saida.evento_id,
                    "pessoa_id": saida.pessoa_id,
                    "status": "exited",
                    "quantidade_deteccoes": saida.quantidade_deteccoes,
                    "camera": saida.camera,
                    "tipo": None,
                    "datetime": momento_saida,
                    "attributes": None,
                    "imagem": None,
                    "aparencia": (
                        aparencia_final.to_dict() if aparencia_final is not None else None
                    ),
                    "movimento": (
                        movimento_final.to_dict() if movimento_final is not None else None
                    ),
                    "contexto_cena": None,
                    "imagem_cena": None,
                    "quantidade_boxes_cena": None,
                }
                await event_hub.publicar(evento_painel)
                logger.info(
                    "[RASTREAMENTO] Pessoa saiu: pessoa=%s camera=%s "
                    "deteccoes=%s amostras_visuais=%s amostras_movimento=%s "
                    "tempo_observado=%ss paineis=%s",
                    saida.pessoa_id,
                    saida.camera,
                    saida.quantidade_deteccoes,
                    (
                        aparencia_final.quantidade_amostras
                        if aparencia_final is not None
                        else 0
                    ),
                    (
                        movimento_final.quantidade_amostras
                        if movimento_final is not None
                        else 0
                    ),
                    (
                        movimento_final.tempo_observado_segundos
                        if movimento_final is not None
                        else 0
                    ),
                    event_hub.total_conexoes,
                )

        except asyncio.CancelledError:
            logger.info("Monitor automático de saídas encerrado")
            raise
        except Exception:
            logger.exception("Erro no monitor automático de saídas")
