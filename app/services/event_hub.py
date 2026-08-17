import asyncio
import logging
from typing import Any

from fastapi import WebSocket


logger = logging.getLogger(__name__)


class EventHub:
    """
    Mantém os navegadores conectados e envia eventos em tempo real.
    """

    def __init__(self) -> None:
        self._conexoes: list[WebSocket] = []
        self._lock = asyncio.Lock()

    @property
    def total_conexoes(self) -> int:
        return len(self._conexoes)

    async def conectar(self, websocket: WebSocket) -> None:
        await websocket.accept()

        async with self._lock:
            self._conexoes.append(websocket)

        logger.info(
            "Painel conectado. Total: %s",
            self.total_conexoes,
        )

    async def desconectar(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self._conexoes:
                self._conexoes.remove(websocket)

        logger.info(
            "Painel desconectado. Total: %s",
            self.total_conexoes,
        )

    async def publicar(
        self,
        evento: dict[str, Any],
    ) -> None:
        async with self._lock:
            conexoes = list(self._conexoes)

        desconectadas: list[WebSocket] = []

        for websocket in conexoes:
            try:
                await websocket.send_json(evento)

            except Exception:
                desconectadas.append(websocket)

        for websocket in desconectadas:
            await self.desconectar(websocket)


event_hub = EventHub()