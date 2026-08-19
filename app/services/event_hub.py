import asyncio
import logging
from typing import Any

from fastapi import WebSocket


logger = logging.getLogger(__name__)


class EventHub:
    """Keeps connected browsers and sends events to them in real time."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    @property
    def total_connections(self) -> int:
        return len(self._connections)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()

        async with self._lock:
            self._connections.append(websocket)

        logger.info("Panel connected. Total: %s", self.total_connections)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self._connections:
                self._connections.remove(websocket)

        logger.info("Panel disconnected. Total: %s", self.total_connections)

    async def publish(self, event: dict[str, Any]) -> None:
        async with self._lock:
            connections = list(self._connections)

        disconnected: list[WebSocket] = []

        for websocket in connections:
            try:
                await websocket.send_json(event)
            except Exception:
                disconnected.append(websocket)

        for websocket in disconnected:
            await self.disconnect(websocket)


event_hub = EventHub()
