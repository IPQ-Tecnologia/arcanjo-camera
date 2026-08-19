import asyncio
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from app.core.config import settings
from app.domain.models.camera_event import RawCameraPackage
from app.messaging.kafka_producer import kafka_publisher
from app.services.camera_event_pipeline import CameraEventPipeline
from app.services.event_hub import event_hub


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


BASE_DIR = Path(__file__).resolve().parent

PANEL_HTML = BASE_DIR / "app" / "static" / "index.html"
DAHUA_CAPTURES_DIR = BASE_DIR / "dahua_captures"
HIKVISION_CAPTURES_DIR = BASE_DIR / "hikvision_captures"


pipeline = CameraEventPipeline(
    publisher=kafka_publisher,
    topic=settings.kafka_topic_normalized,
    maxsize=settings.ingestion_queue_size,
    worker_count=settings.ingestion_workers,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await pipeline.start()
    app.state.camera_pipeline = pipeline
    yield
    await pipeline.stop()


app = FastAPI(title="Camera API with Kafka", lifespan=lifespan)


def enqueue_package(
    request: Request,
    body: bytes,
    event_id: str,
    start: float,
    origin: str,
):
    """
    Puts the received package on the asynchronous queue.

    This function is used by both Hikvision and Dahua.
    """
    content_type = request.headers.get("content-type", "application/octet-stream")
    camera_ip = request.client.host if request.client else None

    package = RawCameraPackage(
        event_id=event_id,
        received_at=datetime.now(timezone.utc),
        content_type=content_type,
        camera_ip=camera_ip,
        package_path=f"memory://{event_id}",
    )

    try:
        request.app.state.camera_pipeline.add(package, body)
    except asyncio.QueueFull:
        logging.warning("[%s][%s] Queue full", origin, event_id)
        return JSONResponse(
            status_code=503,
            content={"status": "queue_full", "event_id": event_id, "origin": origin},
        )

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    logging.info(
        "[%s][%s] Received: bytes=%s queue=%s/%s response=%sms",
        origin,
        event_id,
        len(body),
        pipeline.queue_size,
        pipeline.queue_capacity,
        elapsed_ms,
    )

    return {
        "status": "received",
        "event_id": event_id,
        "origin": origin,
        "processing": "async_queue",
        "response_time_ms": elapsed_ms,
    }


def save_dahua_capture(
    event_id: str,
    moment: str,
    camera_ip: str,
    content_type: str,
    headers: list[str],
    body: bytes,
) -> None:
    """Saves Dahua's raw package after the response has already been sent to the camera."""
    DAHUA_CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    base_name = f"{moment}_{event_id}"
    package_path = DAHUA_CAPTURES_DIR / f"{base_name}_package.bin"
    headers_path = DAHUA_CAPTURES_DIR / f"{base_name}_headers.txt"

    package_path.write_bytes(body)

    info = [
        f"event_id: {event_id}",
        f"received_at_utc: {moment}",
        f"camera_ip: {camera_ip}",
        f"content_type: {content_type}",
        f"byte_count: {len(body)}",
        "",
        "===== HEADERS =====",
        *headers,
    ]

    headers_path.write_text("\n".join(info), encoding="utf-8")

    logging.info(
        "[DAHUA][%s] Capture saved: package=%s headers=%s bytes=%s first_bytes=%r",
        event_id,
        package_path,
        headers_path,
        len(body),
        body[:100],
    )


HIKVISION_ANALYSIS_TYPES = {"fielddetection", "linedetection"}

HIKVISION_CAPTURE_LIMIT = 10

HIKVISION_CAPTURE_TASKS: set[asyncio.Task] = set()

HIKVISION_EVENT_TYPE_PATTERN = re.compile(
    rb"<(?:[A-Za-z0-9_-]+:)?eventType>\s*([^<]+)",
    flags=re.IGNORECASE,
)


def save_hikvision_capture(
    event_id: str,
    moment: str,
    camera_ip: str,
    content_type: str,
    headers: list[str],
    body: bytes,
) -> None:
    """Saves raw samples of Hikvision events used for attribute discovery."""
    match = HIKVISION_EVENT_TYPE_PATTERN.search(body)
    if match is None:
        return

    event_type = match.group(1).decode("utf-8", errors="ignore").strip().lower()
    if event_type not in HIKVISION_ANALYSIS_TYPES:
        return

    has_image = b"\xff\xd8\xff" in body
    category = "with_image" if has_image else "without_image"
    event_folder = HIKVISION_CAPTURES_DIR / event_type / category
    event_folder.mkdir(parents=True, exist_ok=True)

    existing_captures = list(event_folder.glob("*_package.bin"))
    if len(existing_captures) >= HIKVISION_CAPTURE_LIMIT:
        return

    base_name = f"{moment}_{event_id}"
    package_path = event_folder / f"{base_name}_package.bin"
    headers_path = event_folder / f"{base_name}_headers.txt"

    package_path.write_bytes(body)

    info = [
        f"event_id: {event_id}",
        f"received_at_utc: {moment}",
        f"camera_ip: {camera_ip}",
        f"content_type: {content_type}",
        f"event_type: {event_type}",
        f"has_image: {has_image}",
        f"byte_count: {len(body)}",
        "",
        "===== HEADERS =====",
        *headers,
    ]

    headers_path.write_text("\n".join(info), encoding="utf-8")

    logging.info(
        "[HIKVISION][%s] Analysis capture saved: type=%s category=%s package=%s bytes=%s",
        event_id,
        event_type,
        category,
        package_path,
        len(body),
    )


def schedule_hikvision_capture(
    event_id: str,
    moment: str,
    camera_ip: str,
    content_type: str,
    headers: list[str],
    body: bytes,
) -> None:
    """Runs the write outside the response flow sent to the camera."""
    task = asyncio.create_task(
        asyncio.to_thread(
            save_hikvision_capture,
            event_id,
            moment,
            camera_ip,
            content_type,
            headers,
            body,
        )
    )
    HIKVISION_CAPTURE_TASKS.add(task)
    task.add_done_callback(HIKVISION_CAPTURE_TASKS.discard)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "kafka_enabled": settings.kafka_enabled,
        "kafka_connected": kafka_publisher.started,
        "kafka_topic": settings.kafka_topic_normalized,
        "queue": {
            "size": pipeline.queue_size,
            "capacity": pipeline.queue_capacity,
        },
        "workers": settings.ingestion_workers,
        "camera_routes": {
            "hikvision": "/hikvision",
            "dahua": "/dahua",
            "legacy_hikvision": "/",
            "legacy_dahua": "/CAM7442",
        },
    }


# Old route disabled. Now /hikvision is used.
@app.post("/hikvision")
async def receive_camera(request: Request):
    """Route currently used by Hikvision."""
    start = time.perf_counter()
    event_id = uuid.uuid4().hex[:12]
    body = await request.body()
    moment = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    content_type = request.headers.get("content-type", "application/octet-stream")
    camera_ip = request.client.host if request.client else "unknown"

    print('body', body)

    blocked_headers = {"authorization", "proxy-authorization", "cookie", "set-cookie"}
    safe_headers = [
        f"{name}: {value}"
        for name, value in request.headers.items()
        if name.lower() not in blocked_headers
    ]

    schedule_hikvision_capture(
        event_id=event_id,
        moment=moment,
        camera_ip=camera_ip,
        content_type=content_type,
        headers=safe_headers,
        body=body,
    )

    return enqueue_package(
        request=request,
        body=body,
        event_id=event_id,
        start=start,
        origin="HIKVISION",
    )


# Old routes disabled. Now /dahua is used.
# @app.post("/dahua")
@app.post("/CAM7442")
async def receive_dahua(request: Request, background_tasks: BackgroundTasks):
    """Receives the Dahua event, responds quickly, and saves the raw package in the background."""
    start = time.perf_counter()
    event_id = uuid.uuid4().hex[:12]
    moment = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    body = await request.body()
    content_type = request.headers.get("content-type", "application/octet-stream")
    camera_ip = request.client.host if request.client else "unknown"

    blocked_headers = {"authorization", "proxy-authorization", "cookie", "set-cookie"}
    safe_headers = []
    for name, value in request.headers.items():
        if name.lower() in blocked_headers:
            continue
        safe_headers.append(f"{name}: {value}")

    background_tasks.add_task(
        save_dahua_capture,
        event_id,
        moment,
        camera_ip,
        content_type,
        safe_headers,
        body,
    )

    logging.info(
        "[DAHUA][%s] Request received: ip=%s content_type=%s bytes=%s first_bytes=%r",
        event_id,
        camera_ip,
        content_type,
        len(body),
        body[:100],
    )

    return enqueue_package(
        request=request,
        body=body,
        event_id=event_id,
        start=start,
        origin="DAHUA",
    )


@app.post("/{device}")
async def receive_by_device(
    device: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Dynamic route for receiving events from different camera
    manufacturers.

    Examples:
    POST /hikvision
    POST /dahua
    """
    manufacturer = device.strip().lower()

    if manufacturer in {"hikvision", "hik"}:
        return await receive_camera(request=request)

    if manufacturer in {"dahua", "dh"}:
        return await receive_dahua(request=request, background_tasks=background_tasks)

    logging.warning("[DEVICE] Unsupported manufacturer: %s", device)

    return JSONResponse(
        status_code=404,
        content={
            "error": "Unsupported manufacturer",
            "device_received": device,
            "supported_manufacturers": ["hikvision", "dahua"],
        },
    )


@app.get("/painel", include_in_schema=False)
async def open_panel():
    if not PANEL_HTML.is_file():
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "message": "Panel file not found",
                "path": str(PANEL_HTML),
            },
        )

    return FileResponse(PANEL_HTML)


@app.websocket("/ws/eventos")
async def websocket_events(websocket: WebSocket):
    await event_hub.connect(websocket)

    try:
        while True:
            # The panel sends "ping" periodically to keep the connection open.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        logging.exception("Error in the panel's WebSocket connection")
    finally:
        await event_hub.disconnect(websocket)
