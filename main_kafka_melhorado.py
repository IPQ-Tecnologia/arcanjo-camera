import asyncio
import logging
from logging.handlers import RotatingFileHandler
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    BackgroundTasks,
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import (
    FileResponse,
    JSONResponse,
)

from app.core.config import settings
from app.domain.models.camera_event import (
    RawCameraPackage,
)
from app.messaging.kafka_producer_melhorado import (
    kafka_publisher,
)
from app.services.camera_event_pipeline_melhorado import (
    CameraEventPipeline,
)
from app.services.event_hub import event_hub


LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | "
    "%(name)s | %(message)s"
)

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "camera_api_melhorada.log"

root_logger = logging.getLogger()

arquivo_ja_configurado = any(
    getattr(handler, "baseFilename", None)
    and Path(handler.baseFilename).resolve() == LOG_FILE.resolve()
    for handler in root_logger.handlers
)

if not arquivo_ja_configurado:
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(file_handler)


BASE_DIR = Path(__file__).resolve().parent

PAINEL_HTML = (
    BASE_DIR
    / "app"
    / "static"
    / "index.html"
)

PASTA_CAPTURAS_DAHUA = (
    BASE_DIR
    / "capturas_dahua"
)

PASTA_CAPTURAS_HIKVISION = (
    BASE_DIR
    / "capturas_hikvision"
)


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


app = FastAPI(
    title="Camera API com Kafka",
    lifespan=lifespan,
)


def enfileirar_pacote(
    request: Request,
    body: bytes,
    evento_id: str,
    inicio: float,
    origem: str,
):
    """
    Coloca o pacote recebido na fila assíncrona.

    Essa função é utilizada tanto pela Hikvision
    quanto pela Dahua.
    """

    content_type = request.headers.get(
        "content-type",
        "application/octet-stream",
    )

    ip_camera = (
        request.client.host
        if request.client
        else None
    )

    pacote = RawCameraPackage(
        evento_id=evento_id,
        recebido_em=datetime.now(
            timezone.utc
        ),
        content_type=content_type,
        ip_camera=ip_camera,
        caminho_pacote=(
            f"memoria://{evento_id}"
        ),
    )

    try:
        request.app.state.camera_pipeline.adicionar(
            pacote,
            body,
        )

    except asyncio.QueueFull:
        logging.warning(
            "[%s][%s] Fila cheia",
            origem,
            evento_id,
        )

        return JSONResponse(
            status_code=503,
            content={
                "status": "fila_cheia",
                "evento_id": evento_id,
                "origem": origem,
            },
        )

    tempo_ms = round(
        (
            time.perf_counter()
            - inicio
        )
        * 1000,
        2,
    )

    logging.info(
        "[%s][%s] Recebido: "
        "bytes=%s fila=%s/%s "
        "resposta=%sms",
        origem,
        evento_id,
        len(body),
        pipeline.tamanho_fila,
        pipeline.capacidade_fila,
        tempo_ms,
    )

    return {
        "status": "recebido",
        "evento_id": evento_id,
        "origem": origem,
        "processamento": "fila_assincrona",
        "tempo_resposta_ms": tempo_ms,
    }


def salvar_captura_dahua(
    evento_id: str,
    momento: str,
    ip_camera: str,
    content_type: str,
    headers: list[str],
    body: bytes,
) -> None:
    """
    Salva o pacote bruto da Dahua depois que
    a resposta já foi enviada para a câmera.
    """

    PASTA_CAPTURAS_DAHUA.mkdir(
        parents=True,
        exist_ok=True,
    )

    nome_base = (
        f"{momento}_{evento_id}"
    )

    caminho_pacote = (
        PASTA_CAPTURAS_DAHUA
        / f"{nome_base}_pacote.bin"
    )

    caminho_headers = (
        PASTA_CAPTURAS_DAHUA
        / f"{nome_base}_headers.txt"
    )

    caminho_pacote.write_bytes(
        body
    )

    informacoes = [
        f"evento_id: {evento_id}",
        f"recebido_em_utc: {momento}",
        f"ip_camera: {ip_camera}",
        f"content_type: {content_type}",
        f"quantidade_bytes: {len(body)}",
        "",
        "===== HEADERS =====",
        *headers,
    ]

    caminho_headers.write_text(
        "\n".join(informacoes),
        encoding="utf-8",
    )

    logging.info(
        "[DAHUA][%s] Captura salva: "
        "pacote=%s headers=%s "
        "bytes=%s primeiros_bytes=%r",
        evento_id,
        caminho_pacote,
        caminho_headers,
        len(body),
        body[:100],
    )



TIPOS_HIKVISION_ANALISE = {
    "fielddetection",
    "linedetection",
}

LIMITE_CAPTURAS_HIKVISION = 10

TAREFAS_CAPTURA_HIKVISION: set[asyncio.Task] = set()

PADRAO_EVENT_TYPE_HIKVISION = re.compile(
    rb"<(?:[A-Za-z0-9_-]+:)?eventType>\s*([^<]+)",
    flags=re.IGNORECASE,
)


def salvar_captura_hikvision(
    evento_id: str,
    momento: str,
    ip_camera: str,
    content_type: str,
    headers: list[str],
    body: bytes,
) -> None:
    """
    Salva amostras brutas dos eventos da Hikvision
    usados no levantamento de atributos.
    """

    correspondencia = (
        PADRAO_EVENT_TYPE_HIKVISION.search(body)
    )

    if correspondencia is None:
        return

    tipo_evento = (
        correspondencia
        .group(1)
        .decode("utf-8", errors="ignore")
        .strip()
        .lower()
    )

    if tipo_evento not in TIPOS_HIKVISION_ANALISE:
        return

    possui_imagem = b"\xff\xd8\xff" in body

    categoria = (
        "com_imagem"
        if possui_imagem
        else "sem_imagem"
    )

    pasta_evento = (
        PASTA_CAPTURAS_HIKVISION
        / tipo_evento
        / categoria
    )

    pasta_evento.mkdir(
        parents=True,
        exist_ok=True,
    )

    capturas_existentes = list(
        pasta_evento.glob("*_pacote.bin")
    )

    if (
        len(capturas_existentes)
        >= LIMITE_CAPTURAS_HIKVISION
    ):
        return

    nome_base = f"{momento}_{evento_id}"

    caminho_pacote = (
        pasta_evento
        / f"{nome_base}_pacote.bin"
    )

    caminho_headers = (
        pasta_evento
        / f"{nome_base}_headers.txt"
    )

    caminho_pacote.write_bytes(body)

    informacoes = [
        f"evento_id: {evento_id}",
        f"recebido_em_utc: {momento}",
        f"ip_camera: {ip_camera}",
        f"content_type: {content_type}",
        f"tipo_evento: {tipo_evento}",
        f"possui_imagem: {possui_imagem}",
        f"quantidade_bytes: {len(body)}",
        "",
        "===== HEADERS =====",
        *headers,
    ]

    caminho_headers.write_text(
        "\n".join(informacoes),
        encoding="utf-8",
    )

    logging.info(
        "[HIKVISION][%s] Captura de análise salva: "
        "tipo=%s categoria=%s pacote=%s bytes=%s",
        evento_id,
        tipo_evento,
        categoria,
        caminho_pacote,
        len(body),
    )


def agendar_captura_hikvision(
    evento_id: str,
    momento: str,
    ip_camera: str,
    content_type: str,
    headers: list[str],
    body: bytes,
) -> None:
    """
    Executa a gravação fora do fluxo de resposta
    enviado para a câmera.
    """

    tarefa = asyncio.create_task(
        asyncio.to_thread(
            salvar_captura_hikvision,
            evento_id,
            momento,
            ip_camera,
            content_type,
            headers,
            body,
        )
    )

    TAREFAS_CAPTURA_HIKVISION.add(tarefa)

    tarefa.add_done_callback(
        TAREFAS_CAPTURA_HIKVISION.discard
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "kafka_enabled": (
            settings.kafka_enabled
        ),
        "kafka_connected": (
            kafka_publisher.iniciado
        ),
        "kafka_topic": (
            settings.kafka_topic_normalized
        ),
        "fila": {
            "tamanho": (
                pipeline.tamanho_fila
            ),
            "capacidade": (
                pipeline.capacidade_fila
            ),
        },
        "workers": (
            settings.ingestion_workers
        ),
        "rotas_cameras": {
            "hikvision": "/hikvision",
            "dahua": "/dahua",
            "legado_hikvision": "/",
            "legado_dahua": "/CAM7442",
        },
    }


# Rota antiga desativada. Agora é utilizada /hikvision.
@app.post("/hikvision")
async def receber_camera(
    request: Request,
):
    """
    Rota utilizada atualmente pela Hikvision.
    """

    inicio = time.perf_counter()

    evento_id = (
        uuid.uuid4().hex[:12]
    )

    body = await request.body()

    momento = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )

    content_type = request.headers.get(
        "content-type",
        "application/octet-stream",
    )

    ip_camera = (
        request.client.host
        if request.client
        else "desconhecido"
    )

    headers_bloqueados = {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
    }

    headers_seguros = [
        f"{nome}: {valor}"
        for nome, valor in request.headers.items()
        if nome.lower() not in headers_bloqueados
    ]

    agendar_captura_hikvision(
        evento_id=evento_id,
        momento=momento,
        ip_camera=ip_camera,
        content_type=content_type,
        headers=headers_seguros,
        body=body,
    )

    return enfileirar_pacote(
        request=request,
        body=body,
        evento_id=evento_id,
        inicio=inicio,
        origem="HIKVISION",
    )


# Rotas antigas desativadas. Agora é utilizada /dahua.
# @app.post("/dahua")
@app.post("/CAM7442")
async def receber_dahua(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Recebe o evento da Dahua, responde rapidamente
    e salva o pacote bruto em segundo plano.
    """

    inicio = time.perf_counter()

    evento_id = (
        uuid.uuid4().hex[:12]
    )

    momento = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )

    body = await request.body()

    content_type = request.headers.get(
        "content-type",
        "application/octet-stream",
    )

    ip_camera = (
        request.client.host
        if request.client
        else "desconhecido"
    )

    headers_seguros = []

    headers_bloqueados = {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
    }

    for nome, valor in request.headers.items():
        if nome.lower() in headers_bloqueados:
            continue

        headers_seguros.append(
            f"{nome}: {valor}"
        )

    background_tasks.add_task(
        salvar_captura_dahua,
        evento_id,
        momento,
        ip_camera,
        content_type,
        headers_seguros,
        body,
    )

    logging.info(
        "[DAHUA][%s] Requisição recebida: "
        "ip=%s content_type=%s bytes=%s "
        "primeiros_bytes=%r",
        evento_id,
        ip_camera,
        content_type,
        len(body),
        body[:100],
    )

    return enfileirar_pacote(
        request=request,
        body=body,
        evento_id=evento_id,
        inicio=inicio,
        origem="DAHUA",
    )


@app.post("/{device}")
async def receber_por_device(
    device: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Rota dinâmica para recebimento de eventos
    de diferentes fabricantes de câmeras.

    Exemplos:
    POST /hikvision
    POST /dahua
    """

    fabricante = device.strip().lower()

    if fabricante in {
        "hikvision",
        "hik",
    }:
        return await receber_camera(
            request=request,
        )

    if fabricante in {
        "dahua",
        "dh",
    }:
        return await receber_dahua(
            request=request,
            background_tasks=background_tasks,
        )

    logging.warning(
        "[DEVICE] Fabricante não suportado: %s",
        device,
    )

    return JSONResponse(
        status_code=404,
        content={
            "erro": "Fabricante não suportado",
            "device_recebido": device,
            "fabricantes_suportados": [
                "hikvision",
                "dahua",
            ],
        },
    )


@app.get(
    "/painel",
    include_in_schema=False,
)
async def abrir_painel():
    if not PAINEL_HTML.is_file():
        return JSONResponse(
            status_code=404,
            content={
                "status": "erro",
                "mensagem": (
                    "Arquivo do painel "
                    "não encontrado"
                ),
                "caminho": str(
                    PAINEL_HTML
                ),
            },
        )

    return FileResponse(
        PAINEL_HTML
    )


@app.websocket("/ws/eventos")
async def websocket_eventos(
    websocket: WebSocket,
):
    await event_hub.conectar(
        websocket
    )

    try:
        while True:
            # O painel envia "ping" periodicamente
            # para manter a conexão aberta.
            await websocket.receive_text()

    except WebSocketDisconnect:
        pass

    except Exception:
        logging.exception(
            "Erro na conexão WebSocket "
            "do painel"
        )

    finally:
        event_hub.desconectar(
            websocket
        )