import json
import threading
import time
import uuid
from datetime import datetime, timezone

from fastapi import BackgroundTasks, FastAPI, Request

from app.adapters.cameras.factory import camera_adapter_factory
from app.domain.models.camera_event import RawCameraPackage


app = FastAPI(
    title="Teste da arquitetura universal de câmeras"
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "modo": "teste_sem_kafka",
        "fabricantes_disponiveis": (
            camera_adapter_factory.listar_fabricantes()
        )
    }


def processar_pacote(
    evento_id: str,
    content_type: str,
    ip_camera: str | None,
    body: bytes,
    recebido_em: datetime
):
    inicio = time.perf_counter()
    thread_atual = threading.current_thread().name

    print(f"\n===== PROCESSAMENTO {evento_id} =====")
    print("Thread:", thread_atual)
    print("Content-Type:", content_type)
    print("Tamanho Body:", len(body))

    try:
        adapter = camera_adapter_factory.encontrar_adapter(
            content_type=content_type,
            body=body
        )

        print(
            f"[{evento_id}] Adaptador escolhido:",
            adapter.__class__.__name__
        )

        pacote = RawCameraPackage(
            evento_id=evento_id,
            recebido_em=recebido_em,
            content_type=content_type,
            ip_camera=ip_camera,
            caminho_pacote=f"memoria://{evento_id}"
        )

        evento = adapter.normalizar(
            pacote=pacote,
            body=body
        )

        tempo_processamento_ms = round(
            (time.perf_counter() - inicio) * 1000,
            2
        )

        dados_evento = evento.model_dump(
            mode="json"
        )

        # Informações apenas para o terminal
        dados_evento["adapter_utilizado"] = (
            adapter.__class__.__name__
        )

        dados_evento["thread_processamento"] = (
            thread_atual
        )

        dados_evento["tempo_processamento_ms"] = (
            tempo_processamento_ms
        )

        print("\n===== EVENTO UNIVERSAL =====")

        print(
            json.dumps(
                dados_evento,
                indent=4,
                ensure_ascii=False
            )
        )

        print(
            f"[{evento_id}] Processamento finalizado "
            f"em {tempo_processamento_ms} ms"
        )

    except ValueError as erro:
        print(
            f"[{evento_id}] Pacote não reconhecido:",
            erro
        )

    except Exception as erro:
        print(
            f"[{evento_id}] Erro ao processar:",
            repr(erro)
        )

    finally:
        print(
            f"===== FIM PROCESSAMENTO {evento_id} =====\n"
        )


@app.post("/")
async def receber_camera(
    request: Request,
    background_tasks: BackgroundTasks
):
    inicio = time.perf_counter()

    evento_id = uuid.uuid4().hex[:12]

    content_type = request.headers.get(
        "content-type",
        "application/octet-stream"
    )

    ip_camera = (
        request.client.host
        if request.client
        else None
    )

    recebido_em = datetime.now(timezone.utc)

    body = await request.body()

    background_tasks.add_task(
        processar_pacote,
        evento_id,
        content_type,
        ip_camera,
        body,
        recebido_em
    )

    tempo_resposta_ms = round(
        (time.perf_counter() - inicio) * 1000,
        2
    )

    print("========== NOVA REQUISIÇÃO ==========")
    print("Evento ID:", evento_id)
    print("Método:", request.method)
    print("URL:", request.url)
    print("IP da câmera:", ip_camera)
    print("Content-Type:", content_type)
    print("Tamanho Body:", len(body))
    print("Enviado para processamento em segundo plano")
    print("Tempo até resposta:", tempo_resposta_ms, "ms")
    print("=====================================")

    return {
        "status": "recebido",
        "mensagem": "Pacote recebido com sucesso",
        "evento_id": evento_id,
        "processamento": "segundo_plano",
        "tempo_resposta_ms": tempo_resposta_ms
    }