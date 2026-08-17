import json
from datetime import datetime
from pathlib import Path

from app.domain.models.camera_event import (
    BoundingBox,
    CameraEvent,
    ImageData
)
from app.services.alarm_detection_payload import (
    montar_alarm_detection_message
)


CAMINHO_ORIGINAL = (
    "imagens_eventos/"
    "20260717T175437Z_linedetection_"
    "288470f11d7a_original.jpg"
)


evento = CameraEvent(
    evento_id="288470f11d7a",
    fabricante="hikvision",
    modelo_camera=None,
    camera_id="1",
    nome_camera="Camera 01",
    ip_camera="192.168.101.214",
    tipo_evento="linedetection",
    estado="active",
    data_hora=datetime.fromisoformat(
        "2026-07-17T17:54:37+00:00"
    ),
    alvo_detectado="human",
    bounding_boxes=[
        BoundingBox(
            origem="targetrect",
            x=604,
            y=262,
            largura=105,
            altura=190,
            x2=709,
            y2=452,
            proporcao_imagem=0.0216
        )
    ],
    bounding_box_escolhida=BoundingBox(
        origem="targetrect",
        x=604,
        y=262,
        largura=105,
        altura=190,
        x2=709,
        y2=452,
        proporcao_imagem=0.0216
    ),
    imagem=ImageData(
        largura=1280,
        altura=720,
        formato="jpeg",
        caminho_original=CAMINHO_ORIGINAL,
        caminho_marcada=(
            "imagens_eventos/"
            "20260717T175437Z_linedetection_"
            "288470f11d7a_marcada.jpg"
        )
    )
)


payload = montar_alarm_detection_message(evento)

dados = payload.model_dump(mode="json")

caminho_saida = Path("payload_alarm_teste.json")

caminho_saida.write_text(
    json.dumps(
        dados,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

print("===== PAYLOAD CRIADO =====")
print("Device:", payload.device.name)
print("Event ID:", payload.event.id)
print("Event type:", payload.event.type)
print("Timestamp:", payload.event.timestamp)
print("Datetime:", payload.event.datetime)

print(
    "Tamanho background Base64:",
    len(payload.event.images.background),
    "caracteres"
)

print(
    "Tamanho detection Base64:",
    len(payload.event.images.detection),
    "caracteres"
)

print("Arquivo salvo:", caminho_saida)
print("==========================")