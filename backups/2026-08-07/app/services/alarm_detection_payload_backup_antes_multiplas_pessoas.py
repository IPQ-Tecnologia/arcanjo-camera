import base64
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

from app.domain.models.alarm_detection_message import (
    AlarmDetectionMessage,
    AlarmDevice,
    AlarmEventData,
    AlarmImages
)
from app.domain.models.camera_event import (
    BoundingBox,
    CameraEvent
)


def arquivo_para_base64(caminho: str) -> str:
    arquivo = Path(caminho)

    if not arquivo.exists():
        raise FileNotFoundError(
            f"Imagem não encontrada: {arquivo}"
        )

    conteudo = arquivo.read_bytes()

    return base64.b64encode(
        conteudo
    ).decode("ascii")


def normalizar_data_utc(
    data_hora: datetime
) -> datetime:
    if data_hora.tzinfo is None:
        data_hora = data_hora.replace(
            tzinfo=timezone.utc
        )

    return data_hora.astimezone(timezone.utc)


def data_para_milisegundos(
    data_hora: datetime
) -> int:
    data_utc = normalizar_data_utc(data_hora)

    return int(data_utc.timestamp() * 1000)


def data_para_iso8601(
    data_hora: datetime
) -> str:
    data_utc = normalizar_data_utc(data_hora)

    return data_utc.isoformat().replace(
        "+00:00",
        "Z"
    )


def recortar_deteccao_base64(
    caminho_imagem: str,
    bounding_box: BoundingBox
) -> str:
    caminho = Path(caminho_imagem)

    if not caminho.exists():
        raise FileNotFoundError(
            f"Imagem não encontrada: {caminho}"
        )

    with Image.open(caminho) as imagem:
        imagem = imagem.convert("RGB")

        largura_imagem, altura_imagem = imagem.size

        x1 = max(
            0,
            min(bounding_box.x, largura_imagem - 1)
        )

        y1 = max(
            0,
            min(bounding_box.y, altura_imagem - 1)
        )

        x2 = max(
            x1 + 1,
            min(bounding_box.x2, largura_imagem)
        )

        y2 = max(
            y1 + 1,
            min(bounding_box.y2, altura_imagem)
        )

        recorte = imagem.crop(
            (x1, y1, x2, y2)
        )

        buffer = BytesIO()

        recorte.save(
            buffer,
            format="JPEG",
            quality=90
        )

        return base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")


def montar_alarm_detection_message(
    evento: CameraEvent
) -> AlarmDetectionMessage:
    if evento.imagem is None:
        raise ValueError(
            "O evento não possui imagem"
        )

    if not evento.imagem.caminho_original:
        raise ValueError(
            "O evento não possui caminho da imagem original"
        )

    if evento.bounding_box_escolhida is None:
        raise ValueError(
            "O evento não possui bounding box selecionada"
        )

    imagem_background = arquivo_para_base64(
        evento.imagem.caminho_original
    )

    imagem_detection = recortar_deteccao_base64(
        caminho_imagem=evento.imagem.caminho_original,
        bounding_box=evento.bounding_box_escolhida
    )

    nome_camera = (
        evento.nome_camera
        or evento.camera_id
        or "camera-desconhecida"
    )

    return AlarmDetectionMessage(
        device=AlarmDevice(
            name=nome_camera
        ),
        event=AlarmEventData(
            id=evento.evento_id,
            type=evento.tipo_evento,
            timestamp=data_para_milisegundos(
                evento.data_hora
            ),
            datetime=data_para_iso8601(
                evento.data_hora
            ),
            images=AlarmImages(
                background=imagem_background,
                detection=imagem_detection
            )
        )
    )