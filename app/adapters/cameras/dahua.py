from __future__ import annotations

import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from app.adapters.cameras.base import CameraAdapter
from app.domain.models.camera_event import (
    BoundingBox,
    CameraEvent,
    EventAttributes,
    EventPoint,
    ImageData,
    RawCameraPackage,
)


PASTA_IMAGENS = Path("imagens_eventos")
ESCALA_NORMALIZADA_DAHUA = 8191.0


def obter_boundary(content_type: str, body: bytes) -> str | None:
    correspondencia = re.search(
        r'boundary\s*=\s*["\']?([^;"\'\s]+)',
        content_type or "",
        flags=re.IGNORECASE,
    )
    if correspondencia:
        return correspondencia.group(1).strip()

    primeira_linha = body.split(b"\r\n", 1)[0].strip()
    if primeira_linha.startswith(b"--"):
        return primeira_linha[2:].decode("utf-8", errors="ignore").strip()

    return None


def carregar_json(conteudo: bytes) -> dict[str, Any]:
    texto = conteudo.decode("utf-8-sig", errors="ignore").strip()
    if not texto:
        raise ValueError("JSON Dahua vazio")

    dados = json.loads(texto)
    if not isinstance(dados, dict):
        raise ValueError("JSON Dahua não é um objeto")

    return dados


def extrair_jpeg(conteudo: bytes) -> bytes | None:
    inicio = conteudo.find(b"\xff\xd8\xff")
    if inicio == -1:
        return None

    fim = conteudo.find(b"\xff\xd9", inicio)
    if fim == -1:
        return conteudo[inicio:]

    return conteudo[inicio : fim + 2]


def extrair_multipart(
    content_type: str,
    body: bytes,
) -> tuple[dict[str, Any], bytes | None]:
    boundary = obter_boundary(content_type, body)
    if not boundary:
        raise ValueError("Boundary Dahua não encontrado")

    delimitador = b"--" + boundary.encode("utf-8", errors="ignore")

    payload_json: dict[str, Any] | None = None
    imagem: bytes | None = None

    for parte in body.split(delimitador):
        parte = parte.strip(b"\r\n")
        if not parte or parte == b"--":
            continue

        if b"\r\n\r\n" not in parte:
            continue

        cabecalho, conteudo = parte.split(b"\r\n\r\n", 1)
        cabecalho_lower = cabecalho.lower()

        if b"application/json" in cabecalho_lower or b"text/plain" in cabecalho_lower:
            try:
                payload_json = carregar_json(conteudo)
            except (json.JSONDecodeError, ValueError):
                continue

        if b"image/jpeg" in cabecalho_lower:
            imagem = extrair_jpeg(conteudo)

    if payload_json is None:
        raise ValueError("Metadados JSON não encontrados no multipart Dahua")

    return payload_json, imagem


def selecionar_evento(payload: dict[str, Any]) -> dict[str, Any]:
    eventos = payload.get("Events")
    if isinstance(eventos, list) and eventos:
        primeiro = eventos[0]
        if isinstance(primeiro, dict):
            return primeiro

    return payload


def converter_data_hora(payload: dict[str, Any], evento: dict[str, Any]) -> datetime:
    dados = evento.get("Data")
    if not isinstance(dados, dict):
        dados = {}

    for chave in ("RealUTC", "UTC"):
        valor = dados.get(chave)
        if valor is None:
            continue

        try:
            timestamp = float(valor)
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            continue

    texto = payload.get("Time")
    if isinstance(texto, str) and texto.strip():
        try:
            data = datetime.strptime(texto.strip(), "%Y-%m-%d %H:%M:%S")
            return data.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    return datetime.now(timezone.utc)


def sanitizar_nome(valor: str) -> str:
    resultado = re.sub(r"[^a-zA-Z0-9_-]+", "_", valor.strip())

    return resultado.strip("_") or "evento"


def criar_nome_base(data_hora: datetime, tipo_evento: str, evento_id: str) -> str:
    horario = data_hora.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tipo = sanitizar_nome(tipo_evento)

    return f"{horario}_{tipo}_{evento_id}"


def obter_bbox_bruta(objeto: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = objeto.get("BoundingBox")

    if isinstance(bbox, list) and len(bbox) >= 4:
        try:
            return tuple(float(valor) for valor in bbox[:4])
        except (TypeError, ValueError):
            return None

    if not isinstance(bbox, dict):
        return None

    chaves_limites = (
        ("Left", "Top", "Right", "Bottom"),
        ("left", "top", "right", "bottom"),
        ("X1", "Y1", "X2", "Y2"),
        ("x1", "y1", "x2", "y2"),
    )

    for chaves in chaves_limites:
        if all(chave in bbox for chave in chaves):
            try:
                return tuple(float(bbox[chave]) for chave in chaves)
            except (TypeError, ValueError):
                return None

    chaves_tamanho = (
        ("X", "Y", "Width", "Height"),
        ("x", "y", "width", "height"),
    )

    for x_chave, y_chave, w_chave, h_chave in chaves_tamanho:
        if all(chave in bbox for chave in (x_chave, y_chave, w_chave, h_chave)):
            try:
                x = float(bbox[x_chave])
                y = float(bbox[y_chave])
                largura = float(bbox[w_chave])
                altura = float(bbox[h_chave])

                return (x, y, x + largura, y + altura)
            except (TypeError, ValueError):
                return None

    return None


def detectar_escala_normalizada(
    dados: dict[str, Any],
    bbox: tuple[float, float, float, float],
    largura_img: int,
    altura_img: int,
) -> bool:
    regiao = dados.get("DetectRegion")
    coordenadas_regiao: list[float] = []

    if isinstance(regiao, list):
        for ponto in regiao:
            if not isinstance(ponto, list):
                continue

            for valor in ponto:
                try:
                    coordenadas_regiao.append(float(valor))
                except (TypeError, ValueError):
                    continue

    if coordenadas_regiao:
        return max(coordenadas_regiao) > max(largura_img, altura_img)

    x1, y1, x2, y2 = bbox

    return x1 > largura_img or x2 > largura_img or y1 > altura_img or y2 > altura_img


def converter_bbox_para_pixels(
    bbox: tuple[float, float, float, float],
    dados: dict[str, Any],
    largura_img: int,
    altura_img: int,
) -> BoundingBox | None:
    x1, y1, x2, y2 = bbox

    maximo = max(abs(x1), abs(y1), abs(x2), abs(y2))

    if maximo <= 1.0:
        x1 *= largura_img
        x2 *= largura_img
        y1 *= altura_img
        y2 *= altura_img
    elif detectar_escala_normalizada(dados, bbox, largura_img, altura_img):
        x1 = x1 / ESCALA_NORMALIZADA_DAHUA * largura_img
        x2 = x2 / ESCALA_NORMALIZADA_DAHUA * largura_img
        y1 = y1 / ESCALA_NORMALIZADA_DAHUA * altura_img
        y2 = y2 / ESCALA_NORMALIZADA_DAHUA * altura_img

    x1_int = max(0, min(largura_img - 1, round(x1)))
    y1_int = max(0, min(altura_img - 1, round(y1)))
    x2_int = max(0, min(largura_img - 1, round(x2)))
    y2_int = max(0, min(altura_img - 1, round(y2)))

    if x2_int <= x1_int or y2_int <= y1_int:
        return None

    largura = x2_int - x1_int
    altura = y2_int - y1_int
    proporcao = largura * altura / (largura_img * altura_img)

    return BoundingBox(
        origem="dahua_object_bounding_box",
        x=x1_int,
        y=y1_int,
        largura=largura,
        altura=altura,
        x2=x2_int,
        y2=y2_int,
        proporcao_imagem=proporcao,
    )


def salvar_imagens(
    imagem: bytes,
    nome_base: str,
    bbox: BoundingBox | None,
) -> tuple[str, str, int, int]:
    PASTA_IMAGENS.mkdir(parents=True, exist_ok=True)

    caminho_original = PASTA_IMAGENS / f"{nome_base}_original.jpg"
    caminho_marcada = PASTA_IMAGENS / f"{nome_base}_marcada.jpg"

    caminho_original.write_bytes(imagem)

    with Image.open(io.BytesIO(imagem)) as imagem_pil:
        imagem_rgb = imagem_pil.convert("RGB")
        largura_img, altura_img = imagem_rgb.size

        if bbox is not None:
            desenho = ImageDraw.Draw(imagem_rgb)
            espessura = max(2, min(largura_img, altura_img) // 300)
            desenho.rectangle(
                [bbox.x, bbox.y, bbox.x2, bbox.y2],
                outline="red",
                width=espessura,
            )

        imagem_rgb.save(caminho_marcada, format="JPEG", quality=95)

    return str(caminho_original), str(caminho_marcada), largura_img, altura_img


def _numero_attributes_dahua(valor) -> float | None:
    if valor is None or isinstance(valor, bool):
        return None

    try:
        return float(str(valor).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def _texto_attributes_dahua(valor) -> str | None:
    if valor is None:
        return None

    if isinstance(valor, (dict, list, tuple)):
        return None

    texto = str(valor).strip()

    return texto or None


def _normalizar_coordenada_dahua(valor) -> float | None:
    numero = _numero_attributes_dahua(valor)
    if numero is None:
        return None

    # Os pontos de região observados na Dahua usam escala de 0 a 8191.
    if numero > 1:
        numero = numero / 8191

    return max(0.0, min(1.0, numero))


def _extrair_geometria_dahua(valor) -> list[EventPoint]:
    pontos: list[EventPoint] = []

    def percorrer(item) -> None:
        if isinstance(item, dict):
            chaves = {str(chave).lower(): conteudo for chave, conteudo in item.items()}

            x = None
            y = None

            for nome in ("x", "positionx", "left"):
                if nome in chaves:
                    x = _normalizar_coordenada_dahua(chaves[nome])
                    break

            for nome in ("y", "positiony", "top"):
                if nome in chaves:
                    y = _normalizar_coordenada_dahua(chaves[nome])
                    break

            if x is not None and y is not None:
                pontos.append(EventPoint(x=x, y=y))
                return

            for conteudo in item.values():
                percorrer(conteudo)

            return

        if isinstance(item, (list, tuple)):
            if len(item) >= 2:
                x = _normalizar_coordenada_dahua(item[0])
                y = _normalizar_coordenada_dahua(item[1])

                if x is not None and y is not None:
                    pontos.append(EventPoint(x=x, y=y))
                    return

            for conteudo in item:
                percorrer(conteudo)

    percorrer(valor)

    # Remove pontos repetidos mantendo a ordem.
    resultado: list[EventPoint] = []
    encontrados: set[tuple[float, float]] = set()

    for ponto in pontos:
        assinatura = (round(ponto.x, 6), round(ponto.y, 6))
        if assinatura in encontrados:
            continue

        encontrados.add(assinatura)
        resultado.append(ponto)

    return resultado


def _normalizar_target_type_dahua(alvo_detectado: str | None) -> str | None:
    if not alvo_detectado:
        return None

    alvo = str(alvo_detectado).strip().lower()
    mapeamento = {
        "human": "person",
        "person": "person",
        "vehicle": "vehicle",
        "car": "vehicle",
        "automobile": "vehicle",
    }

    return mapeamento.get(alvo, alvo.replace(" ", "_").replace("-", "_"))


def _normalizar_event_type_dahua(tipo_evento: str, target_type: str | None) -> str:
    if target_type == "person":
        return "person_detection"

    if target_type == "vehicle":
        return "vehicle_detection"

    tipo = str(tipo_evento or "").strip().lower()
    mapeamento = {
        "crossregiondetection": "intrusion_detection",
        "tripwiredetection": "line_crossing_detection",
        "leftdetection": "object_left_detection",
        "motiondetect": "motion_detection",
        "videoloss": "video_loss",
    }

    return mapeamento.get(tipo, tipo.replace(" ", "_").replace("-", "_") or "unknown_event")


def _categoria_evento_dahua(tipo_evento: str) -> str:
    tipo_normalizado = tipo_evento.strip().lower()

    categorias = {
        "crossregiondetection": "intrusion",
        "leftdetection": "object_left",
        "tripwiredetection": "line_crossing",
        "motiondetect": "motion",
        "videoloss": "video_loss",
    }

    return categorias.get(tipo_normalizado, tipo_normalizado)


def _montar_atributos_dahua(
    payload: dict,
    dados: dict,
    objeto: dict,
    tipo_evento: str,
    estado: str | None,
    alvo_detectado: str | None,
    acao_original,
) -> EventAttributes:
    payload = payload if isinstance(payload, dict) else {}
    dados = dados if isinstance(dados, dict) else {}
    objeto = objeto if isinstance(objeto, dict) else {}

    geometria = _extrair_geometria_dahua(
        dados.get("DetectRegion") or dados.get("Region") or dados.get("Polygon")
    )

    if len(geometria) == 2:
        geometry_type = "line"
    elif len(geometria) >= 3:
        geometry_type = "region"
    else:
        geometry_type = None

    acao = dados.get("Action") or objeto.get("Action") or acao_original
    direcao = (
        dados.get("Direction") or objeto.get("Direction") or objeto.get(
            "humanTripLineDirection"
        )
    )
    direcao_normalizada = _texto_attributes_dahua(direcao)

    if direcao_normalizada and direcao_normalizada.lower() in {
        "0",
        "0.0",
        "unknown",
        "none",
    }:
        direcao_normalizada = None

    confidence = _numero_attributes_dahua(objeto.get("Confidence"))

    # Nas capturas analisadas, zero significa que a confiança não foi
    # disponibilizada.
    if confidence is not None and confidence <= 0:
        confidence = None

    bbox_bruta = obter_bbox_bruta(objeto)

    return EventAttributes(
        manufacturer="dahua",
        vendor_event_type=tipo_evento,
        category=_categoria_evento_dahua(tipo_evento),
        target_type=alvo_detectado,
        state=estado,
        action=_texto_attributes_dahua(acao),
        source_event_id=_texto_attributes_dahua(
            dados.get("EventID") or payload.get("EventID")
        ),
        rule_id=_texto_attributes_dahua(dados.get("RuleID")),
        rule_name=_texto_attributes_dahua(dados.get("Name")),
        group_id=_texto_attributes_dahua(dados.get("GroupID")),
        object_id=_texto_attributes_dahua(objeto.get("ObjectID")),
        sensitivity=None,
        direction=direcao_normalizada,
        confidence=confidence,
        geometry_type=geometry_type,
        geometry=geometria,
        raw_bounding_box=(
            [float(valor) for valor in bbox_bruta] if bbox_bruta is not None else None
        ),
        vendor_data={
            chave: valor
            for chave, valor in {
                "vendor_target_type": _texto_attributes_dahua(objeto.get("ObjectType")),
                "vendor_state": estado,
                "event_action": _texto_attributes_dahua(acao_original),
            }.items()
            if valor is not None
        },
    )


class DahuaAdapter(CameraAdapter):
    fabricante = "dahua"

    def consegue_processar(self, content_type: str, body: bytes) -> bool:
        tipo = (content_type or "").lower()
        corpo = body.lower()

        possui_estrutura_dahua = (
            b'"code"' in corpo
            and b'"action"' in corpo
            and (
                b'"objecttype"' in corpo
                or b'"detectregion"' in corpo
                or b'"events"' in corpo
                or b'"picturetype"' in corpo
            )
        )

        return possui_estrutura_dahua and (
            "application/json" in tipo
            or "multipart/x-mixed-replace" in tipo
            or corpo.startswith(b"{")
            or corpo.startswith(b"--")
        )

    def normalizar(self, pacote: RawCameraPackage, body: bytes) -> CameraEvent:
        content_type = pacote.content_type or ""
        tipo_conteudo = content_type.lower()

        if "multipart/x-mixed-replace" in tipo_conteudo:
            formato_pacote = "multipart"
            payload, imagem = extrair_multipart(content_type, body)
        else:
            formato_pacote = "json_direto"
            payload = carregar_json(body)
            imagem = None

        evento = selecionar_evento(payload)

        dados = evento.get("Data")
        if not isinstance(dados, dict):
            dados = {}

        objeto = dados.get("Object")
        if not isinstance(objeto, dict):
            objeto = {}

        tipo_evento = str(evento.get("Code") or "desconhecido")
        acao_original = str(evento.get("Action") or dados.get("Action") or "").strip()

        estados = {
            "start": "active",
            "appear": "active",
            "pulse": "active",
            "stop": "inactive",
            "disappear": "inactive",
        }
        estado = estados.get(acao_original.lower(), acao_original.lower() or None)

        data_hora = converter_data_hora(payload, evento)

        canal = payload.get("Channel")
        if canal is None:
            canal = evento.get("Index")

        camera_id = str(canal) if canal is not None else None
        nome_regra = dados.get("Name")
        nome_camera = f"Dahua Canal {camera_id}" if camera_id is not None else "Dahua"

        alvo = objeto.get("ObjectType")
        alvo_detectado = str(alvo).lower() if alvo is not None else None

        alvo_normalizado = _normalizar_target_type_dahua(alvo_detectado)
        tipo_evento_normalizado = _normalizar_event_type_dahua(
            tipo_evento, alvo_normalizado
        )

        image_data: ImageData | None = None
        bounding_box: BoundingBox | None = None

        caminho_original = None
        caminho_marcada = None
        largura_img = None
        altura_img = None

        bbox_bruta = obter_bbox_bruta(objeto)

        if imagem is not None:
            with Image.open(io.BytesIO(imagem)) as imagem_pil:
                largura_img, altura_img = imagem_pil.size

            if bbox_bruta is not None:
                bounding_box = converter_bbox_para_pixels(
                    bbox_bruta, dados, largura_img, altura_img
                )

            nome_base = criar_nome_base(data_hora, tipo_evento, pacote.evento_id)
            caminho_original, caminho_marcada, largura_img, altura_img = salvar_imagens(
                imagem, nome_base, bounding_box
            )

            image_data = ImageData(
                largura=largura_img,
                altura=altura_img,
                formato="jpeg",
                caminho_original=caminho_original,
                caminho_marcada=caminho_marcada,
            )

        bounding_boxes = [bounding_box] if bounding_box is not None else []

        return CameraEvent(
            evento_id=pacote.evento_id,
            fabricante=self.fabricante,
            modelo_camera=None,
            camera_id=camera_id,
            nome_camera=nome_camera,
            ip_camera=pacote.ip_camera,
            tipo_evento=tipo_evento_normalizado,
            estado=estado,
            data_hora=data_hora,
            alvo_detectado=alvo_normalizado,
            attributes=_montar_atributos_dahua(
                payload=payload,
                dados=dados,
                objeto=objeto,
                tipo_evento=tipo_evento,
                estado=estado,
                alvo_detectado=alvo_detectado,
                acao_original=acao_original,
            ),
            bounding_boxes=bounding_boxes,
            bounding_box_escolhida=bounding_box,
            imagem=image_data,
            dados_extras={
                "formato_pacote": formato_pacote,
                "content_type": content_type,
                "acao_original": acao_original,
                "nome_regra": nome_regra,
                "event_id_dahua": dados.get("EventID"),
                "group_id": dados.get("GroupID"),
                "rule_id": dados.get("RuleID"),
                "object_id": objeto.get("ObjectID"),
                "confidence": objeto.get("Confidence"),
                "bounding_box_bruta": bbox_bruta,
                "imagem_recebida": imagem is not None,
                "resolucao_payload": payload.get("Resolution"),
            },
        )
