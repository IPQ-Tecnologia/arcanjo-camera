import io
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

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


logger = logging.getLogger(__name__)


ESCALA_COORDENADAS = None
LIMITE_BOX_IMAGEM_INTEIRA = 0.90
PASTA_IMAGENS = Path("imagens_eventos")
NAMESPACE_HIKVISION = {"ns": "http://www.hikvision.com/ver20/XMLSchema"}


def obter_boundary(content_type: str, body: bytes) -> str | None:
    correspondencia = re.search(
        r"boundary\s*=\s*[\"']?([^;\"'\s]+)",
        content_type or "",
        flags=re.IGNORECASE,
    )
    if correspondencia:
        return correspondencia.group(1).strip()

    primeira_linha = body.split(b"\r\n", 1)[0].strip()
    if primeira_linha.startswith(b"--") and len(primeira_linha) > 2:
        return primeira_linha[2:].decode("utf-8", errors="ignore").strip()

    return None


def extrair_imagem(body: bytes, boundary: str) -> bytes | None:
    for parte in body.split(boundary.encode()):
        if b"Content-Type: image/jpeg" not in parte:
            continue

        inicio = parte.find(b"\xff\xd8\xff")
        if inicio == -1:
            continue

        fim = parte.find(b"\xff\xd9", inicio)
        if fim != -1:
            return parte[inicio : fim + 2]

        return parte[inicio:]

    return None


def extrair_xml(body: bytes, boundary: str) -> str | None:
    for parte in body.split(boundary.encode()):
        eh_xml = (
            b"Content-Type: application/xml" in parte
            or b"Content-Type: text/xml" in parte
        )
        if not eh_xml:
            continue

        inicio = parte.find(b"\r\n\r\n")
        if inicio == -1:
            continue

        xml_bytes = parte[inicio + 4 :]
        fim = xml_bytes.rfind(b">")
        if fim != -1:
            xml_bytes = xml_bytes[: fim + 1]

        return xml_bytes.decode("utf-8", errors="ignore").strip()

    return None


def limpar_xml_direto(body: bytes) -> str:
    xml = body.decode("utf-8-sig", errors="ignore").strip()

    inicio = xml.find("<")
    fim = xml.rfind(">")
    if inicio == -1 or fim == -1:
        raise ValueError("Corpo XML inválido")

    return xml[inicio : fim + 1]


def converter_data_utc(data_hora: str | None) -> datetime:
    if not data_hora:
        data_utc = datetime.now(timezone.utc)
        logger.info(
            "[CONVERSÃO DATA HIKVISION] data_hora vazia; usando=%s", data_utc.isoformat()
        )
        return data_utc

    logger.info("[CONVERSÃO DATA HIKVISION] data_hora_original=%r", data_hora)

    texto = data_hora.strip()
    logger.info("[CONVERSÃO DATA HIKVISION] texto_apos_strip=%r", texto)

    if texto.endswith("Z"):
        texto = texto[:-1] + "+00:00"

    logger.info("[CONVERSÃO DATA HIKVISION] texto_para_fromisoformat=%r", texto)

    data = datetime.fromisoformat(texto)
    logger.info(
        "[CONVERSÃO DATA HIKVISION] data_fromisoformat=%s tzinfo=%s",
        data.isoformat(),
        data.tzinfo,
    )

    if data.tzinfo is None:
        data = data.replace(tzinfo=timezone.utc)

    data_utc = data.astimezone(timezone.utc)
    logger.info("[CONVERSÃO DATA HIKVISION] data_astimezone_utc=%s", data_utc.isoformat())

    return data_utc


def pegar_texto(raiz: ET.Element, caminho: str, namespace: dict[str, str]) -> str | None:
    elemento = raiz.find(caminho, namespace)
    if elemento is not None and elemento.text:
        return elemento.text.strip()

    return None


def pegar_texto_por_nome_local(raiz: ET.Element, nomes: tuple[str, ...]) -> str | None:
    nomes_normalizados = {nome.lower() for nome in nomes}

    for elemento in raiz.iter():
        if nome_local(elemento.tag) not in nomes_normalizados:
            continue

        if elemento.text and elemento.text.strip():
            return elemento.text.strip()

    return None


def nome_local(tag: str) -> str:
    return tag.split("}")[-1].strip().lower()


def numero(valor) -> float | None:
    try:
        return float(str(valor).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def primeiro_valor(valores: dict[str, str], nomes: tuple[str, ...]) -> float | None:
    for nome in nomes:
        if nome not in valores:
            continue

        valor = numero(valores[nome])
        if valor is not None:
            return valor

    return None


def criar_box(elemento: ET.Element) -> dict | None:
    valores: dict[str, str] = {}

    for filho in elemento.iter():
        if filho is elemento or not filho.text:
            continue

        chave = nome_local(filho.tag)
        valores.setdefault(chave, filho.text.strip())

    x = primeiro_valor(valores, ("x", "positionx", "left", "xmin", "startx"))
    y = primeiro_valor(valores, ("y", "positiony", "top", "ymin", "starty"))
    largura = primeiro_valor(valores, ("width", "largura", "boxwidth", "targetwidth"))
    altura = primeiro_valor(valores, ("height", "altura", "boxheight", "targetheight"))
    direita = primeiro_valor(valores, ("right", "xmax", "endx", "x2"))
    inferior = primeiro_valor(valores, ("bottom", "ymax", "endy", "y2"))

    if largura is None and x is not None and direita is not None:
        largura = direita - x

    if altura is None and y is not None and inferior is not None:
        altura = inferior - y

    if any(valor is None for valor in (x, y, largura, altura)):
        return None

    if largura <= 0 or altura <= 0:
        return None

    return {
        "origem_xml": nome_local(elemento.tag),
        "x": x,
        "y": y,
        "largura": largura,
        "altura": altura,
    }


def extrair_bounding_boxes(raiz: ET.Element) -> list[dict]:
    """
    Extrai somente caixas que podem representar o alvo detectado.

    A Hikvision também envia retângulos referentes à região configurada
    da regra. Essas regiões não devem ser tratadas como pessoas.
    """
    grupos_prioridade = (
        ("targetrect", "targetrectangle", "objectrect"),
        ("boundingbox", "bounding_box", "bounding-box"),
        ("detectionrect",),
    )

    for nomes in grupos_prioridade:
        boxes: list[dict] = []
        assinaturas: set[tuple[float, float, float, float]] = set()

        for elemento in raiz.iter():
            tag = nome_local(elemento.tag)
            if tag not in nomes:
                continue

            box = criar_box(elemento)
            if box is None:
                continue

            assinatura = tuple(
                round(box[chave], 6) for chave in ("x", "y", "largura", "altura")
            )
            if assinatura in assinaturas:
                continue

            assinaturas.add(assinatura)
            boxes.append(box)

        # Usa o primeiro grupo que realmente encontrou caixas.
        # targetRect tem prioridade sobre regiões genéricas.
        if boxes:
            return boxes

    return []


def obter_escala_xml(raiz: ET.Element) -> tuple[float, float] | None:
    largura = None
    altura = None

    nomes_largura = {"normalizedscreenwidth", "coordinatewidth", "referencewidth"}
    nomes_altura = {"normalizedscreenheight", "coordinateheight", "referenceheight"}

    for elemento in raiz.iter():
        tag = nome_local(elemento.tag)
        if tag in nomes_largura and elemento.text:
            largura = numero(elemento.text)

        if tag in nomes_altura and elemento.text:
            altura = numero(elemento.text)

    if largura and altura:
        return largura, altura

    return None


def definir_escala(
    boxes: list[dict],
    raiz: ET.Element,
    largura_img: int,
    altura_img: int,
) -> tuple[float, float]:
    if ESCALA_COORDENADAS == "pixels":
        return float(largura_img), float(altura_img)

    if isinstance(ESCALA_COORDENADAS, (int, float)):
        escala = float(ESCALA_COORDENADAS)
        return escala, escala

    escala_xml = obter_escala_xml(raiz)
    if escala_xml:
        return escala_xml

    maior_valor = max(
        max(box["x"] + box["largura"], box["y"] + box["altura"]) for box in boxes
    )
    if maior_valor <= 1.5:
        return 1.0, 1.0

    ultrapassa_imagem = any(
        box["x"] + box["largura"] > largura_img or box["y"] + box["altura"] > altura_img
        for box in boxes
    )
    if ultrapassa_imagem and maior_valor <= 1100:
        return 1000.0, 1000.0

    return float(largura_img), float(altura_img)


def converter_boxes_para_pixels(
    boxes: list[dict],
    raiz: ET.Element,
    largura_img: int,
    altura_img: int,
) -> list[dict]:
    if not boxes:
        return []

    escala_x, escala_y = definir_escala(boxes, raiz, largura_img, altura_img)

    resultado: list[dict] = []

    for box in boxes:
        x = round(box["x"] * largura_img / escala_x)
        y = round(box["y"] * altura_img / escala_y)
        largura = round(box["largura"] * largura_img / escala_x)
        altura = round(box["altura"] * altura_img / escala_y)

        x = max(0, min(x, largura_img - 1))
        y = max(0, min(y, altura_img - 1))
        largura = max(1, min(largura, largura_img - x))
        altura = max(1, min(altura, altura_img - y))

        proporcao = (largura * altura) / (largura_img * altura_img)

        resultado.append(
            {
                "origem_xml": box["origem_xml"],
                "x": x,
                "y": y,
                "largura": largura,
                "altura": altura,
                "x2": x + largura,
                "y2": y + altura,
                "proporcao_imagem": round(proporcao, 4),
            }
        )

    return resultado


def escolher_bounding_box(boxes_pixels: list[dict]) -> dict | None:
    if not boxes_pixels:
        return None

    especificos = [
        box for box in boxes_pixels if box["proporcao_imagem"] < LIMITE_BOX_IMAGEM_INTEIRA
    ]
    candidatos = especificos or boxes_pixels

    return min(candidatos, key=lambda box: box["largura"] * box["altura"])


def sanitizar_nome(valor, padrao: str) -> str:
    texto = str(valor or padrao)
    return "".join(
        caractere if caractere.isalnum() or caractere in "-_" else "_"
        for caractere in texto
    )


def criar_nome_base(data_hora: datetime, tipo_evento: str | None, evento_id: str) -> str:
    data_nome = data_hora.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tipo_nome = sanitizar_nome(tipo_evento, "evento")

    return f"{data_nome}_{tipo_nome}_{evento_id}"


def salvar_imagem_original(imagem: bytes, nome_base: str) -> str:
    PASTA_IMAGENS.mkdir(parents=True, exist_ok=True)
    caminho = PASTA_IMAGENS / f"{nome_base}_original.jpg"
    caminho.write_bytes(imagem)

    return str(caminho)


def salvar_imagem_marcada(imagem: bytes, nome_base: str, box: dict | None) -> str | None:
    if box is None:
        return None

    PASTA_IMAGENS.mkdir(parents=True, exist_ok=True)
    caminho = PASTA_IMAGENS / f"{nome_base}_marcada.jpg"

    with Image.open(io.BytesIO(imagem)) as imagem_pil:
        imagem_pil = imagem_pil.convert("RGB")
        desenho = ImageDraw.Draw(imagem_pil)
        desenho.rectangle(
            [box["x"], box["y"], box["x2"], box["y2"]],
            outline="red",
            width=5,
        )
        imagem_pil.save(caminho, "JPEG", quality=95)

    return str(caminho)


def converter_box_modelo(box: dict | None) -> BoundingBox | None:
    if box is None:
        return None

    return BoundingBox(
        origem=box["origem_xml"],
        x=box["x"],
        y=box["y"],
        largura=box["largura"],
        altura=box["altura"],
        x2=box["x2"],
        y2=box["y2"],
        proporcao_imagem=box.get("proporcao_imagem"),
    )


def _nome_local_attributes_hikvision(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _primeiro_texto_attributes_hikvision(
    raiz: ET.Element, nomes: tuple[str, ...]
) -> str | None:
    nomes_normalizados = {nome.lower() for nome in nomes}

    for elemento in raiz.iter():
        nome = _nome_local_attributes_hikvision(elemento.tag)
        if nome not in nomes_normalizados:
            continue

        if elemento.text and elemento.text.strip():
            return elemento.text.strip()

    return None


def _numero_attributes_hikvision(valor) -> float | None:
    try:
        return float(str(valor).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def _normalizar_coordenada_hikvision(valor) -> float | None:
    numero = _numero_attributes_hikvision(valor)
    if numero is None:
        return None

    # As coordenadas das regiões Hikvision normalmente usam uma escala
    # de 0 a 1000.
    if numero > 1:
        numero = numero / 1000

    return max(0.0, min(1.0, numero))


def _extrair_geometria_hikvision(raiz: ET.Element) -> list[EventPoint]:
    pontos: list[EventPoint] = []

    for elemento in raiz.iter():
        if _nome_local_attributes_hikvision(elemento.tag) != "regioncoordinates":
            continue

        x = None
        y = None

        for filho in elemento.iter():
            nome = _nome_local_attributes_hikvision(filho.tag)
            if not filho.text:
                continue

            if nome == "positionx":
                x = _normalizar_coordenada_hikvision(filho.text)
            elif nome == "positiony":
                y = _normalizar_coordenada_hikvision(filho.text)

        if x is not None and y is not None:
            pontos.append(EventPoint(x=x, y=y))

    return pontos


def _normalizar_target_type_hikvision(
    alvo_detectado: str | None,
    tipo_evento: str | None = None,
) -> str | None:
    if alvo_detectado:
        alvo = str(alvo_detectado).strip().lower()
        mapeamento = {
            "human": "person",
            "person": "person",
            "vehicle": "vehicle",
            "car": "vehicle",
            "face": "face",
        }
        return mapeamento.get(alvo, alvo.replace(" ", "_").replace("-", "_"))

    tipo = str(tipo_evento or "").strip().lower()
    if tipo == "facedetection":
        return "face"

    return None


def _normalizar_event_type_hikvision(tipo_evento: str, target_type: str | None) -> str:
    if target_type == "person":
        return "person_detection"

    if target_type == "vehicle":
        return "vehicle_detection"

    if target_type == "face":
        return "face_detection"

    tipo = str(tipo_evento or "").strip().lower()
    mapeamento = {
        "fielddetection": "intrusion_detection",
        "linedetection": "line_crossing_detection",
        "vmd": "motion_detection",
        "videoloss": "video_loss",
        "facedetection": "face_detection",
        "duration": "duration",
    }

    return mapeamento.get(tipo, tipo.replace(" ", "_").replace("-", "_") or "unknown_event")


def _categoria_evento_hikvision(tipo_evento: str) -> str:
    categorias = {
        "fielddetection": "intrusion",
        "linedetection": "line_crossing",
        "vmd": "motion",
        "videoloss": "video_loss",
    }
    tipo_normalizado = tipo_evento.strip().lower()

    return categorias.get(tipo_normalizado, tipo_normalizado)


def _montar_atributos_hikvision(
    raiz: ET.Element,
    tipo_evento: str,
    estado: str | None,
    alvo_detectado: str | None,
) -> EventAttributes:
    geometria = _extrair_geometria_hikvision(raiz)
    tipo_normalizado = tipo_evento.strip().lower()

    if tipo_normalizado == "linedetection":
        geometry_type = "line"
    elif tipo_normalizado == "fielddetection":
        geometry_type = "region"
    elif len(geometria) == 2:
        geometry_type = "line"
    elif len(geometria) >= 3:
        geometry_type = "region"
    else:
        geometry_type = None

    source_event_id = None
    for nome, valor in raiz.attrib.items():
        if _nome_local_attributes_hikvision(nome) == "id":
            source_event_id = str(valor)
            break

    sensitivity = _numero_attributes_hikvision(
        _primeiro_texto_attributes_hikvision(raiz, ("sensitivityLevel",))
    )

    return EventAttributes(
        manufacturer="hikvision",
        vendor_event_type=tipo_evento,
        category=_categoria_evento_hikvision(tipo_evento),
        target_type=alvo_detectado,
        state=estado,
        action=None,
        source_event_id=source_event_id,
        rule_id=_primeiro_texto_attributes_hikvision(raiz, ("regionID", "ruleID")),
        rule_name=_primeiro_texto_attributes_hikvision(
            raiz, ("ruleName", "regionName", "lineName")
        ),
        object_id=None,
        sensitivity=sensitivity,
        direction=_primeiro_texto_attributes_hikvision(
            raiz, ("direction", "targetDirection", "lineCrossingDirection")
        ),
        confidence=None,
        geometry_type=geometry_type,
        geometry=geometria,
        vendor_data={
            key: value
            for key, value in {
                "vendor_target_type": alvo_detectado,
                "vendor_state": estado,
            }.items()
            if value is not None
        },
    )


class HikvisionAdapter(CameraAdapter):
    fabricante = "hikvision"

    def consegue_processar(self, content_type: str, body: bytes) -> bool:
        body_lower = body.lower()

        return (
            b"hikvision.com" in body_lower
            or b"eventnotificationalert" in body_lower
            or b"<eventtype>" in body_lower
        )

    def normalizar(self, pacote: RawCameraPackage, body: bytes) -> CameraEvent:
        content_type = pacote.content_type or ""
        tipo_conteudo = content_type.lower()
        xml_direto = "application/xml" in tipo_conteudo or "text/xml" in tipo_conteudo

        imagem: bytes | None = None

        if xml_direto:
            formato_pacote = "xml_direto"
            xml = limpar_xml_direto(body)
        else:
            formato_pacote = "multipart"
            boundary = obter_boundary(content_type, body)
            if not boundary:
                raise ValueError("Boundary não encontrado")

            xml = extrair_xml(body, boundary)
            imagem = extrair_imagem(body, boundary)
            if xml is None:
                raise ValueError("XML não encontrado no multipart")

        raiz = ET.fromstring(xml)

        tipo_evento = (
            pegar_texto(raiz, "ns:eventType", NAMESPACE_HIKVISION) or "desconhecido"
        )
        data_hora_texto = pegar_texto(raiz, "ns:dateTime", NAMESPACE_HIKVISION)
        data_hora = converter_data_utc(data_hora_texto)
        estado = pegar_texto(raiz, "ns:eventState", NAMESPACE_HIKVISION)
        nome_camera = pegar_texto(raiz, "ns:channelName", NAMESPACE_HIKVISION)
        camera_id = pegar_texto(raiz, "ns:channelID", NAMESPACE_HIKVISION)
        modelo_camera = pegar_texto_por_nome_local(
            raiz, ("deviceModel", "model", "modelName")
        )
        alvo_detectado = pegar_texto(raiz, ".//ns:detectionTarget", NAMESPACE_HIKVISION)

        alvo_normalizado = _normalizar_target_type_hikvision(alvo_detectado, tipo_evento)
        tipo_evento_normalizado = _normalizar_event_type_hikvision(
            tipo_evento, alvo_normalizado
        )

        boxes_xml = extrair_bounding_boxes(raiz)
        boxes_pixels: list[dict] = []
        box_escolhido: dict | None = None

        largura_img = None
        altura_img = None
        caminho_original = None
        caminho_marcada = None
        image_data = None

        if imagem is not None:
            with Image.open(io.BytesIO(imagem)) as imagem_pil:
                largura_img, altura_img = imagem_pil.size

            boxes_pixels = converter_boxes_para_pixels(
                boxes_xml, raiz, largura_img, altura_img
            )
            box_escolhido = escolher_bounding_box(boxes_pixels)

            nome_base = criar_nome_base(data_hora, tipo_evento, pacote.evento_id)
            caminho_original = salvar_imagem_original(imagem, nome_base)
            caminho_marcada = salvar_imagem_marcada(imagem, nome_base, box_escolhido)

            image_data = ImageData(
                largura=largura_img,
                altura=altura_img,
                formato="jpeg",
                caminho_original=caminho_original,
                caminho_marcada=caminho_marcada,
            )

        boxes_modelo = [converter_box_modelo(box) for box in boxes_pixels]

        return CameraEvent(
            evento_id=pacote.evento_id,
            fabricante=self.fabricante,
            modelo_camera=modelo_camera,
            camera_id=camera_id,
            nome_camera=nome_camera,
            ip_camera=pacote.ip_camera,
            tipo_evento=tipo_evento_normalizado,
            estado=estado,
            data_hora=data_hora,
            alvo_detectado=alvo_normalizado,
            attributes=_montar_atributos_hikvision(
                raiz=raiz,
                tipo_evento=tipo_evento,
                estado=estado,
                alvo_detectado=alvo_detectado,
            ),
            bounding_boxes=[box for box in boxes_modelo if box is not None],
            bounding_box_escolhida=converter_box_modelo(box_escolhido),
            imagem=image_data,
            dados_extras={
                "formato_pacote": formato_pacote,
                "imagem_recebida": imagem is not None,
                "quantidade_bounding_boxes": len(boxes_pixels),
                "bounding_boxes_xml": boxes_xml,
                "content_type": content_type,
            },
        )
