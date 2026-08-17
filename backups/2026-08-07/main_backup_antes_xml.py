from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
from datetime import datetime, timezone
from threading import current_thread
from time import perf_counter
import base64
import io
import json
import os
import uuid
import xml.etree.ElementTree as ET
import re

app = FastAPI()

# None = automático | 1000 = escala 0..1000 | 1 = escala 0..1 | "pixels" = pixels
ESCALA_COORDENADAS = None
LIMITE_BOX_IMAGEM_INTEIRA = 0.90


@app.get("/")
async def inicio():
    return {"status": "API funcionando"}


def obter_boundary(content_type: str, body: bytes):
    """Obtém o boundary pelo cabeçalho ou pela primeira linha do corpo."""
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


def extrair_imagem(body: bytes, boundary: str):
    for parte in body.split(boundary.encode()):
        if b"Content-Type: image/jpeg" not in parte:
            continue

        inicio = parte.find(b"\xff\xd8\xff")
        if inicio == -1:
            continue

        fim = parte.find(b"\xff\xd9", inicio)
        return parte[inicio:fim + 2] if fim != -1 else parte[inicio:]

    return None


def extrair_xml(body: bytes, boundary: str):
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

        xml_bytes = parte[inicio + 4:]
        fim = xml_bytes.rfind(b">")
        if fim != -1:
            xml_bytes = xml_bytes[:fim + 1]

        return xml_bytes.decode("utf-8", errors="ignore").strip()

    return None


def converter_base64(imagem: bytes):
    return base64.b64encode(imagem).decode("utf-8")


def converter_data_utc(data_hora: str):
    texto = data_hora.strip()
    if texto.endswith("Z"):
        texto = texto[:-1] + "+00:00"

    data = datetime.fromisoformat(texto)
    if data.tzinfo is None:
        data = data.replace(tzinfo=timezone.utc)

    return data.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def pegar_texto(raiz, caminho: str, namespace):
    elemento = raiz.find(caminho, namespace)
    return elemento.text.strip() if elemento is not None and elemento.text else None


def nome_local(tag: str):
    return tag.split("}")[-1].strip().lower()


def numero(valor):
    try:
        return float(str(valor).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def primeiro_valor(valores: dict, nomes: tuple[str, ...]):
    for nome in nomes:
        if nome in valores:
            valor = numero(valores[nome])
            if valor is not None:
                return valor
    return None


def criar_box(elemento):
    valores = {}
    for filho in elemento.iter():
        if filho is not elemento and filho.text:
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


def extrair_bounding_boxes(raiz):
    nomes = (
        "boundingbox", "bounding_box", "bounding-box", "targetrect",
        "targetrectangle", "objectrect", "detectionrect", "rectangle", "rect"
    )
    boxes = []
    assinaturas = set()

    for elemento in raiz.iter():
        tag = nome_local(elemento.tag)
        if not any(nome in tag for nome in nomes):
            continue

        box = criar_box(elemento)
        if box is None:
            continue

        assinatura = tuple(round(box[chave], 6) for chave in ("x", "y", "largura", "altura"))
        if assinatura not in assinaturas:
            assinaturas.add(assinatura)
            boxes.append(box)

    return boxes


def obter_escala_xml(raiz):
    largura = altura = None
    nomes_largura = {"normalizedscreenwidth", "coordinatewidth", "referencewidth"}
    nomes_altura = {"normalizedscreenheight", "coordinateheight", "referenceheight"}

    for elemento in raiz.iter():
        tag = nome_local(elemento.tag)
        if tag in nomes_largura and elemento.text:
            largura = numero(elemento.text)
        if tag in nomes_altura and elemento.text:
            altura = numero(elemento.text)

    return (largura, altura) if largura and altura else None


def definir_escala(boxes, raiz, largura_img, altura_img):
    if ESCALA_COORDENADAS == "pixels":
        return float(largura_img), float(altura_img)
    if isinstance(ESCALA_COORDENADAS, (int, float)):
        escala = float(ESCALA_COORDENADAS)
        return escala, escala

    escala_xml = obter_escala_xml(raiz)
    if escala_xml:
        return escala_xml

    maior_valor = max(
        max(box["x"] + box["largura"], box["y"] + box["altura"])
        for box in boxes
    )
    if maior_valor <= 1.5:
        return 1.0, 1.0

    ultrapassa_imagem = any(
        box["x"] + box["largura"] > largura_img
        or box["y"] + box["altura"] > altura_img
        for box in boxes
    )
    if ultrapassa_imagem and maior_valor <= 1100:
        return 1000.0, 1000.0

    return float(largura_img), float(altura_img)


def converter_boxes_para_pixels(boxes, raiz, largura_img, altura_img):
    if not boxes:
        return []

    escala_x, escala_y = definir_escala(boxes, raiz, largura_img, altura_img)
    resultado = []

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

        resultado.append({
            "origem_xml": box["origem_xml"],
            "x": x,
            "y": y,
            "largura": largura,
            "altura": altura,
            "x2": x + largura,
            "y2": y + altura,
            "proporcao_imagem": round(proporcao, 4),
        })

    return resultado


def escolher_bounding_box(boxes_pixels):
    if not boxes_pixels:
        return None

    especificos = [
        box for box in boxes_pixels
        if box["proporcao_imagem"] < LIMITE_BOX_IMAGEM_INTEIRA
    ]
    candidatos = especificos or boxes_pixels
    return min(candidatos, key=lambda box: box["largura"] * box["altura"])


def sanitizar_nome(valor, padrao):
    texto = str(valor or padrao)
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in texto)


def criar_nome_base(data_hora_utc, tipo_evento, evento_id):
    if data_hora_utc:
        data_nome = data_hora_utc.replace("-", "").replace(":", "").replace(".", "_")
    else:
        data_nome = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    tipo_nome = sanitizar_nome(tipo_evento, "evento")
    return f"{data_nome}_{tipo_nome}_{evento_id}"


def salvar_imagem_original(imagem: bytes, nome_base: str):
    os.makedirs("imagens_eventos", exist_ok=True)
    caminho = os.path.join("imagens_eventos", f"{nome_base}_original.jpg")

    with open(caminho, "wb") as arquivo:
        arquivo.write(imagem)

    return caminho


def salvar_imagem_marcada(imagem: bytes, nome_base: str, box):
    if box is None:
        return None

    os.makedirs("imagens_eventos", exist_ok=True)
    caminho = os.path.join("imagens_eventos", f"{nome_base}_marcada.jpg")

    with Image.open(io.BytesIO(imagem)) as imagem_pil:
        imagem_pil = imagem_pil.convert("RGB")
        desenho = ImageDraw.Draw(imagem_pil)
        desenho.rectangle(
            [box["x"], box["y"], box["x2"], box["y2"]],
            outline="red",
            width=5,
        )
        imagem_pil.save(caminho, "JPEG", quality=95)

    return caminho


def processar_evento(body: bytes, content_type: str, evento_id: str):
    inicio_processamento = perf_counter()
    nome_thread = current_thread().name

    print(f"\n===== PROCESSAMENTO {evento_id} =====")
    print("Thread:", nome_thread)
    print("Tamanho Body:", len(body))

    try:
        boundary = obter_boundary(content_type, body)

        if not boundary:
            print(f"[{evento_id}] Boundary não encontrado no cabeçalho nem no corpo")
            return

        imagem = extrair_imagem(body, boundary)
        if imagem is None:
            print(f"[{evento_id}] Imagem não encontrada")
            return

        xml = extrair_xml(body, boundary)
        if xml is None:
            print(f"[{evento_id}] XML não encontrado")
            return

        raiz = ET.fromstring(xml)
        namespace = {"ns": "http://www.hikvision.com/ver20/XMLSchema"}

        tipo_evento = pegar_texto(raiz, "ns:eventType", namespace)
        data_hora = pegar_texto(raiz, "ns:dateTime", namespace)
        estado = pegar_texto(raiz, "ns:eventState", namespace)
        nome_camera = pegar_texto(raiz, "ns:channelName", namespace)
        alvo_detectado = pegar_texto(raiz, ".//ns:detectionTarget", namespace)
        data_hora_utc = converter_data_utc(data_hora) if data_hora else None

        with Image.open(io.BytesIO(imagem)) as imagem_pil:
            largura_img, altura_img = imagem_pil.size

        boxes_xml = extrair_bounding_boxes(raiz)
        boxes_pixels = converter_boxes_para_pixels(
            boxes_xml, raiz, largura_img, altura_img
        )
        box_escolhido = escolher_bounding_box(boxes_pixels)

        nome_base = criar_nome_base(data_hora_utc, tipo_evento, evento_id)
        caminho_original = salvar_imagem_original(imagem, nome_base)
        caminho_marcada = salvar_imagem_marcada(imagem, nome_base, box_escolhido)
        imagem_base64 = converter_base64(imagem)

        tempo_processamento_ms = round(
            (perf_counter() - inicio_processamento) * 1000,
            2
        )

        dados_evento = {
            "evento_id": evento_id,
            "tipo_evento": tipo_evento,
            "data_hora": data_hora_utc,
            "estado": estado,
            "nome_camera": nome_camera,
            "alvo_detectado": alvo_detectado,
            "thread_processamento": nome_thread,
            "tempo_processamento_ms": tempo_processamento_ms,
            "dimensoes_imagem": {"largura": largura_img, "altura": altura_img},
            "quantidade_bounding_boxes": len(boxes_pixels),
            "bounding_boxes_encontradas": boxes_pixels,
            "bounding_box_escolhida": box_escolhido,
            "caminho_imagem_original": caminho_original,
            "caminho_imagem_marcada": caminho_marcada,
            "imagem_base64": imagem_base64,
        }

        dados_log = dados_evento.copy()
        dados_log["imagem_base64"] = f"{len(imagem_base64)} caracteres"

        print("\n===== JSON DO EVENTO =====")
        print(json.dumps(dados_log, indent=4, ensure_ascii=False))
        print(f"[{evento_id}] Processamento finalizado em {tempo_processamento_ms} ms")

    except Exception as erro:
        tempo_erro_ms = round((perf_counter() - inicio_processamento) * 1000, 2)
        print(f"[{evento_id}] Erro após {tempo_erro_ms} ms: {erro}")

    finally:
        print(f"===== FIM PROCESSAMENTO {evento_id} =====\n")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    inicio_requisicao = perf_counter()

    print("========== NOVA REQUISIÇÃO ==========")
    print("Método:", request.method)
    print("URL:", request.url)

    response = await call_next(request)

    tempo_resposta_ms = round((perf_counter() - inicio_requisicao) * 1000, 2)
    print("Status:", response.status_code)
    print("Tempo até resposta:", tempo_resposta_ms, "ms")
    print("=====================================")

    return response


@app.post("/")
async def receber_camera(request: Request, background_tasks: BackgroundTasks):
    inicio_recebimento = perf_counter()
    content_type = request.headers.get("content-type", "")
    body = await request.body()
    evento_id = uuid.uuid4().hex[:12]

    print(f"[{evento_id}] Content-Type: {content_type or 'não informado'}")
    print(f"[{evento_id}] Pacote recebido com {len(body)} bytes")

    if body:
        background_tasks.add_task(
            processar_evento,
            body,
            content_type,
            evento_id,
        )
        processamento = "segundo_plano"
        print(f"[{evento_id}] Enviado para processamento em segundo plano")
    else:
        processamento = "ignorado_pacote_vazio"
        print(f"[{evento_id}] Pacote vazio; nada para processar")

    tempo_recebimento_ms = round(
        (perf_counter() - inicio_recebimento) * 1000,
        2,
    )

    # A câmera recebe 200 imediatamente e fica livre para enviar o próximo evento.
    return {
        "status": "recebido",
        "mensagem": "Pacote recebido com sucesso",
        "evento_id": evento_id,
        "processamento": processamento,
        "tempo_recebimento_ms": tempo_recebimento_ms,
    }