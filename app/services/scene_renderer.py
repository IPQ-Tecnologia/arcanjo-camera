from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.domain.models.camera_event import (
    BoundingBox,
)


@dataclass(frozen=True)
class DrawableBoundingBox:
    """
    Bounding box já convertida para pixels e pronta
    para ser desenhada sobre a imagem.
    """

    origem: str

    x1: int
    y1: int
    x2: int
    y2: int

    largura: int
    altura: int

    percentual_quadro: float


@dataclass(frozen=True)
class SceneRenderResult:
    """
    Resultado da renderização da cena.
    """

    imagem_base64: str
    quantidade_boxes: int
    largura_imagem: int
    altura_imagem: int


def renderizar_cena_com_boxes(
    caminho_imagem: str,
    bounding_boxes: list[BoundingBox],
) -> SceneRenderResult:
    """
    Abre a imagem original e desenha todas as
    bounding boxes válidas.

    Retorna a imagem marcada em Base64.
    """

    caminho = Path(caminho_imagem)

    if not caminho.is_file():
        raise FileNotFoundError(
            f"Imagem não encontrada: {caminho}"
        )

    with Image.open(caminho) as imagem_aberta:
        imagem = imagem_aberta.convert("RGB")

    largura_imagem, altura_imagem = (
        imagem.size
    )

    caixas_validas = (
        _preparar_bounding_boxes(
            bounding_boxes=bounding_boxes,
            largura_imagem=largura_imagem,
            altura_imagem=altura_imagem,
        )
    )

    desenho = ImageDraw.Draw(
        imagem
    )

    fonte = ImageFont.load_default()

    cores = [
        (239, 68, 68),
        (34, 197, 94),
        (59, 130, 246),
        (234, 179, 8),
        (168, 85, 247),
        (236, 72, 153),
        (6, 182, 212),
        (249, 115, 22),
    ]

    largura_linha = max(
        2,
        min(
            6,
            round(
                largura_imagem / 450
            ),
        ),
    )

    raio_centro = max(
        3,
        largura_linha + 1,
    )

    for indice, caixa in enumerate(
        caixas_validas,
        start=1,
    ):
        cor = cores[
            (indice - 1) % len(cores)
        ]

        desenho.rectangle(
            (
                caixa.x1,
                caixa.y1,
                caixa.x2,
                caixa.y2,
            ),
            outline=cor,
            width=largura_linha,
        )

        centro_x = int(
            (
                caixa.x1
                + caixa.x2
            )
            / 2
        )

        centro_y = int(
            (
                caixa.y1
                + caixa.y2
            )
            / 2
        )

        desenho.ellipse(
            (
                centro_x - raio_centro,
                centro_y - raio_centro,
                centro_x + raio_centro,
                centro_y + raio_centro,
            ),
            fill=cor,
        )

        texto = (
            f"Pessoa {indice} "
            f"({caixa.percentual_quadro:.2f}%)"
        )

        limites_texto = desenho.textbbox(
            (0, 0),
            texto,
            font=fonte,
        )

        largura_texto = (
            limites_texto[2]
            - limites_texto[0]
        )

        altura_texto = (
            limites_texto[3]
            - limites_texto[1]
        )

        margem_texto = 5

        texto_x1 = caixa.x1

        texto_y1 = max(
            0,
            caixa.y1
            - altura_texto
            - margem_texto * 2,
        )

        texto_x2 = min(
            largura_imagem - 1,
            texto_x1
            + largura_texto
            + margem_texto * 2,
        )

        texto_y2 = min(
            altura_imagem - 1,
            texto_y1
            + altura_texto
            + margem_texto * 2,
        )

        desenho.rectangle(
            (
                texto_x1,
                texto_y1,
                texto_x2,
                texto_y2,
            ),
            fill=cor,
        )

        desenho.text(
            (
                texto_x1 + margem_texto,
                texto_y1 + margem_texto,
            ),
            texto,
            fill=(255, 255, 255),
            font=fonte,
        )

    buffer = BytesIO()

    imagem.save(
        buffer,
        format="JPEG",
        quality=88,
        optimize=True,
    )

    imagem_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return SceneRenderResult(
        imagem_base64=imagem_base64,
        quantidade_boxes=len(
            caixas_validas
        ),
        largura_imagem=largura_imagem,
        altura_imagem=altura_imagem,
    )


def _preparar_bounding_boxes(
    bounding_boxes: list[BoundingBox],
    largura_imagem: int,
    altura_imagem: int,
) -> list[DrawableBoundingBox]:
    """
    Converte coordenadas para pixels, limita as
    caixas ao tamanho da imagem e remove caixas
    inválidas ou duplicadas.
    """

    caixas_validas: list[
        DrawableBoundingBox
    ] = []

    caixas_encontradas: set[
        tuple[int, int, int, int]
    ] = set()

    area_imagem = (
        largura_imagem
        * altura_imagem
    )

    for bounding_box in bounding_boxes:
        coordenadas = (
            _converter_para_pixels(
                bounding_box=bounding_box,
                largura_imagem=largura_imagem,
                altura_imagem=altura_imagem,
            )
        )

        if coordenadas is None:
            continue

        x1, y1, x2, y2 = coordenadas

        x1 = max(
            0,
            min(
                largura_imagem - 1,
                x1,
            ),
        )

        y1 = max(
            0,
            min(
                altura_imagem - 1,
                y1,
            ),
        )

        x2 = max(
            0,
            min(
                largura_imagem - 1,
                x2,
            ),
        )

        y2 = max(
            0,
            min(
                altura_imagem - 1,
                y2,
            ),
        )

        if (
            x2 <= x1
            or y2 <= y1
        ):
            continue

        largura = x2 - x1
        altura = y2 - y1

        area_caixa = (
            largura
            * altura
        )

        percentual_quadro = (
            area_caixa
            / area_imagem
            * 100
        )

        # Ignora a caixa que representa quase
        # toda a imagem.
        if percentual_quadro >= 60:
            continue

        # Ignora caixas pequenas demais para
        # representar uma detecção útil.
        if percentual_quadro < 0.05:
            continue

        chave = (
            x1,
            y1,
            x2,
            y2,
        )

        if chave in caixas_encontradas:
            continue

        caixas_encontradas.add(
            chave
        )

        caixas_validas.append(
            DrawableBoundingBox(
                origem=(
                    bounding_box.origem
                    or "desconhecida"
                ),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                largura=largura,
                altura=altura,
                percentual_quadro=round(
                    percentual_quadro,
                    2,
                ),
            )
        )

    return caixas_validas


def _converter_para_pixels(
    bounding_box: BoundingBox,
    largura_imagem: int,
    altura_imagem: int,
) -> tuple[int, int, int, int] | None:
    """
    Suporta coordenadas em pixels e coordenadas
    normalizadas entre zero e um.
    """

    try:
        x = float(
            bounding_box.x
        )

        y = float(
            bounding_box.y
        )

        largura = float(
            bounding_box.largura
        )

        altura = float(
            bounding_box.altura
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if (
        largura <= 0
        or altura <= 0
    ):
        return None

    valores_normalizados = (
        0 <= x <= 1
        and 0 <= y <= 1
        and 0 < largura <= 1
        and 0 < altura <= 1
    )

    if valores_normalizados:
        x1 = round(
            x * largura_imagem
        )

        y1 = round(
            y * altura_imagem
        )

        x2 = round(
            (x + largura)
            * largura_imagem
        )

        y2 = round(
            (y + altura)
            * altura_imagem
        )

    else:
        x1 = round(x)
        y1 = round(y)

        x2 = round(
            x + largura
        )

        y2 = round(
            y + altura
        )

    return (
        x1,
        y1,
        x2,
        y2,
    )