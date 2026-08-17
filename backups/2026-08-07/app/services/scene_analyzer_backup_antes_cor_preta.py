from __future__ import annotations

import colorsys
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from PIL import Image


PosicaoHorizontal = Literal[
    "esquerda",
    "centro",
    "direita",
]

TamanhoNoQuadro = Literal[
    "pequeno",
    "medio",
    "grande",
]


@dataclass(frozen=True)
class PersonVisualAnalysis:
    cor_roupa_aproximada: str

    rgb_representativo: tuple[
        int,
        int,
        int,
    ]

    posicao_horizontal: PosicaoHorizontal
    tamanho_no_quadro: TamanhoNoQuadro
    percentual_quadro: float
    descricao: str

    def to_dict(self) -> dict:
        dados = asdict(self)

        dados["rgb_representativo"] = list(
            self.rgb_representativo
        )

        return dados


CORES_REFERENCIA: dict[
    str,
    tuple[int, int, int],
] = {
    "vermelha": (190, 45, 45),
    "laranja": (220, 115, 35),
    "amarela": (220, 200, 45),
    "verde": (55, 145, 70),
    "azul": (55, 95, 180),
    "roxa": (125, 65, 155),
    "rosa": (215, 105, 150),
    "marrom": (110, 70, 45),
    "bege": (195, 170, 125),
}


def analisar_pessoa(
    caminho_imagem: str | Path,
    x: int,
    y: int,
    largura: int,
    altura: int,
) -> PersonVisualAnalysis:
    caminho = Path(caminho_imagem)

    if not caminho.exists():
        raise FileNotFoundError(
            f"Imagem não encontrada: {caminho}"
        )

    if largura <= 0 or altura <= 0:
        raise ValueError(
            "A bounding box deve possuir largura "
            "e altura positivas"
        )

    with Image.open(caminho) as imagem_aberta:
        imagem = imagem_aberta.convert("RGB")

        largura_imagem, altura_imagem = (
            imagem.size
        )

        x1 = max(
            0,
            min(x, largura_imagem - 1),
        )

        y1 = max(
            0,
            min(y, altura_imagem - 1),
        )

        x2 = max(
            x1 + 1,
            min(
                x + largura,
                largura_imagem,
            ),
        )

        y2 = max(
            y1 + 1,
            min(
                y + altura,
                altura_imagem,
            ),
        )

        largura_pessoa = x2 - x1
        altura_pessoa = y2 - y1

        regiao_roupa = _recortar_regiao_roupa(
            imagem=imagem,
            x1=x1,
            y1=y1,
            largura=largura_pessoa,
            altura=altura_pessoa,
        )

        rgb = _obter_cor_representativa(
            regiao_roupa
        )

        cor = _classificar_cor(rgb)

        posicao = _classificar_posicao(
            centro_x=(
                x1 + largura_pessoa / 2
            ),
            largura_imagem=largura_imagem,
        )

        percentual_quadro = round(
            (
                largura_pessoa
                * altura_pessoa
            )
            / (
                largura_imagem
                * altura_imagem
            )
            * 100,
            2,
        )

        tamanho = _classificar_tamanho(
            percentual_quadro
        )

        descricao = _montar_descricao(
            cor=cor,
            posicao=posicao,
            tamanho=tamanho,
        )

        return PersonVisualAnalysis(
            cor_roupa_aproximada=cor,
            rgb_representativo=rgb,
            posicao_horizontal=posicao,
            tamanho_no_quadro=tamanho,
            percentual_quadro=percentual_quadro,
            descricao=descricao,
        )


def _recortar_regiao_roupa(
    imagem: Image.Image,
    x1: int,
    y1: int,
    largura: int,
    altura: int,
) -> Image.Image:
    """
    Recorta a região central do tronco.

    Evita usar toda a bounding box, pois ela pode
    conter bastante fundo, cabeça e pernas.
    """

    roupa_x1 = int(
        x1 + largura * 0.20
    )

    roupa_x2 = int(
        x1 + largura * 0.80
    )

    roupa_y1 = int(
        y1 + altura * 0.22
    )

    roupa_y2 = int(
        y1 + altura * 0.62
    )

    roupa_x2 = max(
        roupa_x1 + 1,
        roupa_x2,
    )

    roupa_y2 = max(
        roupa_y1 + 1,
        roupa_y2,
    )

    return imagem.crop(
        (
            roupa_x1,
            roupa_y1,
            roupa_x2,
            roupa_y2,
        )
    )


def _obter_cor_representativa(
    regiao: Image.Image,
) -> tuple[int, int, int]:
    """
    Reduz a região e encontra a cor mais frequente.
    """

    regiao_reduzida = regiao.resize(
        (48, 48),
        Image.Resampling.LANCZOS,
    )

    regiao_quantizada = (
        regiao_reduzida
        .quantize(
            colors=6,
            method=Image.Quantize.MEDIANCUT,
        )
        .convert("RGB")
    )

    contagem = Counter(
        regiao_quantizada.getdata()
    )

    cor, _ = contagem.most_common(1)[0]

    return (
        int(cor[0]),
        int(cor[1]),
        int(cor[2]),
    )


def _classificar_cor(
    rgb: tuple[int, int, int],
) -> str:
    vermelho, verde, azul = rgb

    r = vermelho / 255
    g = verde / 255
    b = azul / 255

    _, saturacao, brilho = (
        colorsys.rgb_to_hsv(
            r,
            g,
            b,
        )
    )

    if brilho <= 0.20:
        return "preta"

    if (
        saturacao <= 0.14
        and brilho >= 0.84
    ):
        return "branca"

    if saturacao <= 0.18:
        return "cinza"

    melhor_cor = "indefinida"
    menor_distancia = math.inf

    for nome, referencia in (
        CORES_REFERENCIA.items()
    ):
        distancia = math.sqrt(
            (
                vermelho
                - referencia[0]
            ) ** 2
            + (
                verde
                - referencia[1]
            ) ** 2
            + (
                azul
                - referencia[2]
            ) ** 2
        )

        if distancia < menor_distancia:
            menor_distancia = distancia
            melhor_cor = nome

    return melhor_cor


def _classificar_posicao(
    centro_x: float,
    largura_imagem: int,
) -> PosicaoHorizontal:
    proporcao = (
        centro_x / largura_imagem
    )

    if proporcao < 0.34:
        return "esquerda"

    if proporcao < 0.67:
        return "centro"

    return "direita"


def _classificar_tamanho(
    percentual_quadro: float,
) -> TamanhoNoQuadro:
    if percentual_quadro < 3:
        return "pequeno"

    if percentual_quadro < 12:
        return "medio"

    return "grande"


def _montar_descricao(
    cor: str,
    posicao: PosicaoHorizontal,
    tamanho: TamanhoNoQuadro,
) -> str:
    descricoes_tamanho = {
        "pequeno": (
            "ocupando uma pequena parte "
            "do enquadramento"
        ),
        "medio": (
            "ocupando uma parte intermediária "
            "do enquadramento"
        ),
        "grande": (
            "ocupando uma grande parte "
            "do enquadramento"
        ),
    }

    return (
        "Pessoa com roupa predominantemente "
        f"{cor}, localizada no {posicao} da cena, "
        f"{descricoes_tamanho[tamanho]}."
    )