from __future__ import annotations

import colorsys
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Literal

from PIL import Image, ImageFilter


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

    if not caminho.is_file():
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
            min(
                int(x),
                largura_imagem - 1,
            ),
        )

        y1 = max(
            0,
            min(
                int(y),
                altura_imagem - 1,
            ),
        )

        x2 = max(
            x1 + 1,
            min(
                int(x + largura),
                largura_imagem,
            ),
        )

        y2 = max(
            y1 + 1,
            min(
                int(y + altura),
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

        cor = _classificar_cor(
            rgb
        )

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
    Recorta somente a parte central do tronco.

    O recorte evita:

    - cabeça e rosto;
    - braços e mãos;
    - pernas;
    - bordas da bounding box;
    - parte do fundo ao redor da pessoa.
    """

    roupa_x1 = int(
        x1 + largura * 0.25
    )

    roupa_x2 = int(
        x1 + largura * 0.75
    )

    roupa_y1 = int(
        y1 + altura * 0.27
    )

    roupa_y2 = int(
        y1 + altura * 0.64
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
    Calcula uma cor robusta para a roupa.

    Em vez de pegar simplesmente a cor que mais
    aparece, o método:

    - reduz a imagem;
    - remove ruídos;
    - descarta parte dos reflexos mais claros;
    - calcula a mediana de cada canal RGB.

    A mediana sofre menos influência da iluminação,
    do fundo e de pequenos reflexos.
    """

    regiao_reduzida = regiao.resize(
        (64, 64),
        Image.Resampling.LANCZOS,
    )

    regiao_suavizada = (
        regiao_reduzida.filter(
            ImageFilter.MedianFilter(
                size=3
            )
        )
    )

    pixels = list(
        regiao_suavizada.getdata()
    )

    if not pixels:
        return (0, 0, 0)

    luminancias = sorted(
        _calcular_luminancia(pixel)
        for pixel in pixels
    )

    indice_limite = min(
        len(luminancias) - 1,
        int(
            len(luminancias) * 0.85
        ),
    )

    limite_luminancia = (
        luminancias[indice_limite]
    )

    # Descarta aproximadamente os 15% de pixels
    # mais claros. Normalmente são reflexos, teto,
    # paredes ou fundo entrando na bounding box.
    candidatos = [
        pixel
        for pixel in pixels
        if (
            _calcular_luminancia(pixel)
            <= limite_luminancia
        )
    ]

    quantidade_minima = max(
        100,
        len(pixels) // 3,
    )

    if (
        len(candidatos)
        < quantidade_minima
    ):
        candidatos = pixels

    vermelho = int(
        round(
            median(
                pixel[0]
                for pixel in candidatos
            )
        )
    )

    verde = int(
        round(
            median(
                pixel[1]
                for pixel in candidatos
            )
        )
    )

    azul = int(
        round(
            median(
                pixel[2]
                for pixel in candidatos
            )
        )
    )

    return (
        vermelho,
        verde,
        azul,
    )


def _calcular_luminancia(
    rgb: tuple[int, int, int],
) -> float:
    """
    Calcula a luminosidade percebida pelo olho
    humano.

    O canal verde possui maior peso porque o olho
    humano é mais sensível a ele.
    """

    vermelho, verde, azul = rgb

    return (
        vermelho * 0.2126
        + verde * 0.7152
        + azul * 0.0722
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

    luminancia = _calcular_luminancia(
        rgb
    )

    maior_canal = max(
        vermelho,
        verde,
        azul,
    )

    menor_canal = min(
        vermelho,
        verde,
        azul,
    )

    diferenca_canais = (
        maior_canal
        - menor_canal
    )

   
    if brilho <= 0.32:
        return "preta"

   
    if (
        luminancia <= 88
        and saturacao <= 0.52
    ):
        return "preta"

    
    if (
        brilho <= 0.42
        and diferenca_canais <= 45
    ):
        return "preta"

    if (
        luminancia <= 105
        and diferenca_canais <= 32
    ):
        return "preta"

    # Branco: alta luminosidade e pouca saturação.
    if (
        luminancia >= 218
        and saturacao <= 0.18
    ):
        return "branca"

    # Tons neutros sem cor forte.
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
        centro_x
        / largura_imagem
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
    descricoes_posicao = {
        "esquerda": "à esquerda",
        "centro": "no centro",
        "direita": "à direita",
    }

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
        f"{cor}, localizada "
        f"{descricoes_posicao[posicao]} da cena, "
        f"{descricoes_tamanho[tamanho]}."
    )