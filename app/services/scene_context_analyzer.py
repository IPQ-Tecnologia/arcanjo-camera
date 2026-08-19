from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from app.domain.models.camera_event import BoundingBox


@dataclass(frozen=True)
class ScenePerson:
    """Representa uma pessoa identificada na cena."""

    indice: int
    origem: str

    x: int
    y: int
    largura: int
    altura: int

    centro_x: float
    centro_y: float

    posicao_horizontal: str
    posicao_vertical: str

    tamanho_no_quadro: str
    percentual_quadro: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PersonProximity:
    """Representa a distância entre duas pessoas."""

    pessoa_a: int
    pessoa_b: int

    distancia_normalizada: float
    classificacao: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SceneContextAnalysis:
    """Resultado completo da análise do contexto da cena."""

    quantidade_pessoas: int

    pessoas: list[ScenePerson]
    proximidades: list[PersonProximity]

    pessoas_esquerda: int
    pessoas_centro: int
    pessoas_direita: int

    pares_muito_proximos: int
    pares_proximos: int
    pares_separados: int

    descricao: str

    def to_dict(self) -> dict:
        return {
            "quantidade_pessoas": self.quantidade_pessoas,
            "pessoas": [pessoa.to_dict() for pessoa in self.pessoas],
            "proximidades": [
                proximidade.to_dict() for proximidade in self.proximidades
            ],
            "pessoas_esquerda": self.pessoas_esquerda,
            "pessoas_centro": self.pessoas_centro,
            "pessoas_direita": self.pessoas_direita,
            "pares_muito_proximos": self.pares_muito_proximos,
            "pares_proximos": self.pares_proximos,
            "pares_separados": self.pares_separados,
            "descricao": self.descricao,
        }


def analisar_contexto_cena(
    largura_imagem: int,
    altura_imagem: int,
    bounding_boxes: list[BoundingBox],
) -> SceneContextAnalysis:
    """
    Analisa a distribuição das pessoas na cena.

    A análise considera:

    - quantidade de pessoas;
    - posição horizontal;
    - posição vertical;
    - tamanho no enquadramento;
    - proximidade entre as pessoas.
    """
    if largura_imagem <= 0:
        raise ValueError("A largura da imagem deve ser positiva")

    if altura_imagem <= 0:
        raise ValueError("A altura da imagem deve ser positiva")

    caixas_validas = _filtrar_caixas_validas(
        largura_imagem=largura_imagem,
        altura_imagem=altura_imagem,
        bounding_boxes=bounding_boxes,
    )

    pessoas = [
        _criar_pessoa(
            indice=indice,
            bounding_box=bounding_box,
            largura_imagem=largura_imagem,
            altura_imagem=altura_imagem,
        )
        for indice, bounding_box in enumerate(caixas_validas, start=1)
    ]

    proximidades = _calcular_proximidades(
        pessoas=pessoas,
        largura_imagem=largura_imagem,
        altura_imagem=altura_imagem,
    )

    pessoas_esquerda = sum(
        1 for pessoa in pessoas if pessoa.posicao_horizontal == "esquerda"
    )
    pessoas_centro = sum(
        1 for pessoa in pessoas if pessoa.posicao_horizontal == "centro"
    )
    pessoas_direita = sum(
        1 for pessoa in pessoas if pessoa.posicao_horizontal == "direita"
    )

    pares_muito_proximos = sum(
        1 for proximidade in proximidades if proximidade.classificacao == "muito_proximas"
    )
    pares_proximos = sum(
        1 for proximidade in proximidades if proximidade.classificacao == "proximas"
    )
    pares_separados = sum(
        1 for proximidade in proximidades if proximidade.classificacao == "separadas"
    )

    descricao = _montar_descricao(
        quantidade_pessoas=len(pessoas),
        pessoas_esquerda=pessoas_esquerda,
        pessoas_centro=pessoas_centro,
        pessoas_direita=pessoas_direita,
        pares_muito_proximos=pares_muito_proximos,
        pares_proximos=pares_proximos,
    )

    return SceneContextAnalysis(
        quantidade_pessoas=len(pessoas),
        pessoas=pessoas,
        proximidades=proximidades,
        pessoas_esquerda=pessoas_esquerda,
        pessoas_centro=pessoas_centro,
        pessoas_direita=pessoas_direita,
        pares_muito_proximos=pares_muito_proximos,
        pares_proximos=pares_proximos,
        pares_separados=pares_separados,
        descricao=descricao,
    )


def _filtrar_caixas_validas(
    largura_imagem: int,
    altura_imagem: int,
    bounding_boxes: list[BoundingBox],
) -> list[BoundingBox]:
    """
    Remove caixas inválidas, extremamente pequenas ou que representam
    praticamente a imagem inteira.
    """
    caixas_validas: list[BoundingBox] = []
    area_imagem = largura_imagem * altura_imagem

    for bounding_box in bounding_boxes:
        if bounding_box.largura <= 0 or bounding_box.altura <= 0:
            continue

        if bounding_box.x2 <= bounding_box.x or bounding_box.y2 <= bounding_box.y:
            continue

        area_caixa = bounding_box.largura * bounding_box.altura
        percentual = area_caixa / area_imagem * 100

        # Algumas câmeras enviam uma caixa que representa
        # praticamente a imagem inteira. Essa caixa não corresponde a
        # uma pessoa.
        if percentual >= 60:
            continue

        # Ignora caixas extremamente pequenas.
        if percentual < 0.05:
            continue

        caixas_validas.append(bounding_box)

    return caixas_validas


def _criar_pessoa(
    indice: int,
    bounding_box: BoundingBox,
    largura_imagem: int,
    altura_imagem: int,
) -> ScenePerson:
    centro_x = bounding_box.x + bounding_box.largura / 2
    centro_y = bounding_box.y + bounding_box.altura / 2

    percentual_quadro = round(
        (bounding_box.largura * bounding_box.altura) / (largura_imagem * altura_imagem)
        * 100,
        2,
    )

    return ScenePerson(
        indice=indice,
        origem=bounding_box.origem,
        x=bounding_box.x,
        y=bounding_box.y,
        largura=bounding_box.largura,
        altura=bounding_box.altura,
        centro_x=round(centro_x, 2),
        centro_y=round(centro_y, 2),
        posicao_horizontal=_classificar_posicao_horizontal(
            centro_x=centro_x, largura_imagem=largura_imagem
        ),
        posicao_vertical=_classificar_posicao_vertical(
            centro_y=centro_y, altura_imagem=altura_imagem
        ),
        tamanho_no_quadro=_classificar_tamanho(percentual_quadro),
        percentual_quadro=percentual_quadro,
    )


def _classificar_posicao_horizontal(centro_x: float, largura_imagem: int) -> str:
    proporcao = centro_x / largura_imagem

    if proporcao < 0.34:
        return "esquerda"

    if proporcao < 0.67:
        return "centro"

    return "direita"


def _classificar_posicao_vertical(centro_y: float, altura_imagem: int) -> str:
    proporcao = centro_y / altura_imagem

    if proporcao < 0.34:
        return "superior"

    if proporcao < 0.67:
        return "central"

    return "inferior"


def _classificar_tamanho(percentual_quadro: float) -> str:
    if percentual_quadro < 3:
        return "pequeno"

    if percentual_quadro < 12:
        return "medio"

    return "grande"


def _calcular_proximidades(
    pessoas: list[ScenePerson],
    largura_imagem: int,
    altura_imagem: int,
) -> list[PersonProximity]:
    proximidades: list[PersonProximity] = []

    for indice_a in range(len(pessoas)):
        for indice_b in range(indice_a + 1, len(pessoas)):
            pessoa_a = pessoas[indice_a]
            pessoa_b = pessoas[indice_b]

            diferenca_x = (pessoa_a.centro_x - pessoa_b.centro_x) / largura_imagem
            diferenca_y = (pessoa_a.centro_y - pessoa_b.centro_y) / altura_imagem
            distancia = math.sqrt(diferenca_x**2 + diferenca_y**2)

            classificacao = _classificar_proximidade(distancia)

            proximidades.append(
                PersonProximity(
                    pessoa_a=pessoa_a.indice,
                    pessoa_b=pessoa_b.indice,
                    distancia_normalizada=round(distancia, 3),
                    classificacao=classificacao,
                )
            )

    return proximidades


def _classificar_proximidade(distancia_normalizada: float) -> str:
    """
    A distância é calculada usando as dimensões normalizadas da
    imagem.

    Valores menores representam pessoas mais próximas.
    """
    if distancia_normalizada <= 0.15:
        return "muito_proximas"

    if distancia_normalizada <= 0.30:
        return "proximas"

    return "separadas"


def _montar_descricao(
    quantidade_pessoas: int,
    pessoas_esquerda: int,
    pessoas_centro: int,
    pessoas_direita: int,
    pares_muito_proximos: int,
    pares_proximos: int,
) -> str:
    if quantidade_pessoas == 0:
        return "Nenhuma pessoa identificada na cena."

    if quantidade_pessoas == 1:
        if pessoas_esquerda == 1:
            posicao = "à esquerda"
        elif pessoas_direita == 1:
            posicao = "à direita"
        else:
            posicao = "no centro"

        return f"Uma pessoa identificada {posicao} da cena."

    partes = [f"{quantidade_pessoas} pessoas identificadas na cena."]

    distribuicao: list[str] = []
    if pessoas_esquerda:
        distribuicao.append(f"{pessoas_esquerda} à esquerda")

    if pessoas_centro:
        distribuicao.append(f"{pessoas_centro} no centro")

    if pessoas_direita:
        distribuicao.append(f"{pessoas_direita} à direita")

    if distribuicao:
        partes.append("Distribuição: " + ", ".join(distribuicao) + ".")

    if pares_muito_proximos == 1:
        partes.append("1 par muito próximo.")
    elif pares_muito_proximos > 1:
        partes.append(f"{pares_muito_proximos} pares muito próximos.")

    if pares_proximos == 1:
        partes.append("1 par próximo.")
    elif pares_proximos > 1:
        partes.append(f"{pares_proximos} pares próximos.")

    if pares_muito_proximos == 0 and pares_proximos == 0:
        partes.append("As pessoas estão separadas.")

    return " ".join(partes)
