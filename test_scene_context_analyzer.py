import json

from app.domain.models.camera_event import (
    BoundingBox,
)
from app.services.scene_context_analyzer import (
    analisar_contexto_cena,
)


def criar_caixa(
    x: int,
    y: int,
    largura: int,
    altura: int,
    origem: str = "targetrect",
) -> BoundingBox:
    return BoundingBox(
        origem=origem,
        x=x,
        y=y,
        largura=largura,
        altura=altura,
        x2=x + largura,
        y2=y + altura,
        proporcao_imagem=False,
    )


def main() -> None:
    caixas = [
        # Pessoa 1: esquerda
        criar_caixa(
            x=110,
            y=180,
            largura=130,
            altura=360,
        ),

        # Pessoa 2: centro
        criar_caixa(
            x=510,
            y=170,
            largura=135,
            altura=370,
        ),

        # Pessoa 3: centro e próxima da pessoa 2
        criar_caixa(
            x=680,
            y=175,
            largura=130,
            altura=365,
        ),

        # Caixa representando quase a imagem inteira.
        # Ela deverá ser ignorada.
        criar_caixa(
            x=0,
            y=0,
            largura=1280,
            altura=720,
            origem="imagem_inteira",
        ),
    ]

    resultado = analisar_contexto_cena(
        largura_imagem=1280,
        altura_imagem=720,
        bounding_boxes=caixas,
    )

    print(
        "===== CONTEXTO DA CENA ====="
    )

    print(
        json.dumps(
            resultado.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
    )

    assert (
        resultado.quantidade_pessoas
        == 3
    ), (
        "Deveriam ser identificadas "
        "três pessoas"
    )

    assert (
        resultado.pessoas_esquerda
        == 1
    ), (
        "Deveria existir uma pessoa "
        "à esquerda"
    )

    assert (
        resultado.pessoas_centro
        == 2
    ), (
        "Deveriam existir duas pessoas "
        "no centro"
    )

    assert (
        resultado.pessoas_direita
        == 0
    ), (
        "Não deveria existir pessoa "
        "à direita"
    )

    assert (
        len(resultado.proximidades)
        == 3
    ), (
        "Três pessoas devem gerar "
        "três comparações"
    )

    assert (
        resultado.pares_muito_proximos
        == 1
    ), (
        "As pessoas 2 e 3 deveriam estar "
        "muito próximas"
    )

    print(
        "\nQuantidade de pessoas:",
        resultado.quantidade_pessoas,
    )

    print(
        "Pessoas à esquerda:",
        resultado.pessoas_esquerda,
    )

    print(
        "Pessoas no centro:",
        resultado.pessoas_centro,
    )

    print(
        "Pessoas à direita:",
        resultado.pessoas_direita,
    )

    print(
        "Pares muito próximos:",
        resultado.pares_muito_proximos,
    )

    print(
        "Pares próximos:",
        resultado.pares_proximos,
    )

    print(
        "Descrição:",
        resultado.descricao,
    )

    print(
        "\nTeste concluído com sucesso."
    )


if __name__ == "__main__":
    main()