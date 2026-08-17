import asyncio

from app.services.appearance_memory import (
    AppearanceMemory,
)
from app.services.scene_analyzer import (
    PersonVisualAnalysis,
)


def criar_analise(
    cor: str,
) -> PersonVisualAnalysis:
    return PersonVisualAnalysis(
        cor_roupa_aproximada=cor,
        rgb_representativo=(40, 40, 40),
        posicao_horizontal="centro",
        tamanho_no_quadro="medio",
        percentual_quadro=4.0,
        descricao="Teste",
    )


async def executar(
    titulo: str,
    cores: list[str],
) -> list[str]:
    memoria = AppearanceMemory()
    resultados: list[str] = []

    print("\n" + "=" * 60)
    print(titulo)

    for indice, cor in enumerate(
        cores,
        start=1,
    ):
        resultado = await memoria.registrar(
            pessoa_id="pessoa-teste",
            analise=criar_analise(cor),
        )

        cor_estavel = (
            resultado.cor_roupa_predominante
        )

        resultados.append(cor_estavel)

        print(
            f"Amostra {indice}: "
            f"leitura={cor:<15} "
            f"resultado={cor_estavel}"
        )

    return resultados


async def main() -> None:
    inconsistentes = await executar(
        "LEITURAS INCONSISTENTES",
        [
            "verde-escura",
            "cinza-escura",
            "marrom",
            "cinza-escura",
        ],
    )

    assert inconsistentes == [
        "indefinida",
        "indefinida",
        "indefinida",
        "indefinida",
    ]

    preta_estavel = await executar(
        "PRETA ESTÁVEL COM RUÍDO POSTERIOR",
        [
            "preta",
            "preta",
            "preta",
            "azul-escura",
            "cinza-escura",
        ],
    )

    assert preta_estavel == [
        "indefinida",
        "indefinida",
        "preta",
        "preta",
        "preta",
    ]

    cinza_estavel = await executar(
        "CINZA COM UMA LEITURA RUIDOSA",
        [
            "cinza",
            "cinza",
            "marrom",
            "cinza",
            "azul",
        ],
    )

    assert cinza_estavel == [
        "indefinida",
        "indefinida",
        "indefinida",
        "cinza",
        "cinza",
    ]

    print(
        "\nTodos os testes passaram."
    )


if __name__ == "__main__":
    asyncio.run(main())
