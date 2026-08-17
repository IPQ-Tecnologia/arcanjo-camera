import asyncio

from app.services.appearance_memory import (
    AppearanceMemory,
)
from app.services.scene_analyzer import (
    PersonVisualAnalysis,
)


def criar_analise(
    cor: str,
    rgb: tuple[int, int, int],
    posicao: str = "centro",
) -> PersonVisualAnalysis:
    return PersonVisualAnalysis(
        cor_roupa_aproximada=cor,
        rgb_representativo=rgb,
        posicao_horizontal=posicao,
        tamanho_no_quadro="medio",
        percentual_quadro=8.0,
        descricao="Teste da memória",
    )


async def main() -> None:
    memoria = AppearanceMemory()

    amostras = [
        ("preta", (25, 25, 25)),
        ("cinza", (85, 85, 85)),
        ("preta", (28, 28, 28)),
        ("preta", (24, 24, 24)),
        ("cinza", (75, 75, 75)),
    ]

    resultados_esperados = [
        "indefinida",
        "indefinida",
        "indefinida",
        "preta",
        "preta",
    ]

    print("===== ESTABILIZAÇÃO DA COR =====")

    for indice, (
        (cor, rgb),
        esperado,
    ) in enumerate(
        zip(
            amostras,
            resultados_esperados,
        ),
        start=1,
    ):
        resultado = await memoria.registrar(
            pessoa_id="pessoa-1",
            analise=criar_analise(
                cor=cor,
                rgb=rgb,
                posicao=(
                    "direita"
                    if indice == len(amostras)
                    else "centro"
                ),
            ),
        )

        print(
            f"Amostra {indice}: "
            f"leitura={cor:<8} "
            f"resultado="
            f"{resultado.cor_roupa_predominante}"
        )

        assert (
            resultado.cor_roupa_predominante
            == esperado
        )

    resultado_obtido = await memoria.obter(
        "pessoa-1"
    )

    assert resultado_obtido is not None
    assert (
        resultado_obtido.cor_roupa_predominante
        == "preta"
    )
    assert (
        resultado_obtido.quantidade_amostras
        == 5
    )
    assert (
        resultado_obtido.posicao_atual
        == "direita"
    )

    resultado_final = await memoria.finalizar(
        "pessoa-1"
    )

    assert resultado_final is not None
    assert (
        resultado_final.cor_roupa_predominante
        == "preta"
    )

    assert (
        await memoria.obter("pessoa-1")
        is None
    )

    assert memoria.quantidade_sessoes == 0

    await memoria.registrar(
        pessoa_id="pessoa-2",
        analise=criar_analise(
            cor="cinza",
            rgb=(80, 80, 80),
        ),
    )

    assert memoria.quantidade_sessoes == 1

    await memoria.limpar()

    assert memoria.quantidade_sessoes == 0

    print(
        "\nTeste da memória concluído "
        "com sucesso."
    )


if __name__ == "__main__":
    asyncio.run(main())
