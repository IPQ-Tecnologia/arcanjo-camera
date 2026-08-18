from app.services.scene_analyzer import (
    classificar_cor_rgb,
)


CASOS = {
    "preto puro": (
        (18, 18, 18),
        "preta",
    ),
    "preto com luz quente": (
        (55, 45, 35),
        "preta",
    ),
    "azul escuro": (
        (25, 42, 78),
        "azul-escura",
    ),
    "azul": (
        (50, 100, 190),
        "azul",
    ),
    "cinza escuro": (
        (90, 90, 90),
        "cinza-escura",
    ),
    "cinza": (
        (160, 160, 160),
        "cinza",
    ),
    "branco": (
        (235, 235, 235),
        "branca",
    ),
    "marrom": (
        (110, 70, 42),
        "marrom",
    ),
    "vermelho": (
        (190, 45, 45),
        "vermelha",
    ),
    "verde": (
        (55, 145, 70),
        "verde",
    ),
    "amarelo": (
        (220, 200, 45),
        "amarela",
    ),
}


def executar_teste() -> None:
    print(
        "===== CLASSIFICAÇÃO DE CORES ====="
    )

    for nome, (
        rgb,
        esperado,
    ) in CASOS.items():
        resultado = classificar_cor_rgb(
            rgb
        )

        print(
            f"{nome}: RGB={rgb} "
            f"resultado={resultado}"
        )

        assert resultado == esperado, (
            f"{nome}: esperado={esperado}, "
            f"recebido={resultado}"
        )

    print(
        "\nTeste concluído com sucesso."
    )


if __name__ == "__main__":
    executar_teste()
