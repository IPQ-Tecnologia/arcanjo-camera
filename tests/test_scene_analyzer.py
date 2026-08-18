import json
from pathlib import Path

from PIL import Image, ImageDraw

from app.services.scene_analyzer import (
    analisar_pessoa,
)


def criar_imagem_teste(
    caminho: Path,
) -> None:
    imagem = Image.new(
        mode="RGB",
        size=(640, 480),
        color=(235, 235, 235),
    )

    desenho = ImageDraw.Draw(imagem)

    # Cabeça
    desenho.ellipse(
        (290, 70, 350, 130),
        fill=(190, 145, 110),
    )

    # Camisa preta
    desenho.rectangle(
        (265, 125, 375, 300),
        fill=(20, 20, 20),
    )

    # Calça
    desenho.rectangle(
        (280, 300, 360, 430),
        fill=(45, 65, 100),
    )

    caminho.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    imagem.save(
        caminho,
        format="JPEG",
        quality=95,
    )


def main() -> None:
    caminho = Path(
        "imagens_eventos/"
        "teste_scene_analyzer.jpg"
    )

    criar_imagem_teste(caminho)

    resultado = analisar_pessoa(
        caminho_imagem=caminho,
        x=250,
        y=60,
        largura=140,
        altura=380,
    )

    print("===== ANÁLISE VISUAL =====")

    print(
        json.dumps(
            resultado.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
    )

    assert (
        resultado.cor_roupa_aproximada
        == "preta"
    )

    assert (
        resultado.posicao_horizontal
        == "centro"
    )

    print(
        "\nTeste concluído com sucesso."
    )


if __name__ == "__main__":
    main()