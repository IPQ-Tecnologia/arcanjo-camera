import colorsys
from pathlib import Path

from PIL import Image

from app.services.person_detector_yolo import (
    obter_rgb_roupa_segmentada,
)
from app.services.scene_analyzer import (
    _calcular_luminancia,
    _obter_cor_representativa,
    _recortar_regiao_roupa,
    classificar_cor_rgb,
)


AMOSTRAS = [
    (
        "1a4e7d6d0087",
        Path(
            "imagens_eventos/"
            "20260803T204454Z_linedetection_"
            "1a4e7d6d0087_original.jpg"
        ),
        579,
        72,
        108,
        320,
    ),
    (
        "7c3ba59c9021",
        Path(
            "imagens_eventos/"
            "20260803T204504Z_fielddetection_"
            "7c3ba59c9021_original.jpg"
        ),
        284,
        103,
        105,
        317,
    ),
]


def mostrar_metricas(
    nome: str,
    rgb: tuple[int, int, int] | None,
) -> None:
    if rgb is None:
        print(f"{nome}: não encontrado")
        return

    vermelho, verde, azul = rgb

    matiz, saturacao, brilho = colorsys.rgb_to_hsv(
        vermelho / 255,
        verde / 255,
        azul / 255,
    )

    print(f"{nome}:")
    print("  RGB:", rgb)
    print("  Cor:", classificar_cor_rgb(rgb))
    print("  Matiz:", round(matiz * 360, 2))
    print("  Saturação:", round(saturacao, 3))
    print("  Brilho:", round(brilho, 3))
    print(
        "  Luminância:",
        round(_calcular_luminancia(rgb), 2),
    )
    print("  Azul - vermelho:", azul - vermelho)
    print("  Azul - verde:", azul - verde)
    print(
        "  Diferença máxima:",
        max(rgb) - min(rgb),
    )


for evento_id, caminho, x, y, largura, altura in AMOSTRAS:
    print("\n" + "=" * 55)
    print("Evento:", evento_id)
    print("Imagem:", caminho)

    rgb_segmentado = obter_rgb_roupa_segmentada(
        caminho_imagem=caminho,
        x=x,
        y=y,
        largura=largura,
        altura=altura,
    )

    mostrar_metricas(
        "Segmentado",
        rgb_segmentado,
    )

    with Image.open(caminho) as imagem_aberta:
        imagem = imagem_aberta.convert("RGB")

        regiao = _recortar_regiao_roupa(
            imagem=imagem,
            x1=x,
            y1=y,
            largura=largura,
            altura=altura,
        )

        rgb_fallback = _obter_cor_representativa(
            regiao
        )

    mostrar_metricas(
        "Fallback",
        rgb_fallback,
    )
