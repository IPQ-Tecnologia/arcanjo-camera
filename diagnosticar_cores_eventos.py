from pathlib import Path

from PIL import Image

from app.services.person_detector_yolo import (
    obter_rgb_roupa_segmentada,
)
from app.services.scene_analyzer import (
    _obter_cor_representativa,
    _recortar_regiao_roupa,
    classificar_cor_rgb,
)


AMOSTRAS = [
    (
        "c7cd54797fdd",
        Path(
            "imagens_eventos/"
            "20260803T203334Z_fielddetection_"
            "c7cd54797fdd_original.jpg"
        ),
        241,
        157,
        182,
        272,
    ),
    (
        "81e03fc98b1a",
        Path(
            "imagens_eventos/"
            "20260803T203345Z_linedetection_"
            "81e03fc98b1a_original.jpg"
        ),
        206,
        125,
        102,
        193,
    ),
    (
        "ea9b125af585",
        Path(
            "imagens_eventos/"
            "20260803T203345Z_linedetection_"
            "ea9b125af585_original.jpg"
        ),
        312,
        145,
        109,
        267,
    ),
    (
        "90b8a7763ebb",
        Path(
            "imagens_eventos/"
            "20260803T203359Z_linedetection_"
            "90b8a7763ebb_original.jpg"
        ),
        170,
        117,
        122,
        285,
    ),
]


for (
    evento_id,
    caminho,
    x,
    y,
    largura,
    altura,
) in AMOSTRAS:
    print("\n" + "=" * 60)
    print("Evento:", evento_id)
    print("Imagem:", caminho)

    if not caminho.is_file():
        print("ERRO: imagem não encontrada")
        continue

    rgb_segmentado = obter_rgb_roupa_segmentada(
        caminho_imagem=caminho,
        x=x,
        y=y,
        largura=largura,
        altura=altura,
    )

    if rgb_segmentado is None:
        print("Segmentação: FALHOU")
    else:
        print(
            "RGB segmentado:",
            rgb_segmentado,
        )
        print(
            "Cor segmentada:",
            classificar_cor_rgb(
                rgb_segmentado
            ),
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
            min(x + largura, largura_imagem),
        )
        y2 = max(
            y1 + 1,
            min(y + altura, altura_imagem),
        )

        regiao_fallback = (
            _recortar_regiao_roupa(
                imagem=imagem,
                x1=x1,
                y1=y1,
                largura=x2 - x1,
                altura=y2 - y1,
            )
        )

        rgb_fallback = (
            _obter_cor_representativa(
                regiao_fallback
            )
        )

    print(
        "RGB fallback:",
        rgb_fallback,
    )
    print(
        "Cor fallback:",
        classificar_cor_rgb(
            rgb_fallback
        ),
    )
