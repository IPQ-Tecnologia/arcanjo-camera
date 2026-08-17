from pathlib import Path

import colorsys
import numpy as np
from PIL import Image

from app.services.person_detector_yolo import (
    obter_rgb_roupa_segmentada,
)


AMOSTRAS = [
    # Camisa azul-escura verdadeira
    (
        "AZUL",
        "c7cd54797fdd",
        Path(
            "imagens_eventos/"
            "20260803T203334Z_fielddetection_"
            "c7cd54797fdd_original.jpg"
        ),
        241, 157, 182, 272,
    ),
    (
        "AZUL",
        "81e03fc98b1a",
        Path(
            "imagens_eventos/"
            "20260803T203345Z_linedetection_"
            "81e03fc98b1a_original.jpg"
        ),
        206, 125, 102, 193,
    ),
    (
        "AZUL",
        "ea9b125af585",
        Path(
            "imagens_eventos/"
            "20260803T203345Z_linedetection_"
            "ea9b125af585_original.jpg"
        ),
        312, 145, 109, 267,
    ),
    (
        "AZUL",
        "90b8a7763ebb",
        Path(
            "imagens_eventos/"
            "20260803T203359Z_linedetection_"
            "90b8a7763ebb_original.jpg"
        ),
        170, 117, 122, 285,
    ),

    # Camisa preta verdadeira
    (
        "PRETA",
        "1a4e7d6d0087",
        Path(
            "imagens_eventos/"
            "20260803T204454Z_linedetection_"
            "1a4e7d6d0087_original.jpg"
        ),
        579, 72, 108, 320,
    ),
    (
        "PRETA",
        "7c3ba59c9021",
        Path(
            "imagens_eventos/"
            "20260803T204504Z_fielddetection_"
            "7c3ba59c9021_original.jpg"
        ),
        284, 103, 105, 317,
    ),
]


def estimar_balanco(
    caminho: Path,
) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(caminho) as imagem:
        pixels = np.asarray(
            imagem.convert("RGB"),
            dtype=np.float32,
        ).reshape(-1, 3)

    maximo = pixels.max(axis=1)
    minimo = pixels.min(axis=1)

    luminancia = (
        pixels[:, 0] * 0.2126
        + pixels[:, 1] * 0.7152
        + pixels[:, 2] * 0.0722
    )

    saturacao = (
        (maximo - minimo)
        / np.maximum(maximo, 1)
    )

    # Retira pixels muito escuros e muito claros.
    validos = (
        (luminancia >= 45)
        & (luminancia <= 220)
    )

    pixels_validos = pixels[validos]
    saturacao_valida = saturacao[validos]

    if len(pixels_validos) < 100:
        return (
            np.array([1.0, 1.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
        )

    # Usa os pixels menos saturados da cena como
    # referência de cinza/branco neutro.
    limite_saturacao = np.quantile(
        saturacao_valida,
        0.20,
    )

    neutros = pixels_validos[
        saturacao_valida <= limite_saturacao
    ]

    referencia = np.median(
        neutros,
        axis=0,
    )

    alvo = float(
        np.mean(referencia)
    )

    ganhos = (
        alvo
        / np.maximum(referencia, 1)
    )

    ganhos = np.clip(
        ganhos,
        0.70,
        1.35,
    )

    return ganhos, referencia


def corrigir_rgb(
    rgb: tuple[int, int, int],
    ganhos: np.ndarray,
) -> tuple[int, int, int]:
    corrigido = np.clip(
        np.asarray(
            rgb,
            dtype=np.float32,
        ) * ganhos,
        0,
        255,
    )

    return tuple(
        int(round(valor))
        for valor in corrigido
    )


def mostrar_metricas(
    rgb: tuple[int, int, int],
) -> None:
    vermelho, verde, azul = rgb

    matiz, saturacao, brilho = (
        colorsys.rgb_to_hsv(
            vermelho / 255,
            verde / 255,
            azul / 255,
        )
    )

    excesso_azul = (
        azul
        - (
            vermelho + verde
        ) / 2
    )

    print("  HSV:")
    print(
        "    matiz=",
        round(matiz * 360, 2),
    )
    print(
        "    saturação=",
        round(saturacao, 3),
    )
    print(
        "    brilho=",
        round(brilho, 3),
    )
    print(
        "  Excesso azul:",
        round(excesso_azul, 2),
    )


for (
    cor_real,
    evento_id,
    caminho,
    x,
    y,
    largura,
    altura,
) in AMOSTRAS:
    print("\n" + "=" * 60)
    print("Cor real:", cor_real)
    print("Evento:", evento_id)

    if not caminho.is_file():
        print("Imagem não encontrada:", caminho)
        continue

    rgb_original = obter_rgb_roupa_segmentada(
        caminho_imagem=caminho,
        x=x,
        y=y,
        largura=largura,
        altura=altura,
    )

    if rgb_original is None:
        print("Segmentação não encontrada")
        continue

    ganhos, referencia = estimar_balanco(
        caminho
    )

    rgb_corrigido = corrigir_rgb(
        rgb_original,
        ganhos,
    )

    print(
        "Referência neutra:",
        tuple(
            round(float(valor), 2)
            for valor in referencia
        ),
    )

    print(
        "Ganhos:",
        tuple(
            round(float(valor), 3)
            for valor in ganhos
        ),
    )

    print("RGB original:", rgb_original)
    mostrar_metricas(rgb_original)

    print("RGB corrigido:", rgb_corrigido)
    mostrar_metricas(rgb_corrigido)
