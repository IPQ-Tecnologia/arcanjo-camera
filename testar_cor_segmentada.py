from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from app.services.scene_analyzer import classificar_cor_rgb


IMAGEM = Path("amostras/teste_original.jpg")
MODELO = Path("yolo11n-seg.pt")
SAIDA = Path("amostras/torso_segmentado.jpg")


def main() -> None:
    imagem_bgr = cv2.imread(str(IMAGEM))

    if imagem_bgr is None:
        raise FileNotFoundError(
            f"Não foi possível abrir: {IMAGEM}"
        )

    imagem_rgb = cv2.cvtColor(
        imagem_bgr,
        cv2.COLOR_BGR2RGB,
    )

    altura_imagem, largura_imagem = imagem_rgb.shape[:2]

    modelo = YOLO(str(MODELO))

    resultado = modelo.predict(
        source=str(IMAGEM),
        classes=[0],
        conf=0.35,
        iou=0.30,
        imgsz=960,
        device="cpu",
        verbose=False,
    )[0]

    if (
        resultado.boxes is None
        or resultado.masks is None
        or len(resultado.boxes) == 0
    ):
        raise RuntimeError(
            "Nenhuma pessoa segmentada."
        )

    confiancas = (
        resultado.boxes.conf
        .cpu()
        .numpy()
    )

    indice = int(np.argmax(confiancas))

    x1, y1, x2, y2 = [
        int(round(valor))
        for valor in (
            resultado.boxes.xyxy[indice]
            .cpu()
            .tolist()
        )
    ]

    x1 = max(0, min(x1, largura_imagem - 1))
    y1 = max(0, min(y1, altura_imagem - 1))
    x2 = max(x1 + 1, min(x2, largura_imagem))
    y2 = max(y1 + 1, min(y2, altura_imagem))

    largura_pessoa = x2 - x1
    altura_pessoa = y2 - y1

    # Região aproximada da camisa.
    torso_x1 = int(x1 + largura_pessoa * 0.18)
    torso_x2 = int(x1 + largura_pessoa * 0.82)

    # Mais abaixo que o teste anterior, para evitar
    # braços levantados, cabeça e rosto.
    torso_y1 = int(y1 + altura_pessoa * 0.42)
    torso_y2 = int(y1 + altura_pessoa * 0.72)

    mascara = (
        resultado.masks.data[indice]
        .cpu()
        .numpy()
    )

    if mascara.shape != (
        altura_imagem,
        largura_imagem,
    ):
        mascara = cv2.resize(
            mascara,
            (
                largura_imagem,
                altura_imagem,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

    mascara = mascara > 0.5

    regiao_valida = np.zeros(
        (
            altura_imagem,
            largura_imagem,
        ),
        dtype=bool,
    )

    regiao_valida[
        torso_y1:torso_y2,
        torso_x1:torso_x2,
    ] = True

    regiao_valida &= mascara

    pixels = imagem_rgb[regiao_valida]

    if len(pixels) < 50:
        raise RuntimeError(
            "Poucos pixels de roupa encontrados."
        )

    luminancia = (
        pixels[:, 0] * 0.2126
        + pixels[:, 1] * 0.7152
        + pixels[:, 2] * 0.0722
    )

    # Remove sombras quase pretas e reflexos claros.
    pixels = pixels[
        (luminancia >= 20)
        & (luminancia <= 235)
    ]

    if len(pixels) < 50:
        raise RuntimeError(
            "Poucos pixels após a filtragem."
        )

    rgb = tuple(
        int(round(valor))
        for valor in np.median(
            pixels,
            axis=0,
        )
    )

    cor = classificar_cor_rgb(rgb)

    recorte = imagem_rgb[
        torso_y1:torso_y2,
        torso_x1:torso_x2,
    ].copy()

    mascara_recorte = regiao_valida[
        torso_y1:torso_y2,
        torso_x1:torso_x2,
    ]

    recorte[~mascara_recorte] = 0

    cv2.imwrite(
        str(SAIDA),
        cv2.cvtColor(
            recorte,
            cv2.COLOR_RGB2BGR,
        ),
    )

    print(
        "Confiança da pessoa:",
        round(float(confiancas[indice]), 2),
    )
    print("Pixels analisados:", len(pixels))
    print("RGB segmentado:", rgb)
    print("Cor identificada:", cor)
    print("Recorte salvo em:", SAIDA)


if __name__ == "__main__":
    main()
