from __future__ import annotations

from collections import OrderedDict
import logging
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO

from app.domain.models.camera_event import BoundingBox


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Modelo de segmentação: fornece caixas e máscaras.
MODEL_PATH = PROJECT_ROOT / "yolo11n-seg.pt"

_modelo: YOLO | None = None

_model_lock = Lock()
_predict_lock = Lock()
_cache_lock = Lock()

# Guarda resultados recentes para não executar o YOLO
# novamente quando a mesma imagem for analisada.
_CACHE_MAXIMO = 4

_resultados_cache: OrderedDict[
    tuple[
        str,
        int,
        int,
        float,
        float,
        int,
    ],
    Any,
] = OrderedDict()


def _obter_modelo() -> YOLO:
    global _modelo

    if _modelo is not None:
        return _modelo

    with _model_lock:
        if _modelo is None:
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(
                    f"Modelo YOLO não encontrado: "
                    f"{MODEL_PATH}"
                )

            logger.info(
                "Carregando modelo de segmentação: %s",
                MODEL_PATH,
            )

            _modelo = YOLO(
                str(MODEL_PATH)
            )

    return _modelo


def _obter_classes_pessoa(
    modelo: YOLO,
) -> list[int]:
    classes = [
        indice
        for indice, nome in modelo.names.items()
        if str(nome).strip().lower() == "person"
    ]

    if not classes:
        raise RuntimeError(
            "Classe person não encontrada "
            "no modelo YOLO."
        )

    return classes


def _montar_chave_cache(
    caminho: Path,
    confianca_minima: float,
    iou: float,
    tamanho_imagem: int,
) -> tuple[
    str,
    int,
    int,
    float,
    float,
    int,
]:
    estatisticas = caminho.stat()

    return (
        str(caminho.resolve()),
        estatisticas.st_mtime_ns,
        estatisticas.st_size,
        round(confianca_minima, 4),
        round(iou, 4),
        tamanho_imagem,
    )


def _obter_resultado_yolo(
    caminho: Path,
    confianca_minima: float,
    iou: float,
    tamanho_imagem: int,
):
    chave = _montar_chave_cache(
        caminho=caminho,
        confianca_minima=confianca_minima,
        iou=iou,
        tamanho_imagem=tamanho_imagem,
    )

    with _cache_lock:
        resultado_cache = (
            _resultados_cache.get(chave)
        )

        if resultado_cache is not None:
            _resultados_cache.move_to_end(
                chave
            )

            return resultado_cache

    modelo = _obter_modelo()

    classes_pessoa = (
        _obter_classes_pessoa(modelo)
    )

    # Evita vários workers executando o modelo
    # simultaneamente e sobrecarregando a CPU.
    with _predict_lock:
        # Confere novamente porque outra thread pode
        # ter terminado a previsão enquanto aguardávamos.
        with _cache_lock:
            resultado_cache = (
                _resultados_cache.get(chave)
            )

            if resultado_cache is not None:
                _resultados_cache.move_to_end(
                    chave
                )

                return resultado_cache

        resultados = modelo.predict(
            source=str(caminho),
            classes=classes_pessoa,
            conf=confianca_minima,
            iou=iou,
            imgsz=tamanho_imagem,
            device="cpu",
            verbose=False,
        )

    if not resultados:
        return None

    resultado = resultados[0]

    with _cache_lock:
        _resultados_cache[chave] = resultado
        _resultados_cache.move_to_end(chave)

        while (
            len(_resultados_cache)
            > _CACHE_MAXIMO
        ):
            _resultados_cache.popitem(
                last=False
            )

    return resultado


def _limitar_coordenadas(
    x1_float: float,
    y1_float: float,
    x2_float: float,
    y2_float: float,
    largura_imagem: int,
    altura_imagem: int,
) -> tuple[int, int, int, int]:
    x1 = max(
        0,
        min(
            round(x1_float),
            largura_imagem - 1,
        ),
    )

    y1 = max(
        0,
        min(
            round(y1_float),
            altura_imagem - 1,
        ),
    )

    x2 = max(
        x1 + 1,
        min(
            round(x2_float),
            largura_imagem,
        ),
    )

    y2 = max(
        y1 + 1,
        min(
            round(y2_float),
            altura_imagem,
        ),
    )

    return x1, y1, x2, y2


def detectar_pessoas_yolo(
    caminho_imagem: str | Path,
    confianca_minima: float = 0.50,
    iou: float = 0.30,
    tamanho_imagem: int = 960,
) -> list[BoundingBox]:
    caminho = Path(caminho_imagem)

    if not caminho.is_file():
        raise FileNotFoundError(
            f"Imagem não encontrada: {caminho}"
        )

    resultado = _obter_resultado_yolo(
        caminho=caminho,
        confianca_minima=confianca_minima,
        iou=iou,
        tamanho_imagem=tamanho_imagem,
    )

    if resultado is None:
        return []

    caixas_resultado = resultado.boxes

    if (
        caixas_resultado is None
        or len(caixas_resultado) == 0
    ):
        return []

    altura_imagem, largura_imagem = (
        resultado.orig_shape
    )

    area_imagem = (
        largura_imagem
        * altura_imagem
    )

    caixas: list[BoundingBox] = []

    for caixa in caixas_resultado:
        (
            x1_float,
            y1_float,
            x2_float,
            y2_float,
        ) = caixa.xyxy[0].tolist()

        x1, y1, x2, y2 = (
            _limitar_coordenadas(
                x1_float=x1_float,
                y1_float=y1_float,
                x2_float=x2_float,
                y2_float=y2_float,
                largura_imagem=largura_imagem,
                altura_imagem=altura_imagem,
            )
        )

        largura = x2 - x1
        altura = y2 - y1
        area = largura * altura

        percentual = (
            area
            / area_imagem
            * 100
        )

        confianca = float(
            caixa.conf[0].item()
        )

        logger.info(
            "YOLO pessoa: "
            "conf=%.4f "
            "bbox=%s,%s-%s,%s "
            "area=%.3f%%",
            confianca,
            x1,
            y1,
            x2,
            y2,
            percentual,
        )

        if (
            percentual < 0.05
            or percentual >= 60
        ):
            continue

        caixas.append(
            BoundingBox(
                origem="yolo_person_seg",
                x=x1,
                y=y1,
                largura=largura,
                altura=altura,
                x2=x2,
                y2=y2,
                proporcao_imagem=(
                    area / area_imagem
                ),
            )
        )

    caixas.sort(
        key=lambda caixa: (
            caixa.x
            + caixa.largura / 2,
            caixa.y
            + caixa.altura / 2,
        )
    )

    return caixas


def _calcular_sobreposicao(
    referencia: tuple[
        int,
        int,
        int,
        int,
    ],
    detectada: tuple[
        int,
        int,
        int,
        int,
    ],
) -> tuple[float, float, float]:
    ref_x1, ref_y1, ref_x2, ref_y2 = (
        referencia
    )

    det_x1, det_y1, det_x2, det_y2 = (
        detectada
    )

    inter_x1 = max(ref_x1, det_x1)
    inter_y1 = max(ref_y1, det_y1)
    inter_x2 = min(ref_x2, det_x2)
    inter_y2 = min(ref_y2, det_y2)

    largura_intersecao = max(
        0,
        inter_x2 - inter_x1,
    )

    altura_intersecao = max(
        0,
        inter_y2 - inter_y1,
    )

    area_intersecao = (
        largura_intersecao
        * altura_intersecao
    )

    area_referencia = max(
        0,
        ref_x2 - ref_x1,
    ) * max(
        0,
        ref_y2 - ref_y1,
    )

    area_detectada = max(
        0,
        det_x2 - det_x1,
    ) * max(
        0,
        det_y2 - det_y1,
    )

    if (
        area_referencia <= 0
        or area_detectada <= 0
        or area_intersecao <= 0
    ):
        return 0.0, 0.0, 0.0

    area_uniao = (
        area_referencia
        + area_detectada
        - area_intersecao
    )

    valor_iou = (
        area_intersecao / area_uniao
        if area_uniao > 0
        else 0.0
    )

    cobertura_referencia = (
        area_intersecao
        / area_referencia
    )

    cobertura_detectada = (
        area_intersecao
        / area_detectada
    )

    return (
        valor_iou,
        cobertura_referencia,
        cobertura_detectada,
    )


def obter_rgb_roupa_segmentada(
    caminho_imagem: str | Path,
    x: int,
    y: int,
    largura: int,
    altura: int,
    confianca_minima: float = 0.35,
    iou: float = 0.30,
    tamanho_imagem: int = 960,
) -> tuple[int, int, int] | None:
    """
    Obtém a cor da roupa usando a máscara da pessoa.

    A bounding box recebida é usada para escolher a
    pessoa correspondente quando existem várias
    pessoas na mesma imagem.

    Retorna None quando a segmentação não encontra
    uma pessoa correspondente. Nesse caso, o
    scene_analyzer utiliza o método antigo.
    """

    caminho = Path(caminho_imagem)

    if not caminho.is_file():
        raise FileNotFoundError(
            f"Imagem não encontrada: {caminho}"
        )

    if largura <= 0 or altura <= 0:
        return None

    resultado = _obter_resultado_yolo(
        caminho=caminho,
        confianca_minima=confianca_minima,
        iou=iou,
        tamanho_imagem=tamanho_imagem,
    )

    if (
        resultado is None
        or resultado.boxes is None
        or resultado.masks is None
        or len(resultado.boxes) == 0
    ):
        return None

    altura_imagem, largura_imagem = (
        resultado.orig_shape
    )

    referencia_x1 = max(
        0,
        min(
            int(x),
            largura_imagem - 1,
        ),
    )

    referencia_y1 = max(
        0,
        min(
            int(y),
            altura_imagem - 1,
        ),
    )

    referencia_x2 = max(
        referencia_x1 + 1,
        min(
            int(x + largura),
            largura_imagem,
        ),
    )

    referencia_y2 = max(
        referencia_y1 + 1,
        min(
            int(y + altura),
            altura_imagem,
        ),
    )

    referencia = (
        referencia_x1,
        referencia_y1,
        referencia_x2,
        referencia_y2,
    )

    caixas_numpy = (
        resultado.boxes.xyxy
        .cpu()
        .numpy()
    )

    confiancas = (
        resultado.boxes.conf
        .cpu()
        .numpy()
    )

    melhor_indice: int | None = None
    melhor_caixa: tuple[
        int,
        int,
        int,
        int,
    ] | None = None

    melhor_pontuacao = (
        -1.0,
        -1.0,
        -1.0,
        -1.0,
    )

    for indice, caixa_numpy in enumerate(
        caixas_numpy
    ):
        detectada = _limitar_coordenadas(
            x1_float=float(caixa_numpy[0]),
            y1_float=float(caixa_numpy[1]),
            x2_float=float(caixa_numpy[2]),
            y2_float=float(caixa_numpy[3]),
            largura_imagem=largura_imagem,
            altura_imagem=altura_imagem,
        )

        (
            valor_iou,
            cobertura_referencia,
            cobertura_detectada,
        ) = _calcular_sobreposicao(
            referencia=referencia,
            detectada=detectada,
        )

        corresponde = (
            valor_iou >= 0.15
            or cobertura_referencia >= 0.45
            or cobertura_detectada >= 0.45
        )

        if not corresponde:
            continue

        pontuacao = (
            max(
                valor_iou,
                cobertura_referencia,
                cobertura_detectada,
            ),
            cobertura_referencia,
            valor_iou,
            float(confiancas[indice]),
        )

        if pontuacao > melhor_pontuacao:
            melhor_pontuacao = pontuacao
            melhor_indice = indice
            melhor_caixa = detectada

    if (
        melhor_indice is None
        or melhor_caixa is None
    ):
        return None

    mascara = (
        resultado.masks.data[
            melhor_indice
        ]
        .cpu()
        .numpy()
    )

    if mascara.shape != (
        altura_imagem,
        largura_imagem,
    ):
        mascara_imagem = Image.fromarray(
            (
                mascara * 255
            ).astype(np.uint8)
        )

        mascara_imagem = mascara_imagem.resize(
            (
                largura_imagem,
                altura_imagem,
            ),
            Image.Resampling.NEAREST,
        )

        mascara = (
            np.asarray(mascara_imagem)
            > 127
        )
    else:
        mascara = mascara > 0.5

    (
        pessoa_x1,
        pessoa_y1,
        pessoa_x2,
        pessoa_y2,
    ) = melhor_caixa

    largura_pessoa = (
        pessoa_x2 - pessoa_x1
    )

    altura_pessoa = (
        pessoa_y2 - pessoa_y1
    )

    # Mesma região que funcionou no teste real:
    # parte central e inferior da camisa.
    torso_x1 = int(
        pessoa_x1
        + largura_pessoa * 0.18
    )

    torso_x2 = int(
        pessoa_x1
        + largura_pessoa * 0.82
    )

    torso_y1 = int(
        pessoa_y1
        + altura_pessoa * 0.42
    )

    torso_y2 = int(
        pessoa_y1
        + altura_pessoa * 0.72
    )

    torso_x1 = max(
        0,
        min(
            torso_x1,
            largura_imagem - 1,
        ),
    )

    torso_y1 = max(
        0,
        min(
            torso_y1,
            altura_imagem - 1,
        ),
    )

    torso_x2 = max(
        torso_x1 + 1,
        min(
            torso_x2,
            largura_imagem,
        ),
    )

    torso_y2 = max(
        torso_y1 + 1,
        min(
            torso_y2,
            altura_imagem,
        ),
    )

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

    with Image.open(caminho) as imagem_aberta:
        imagem_rgb = np.asarray(
            imagem_aberta.convert("RGB")
        )

    pixels = imagem_rgb[
        regiao_valida
    ]

    if len(pixels) < 50:
        return None

    pixels_float = pixels.astype(
        np.float32
    )

    luminancia = (
        pixels_float[:, 0] * 0.2126
        + pixels_float[:, 1] * 0.7152
        + pixels_float[:, 2] * 0.0722
    )

    pixels = pixels[
        (luminancia >= 20)
        & (luminancia <= 235)
    ]

    if len(pixels) < 50:
        return None

    rgb = tuple(
        int(round(valor))
        for valor in np.median(
            pixels,
            axis=0,
        )
    )

    logger.debug(
        "Cor segmentada: imagem=%s rgb=%s "
        "pixels=%s",
        caminho,
        rgb,
        len(pixels),
    )

    return rgb


def _area(
    caixa: BoundingBox,
) -> int:
    return max(
        0,
        caixa.largura,
    ) * max(
        0,
        caixa.altura,
    )


def _area_intersecao(
    primeira: BoundingBox,
    segunda: BoundingBox,
) -> int:
    x1 = max(
        primeira.x,
        segunda.x,
    )

    y1 = max(
        primeira.y,
        segunda.y,
    )

    x2 = min(
        primeira.x2,
        segunda.x2,
    )

    y2 = min(
        primeira.y2,
        segunda.y2,
    )

    largura = max(
        0,
        x2 - x1,
    )

    altura = max(
        0,
        y2 - y1,
    )

    return largura * altura


def _representam_mesma_pessoa(
    primeira: BoundingBox,
    segunda: BoundingBox,
) -> bool:
    area_primeira = _area(
        primeira
    )

    area_segunda = _area(
        segunda
    )

    if (
        area_primeira <= 0
        or area_segunda <= 0
    ):
        return False

    intersecao = _area_intersecao(
        primeira,
        segunda,
    )

    if intersecao <= 0:
        return False

    uniao = (
        area_primeira
        + area_segunda
        - intersecao
    )

    valor_iou = (
        intersecao / uniao
        if uniao > 0
        else 0
    )

    cobertura_menor = (
        intersecao
        / min(
            area_primeira,
            area_segunda,
        )
    )

    return (
        valor_iou >= 0.30
        or cobertura_menor >= 0.60
    )


def combinar_bounding_boxes(
    caixas_camera: list[BoundingBox],
    caixas_yolo: list[BoundingBox],
) -> list[BoundingBox]:
    if not caixas_yolo:
        return list(
            caixas_camera
        )

    resultado = list(
        caixas_yolo
    )

    for caixa_camera in caixas_camera:
        duplicada = any(
            _representam_mesma_pessoa(
                caixa_camera,
                caixa_yolo,
            )
            for caixa_yolo in caixas_yolo
        )

        if not duplicada:
            resultado.append(
                caixa_camera
            )

    resultado.sort(
        key=lambda caixa: (
            caixa.x
            + caixa.largura / 2,
            caixa.y
            + caixa.altura / 2,
        )
    )

    return resultado
