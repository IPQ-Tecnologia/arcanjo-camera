from __future__ import annotations

import asyncio
import math
import time
from dataclasses import asdict, dataclass
from typing import Literal

from app.services.person_tracker import DetectionBox


MovimentoHorizontal = Literal[
    "inicial",
    "parado",
    "esquerda",
    "direita",
]

MovimentoVertical = Literal[
    "inicial",
    "parado",
    "cima",
    "baixo",
]

TendenciaDistancia = Literal[
    "inicial",
    "estavel",
    "aproximando",
    "afastando",
]


@dataclass(frozen=True)
class PersonMovementAnalysis:
    pessoa_id: str

    movimento_horizontal: MovimentoHorizontal
    movimento_vertical: MovimentoVertical
    tendencia_distancia: TendenciaDistancia

    deslocamento_x: float
    deslocamento_y: float
    distancia_pixels: float

    velocidade_pixels_segundo: float
    variacao_area_percentual: float

    distancia_total_pixels: float
    tempo_observado_segundos: float
    quantidade_amostras: int

    descricao: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _MovementState:
    pessoa_id: str

    primeira_amostra_em: float
    ultima_amostra_em: float

    ultima_bbox: DetectionBox

    quantidade_amostras: int
    distancia_total_pixels: float

    ultima_analise: PersonMovementAnalysis


class PersonMovementMemory:
    def __init__(
        self,
        limite_movimento_percentual: float = 0.05,
        limite_movimento_minimo_pixels: float = 6.0,
        limite_variacao_area_percentual: float = 12.0,
    ) -> None:
        if limite_movimento_percentual < 0:
            raise ValueError(
                "limite_movimento_percentual "
                "não pode ser negativo"
            )

        if limite_movimento_minimo_pixels < 0:
            raise ValueError(
                "limite_movimento_minimo_pixels "
                "não pode ser negativo"
            )

        if limite_variacao_area_percentual < 0:
            raise ValueError(
                "limite_variacao_area_percentual "
                "não pode ser negativo"
            )

        self.limite_movimento_percentual = (
            limite_movimento_percentual
        )

        self.limite_movimento_minimo_pixels = (
            limite_movimento_minimo_pixels
        )

        self.limite_variacao_area_percentual = (
            limite_variacao_area_percentual
        )

        self._estados: dict[
            str,
            _MovementState,
        ] = {}

        self._lock = asyncio.Lock()

    async def registrar(
        self,
        pessoa_id: str,
        bbox: DetectionBox,
        agora: float | None = None,
    ) -> PersonMovementAnalysis:
        if not pessoa_id.strip():
            raise ValueError(
                "pessoa_id não pode ser vazio"
            )

        if (
            bbox.largura <= 0
            or bbox.altura <= 0
        ):
            raise ValueError(
                "A bounding box deve possuir "
                "largura e altura positivas"
            )

        momento = (
            time.monotonic()
            if agora is None
            else agora
        )

        async with self._lock:
            estado = self._estados.get(
                pessoa_id
            )

            if estado is None:
                analise = self._criar_primeira_analise(
                    pessoa_id=pessoa_id,
                )

                self._estados[pessoa_id] = (
                    _MovementState(
                        pessoa_id=pessoa_id,
                        primeira_amostra_em=momento,
                        ultima_amostra_em=momento,
                        ultima_bbox=bbox,
                        quantidade_amostras=1,
                        distancia_total_pixels=0.0,
                        ultima_analise=analise,
                    )
                )

                return analise

            analise = self._calcular_movimento(
                estado=estado,
                bbox=bbox,
                agora=momento,
            )

            estado.ultima_amostra_em = momento
            estado.ultima_bbox = bbox
            estado.quantidade_amostras += 1

            estado.distancia_total_pixels = (
                analise.distancia_total_pixels
            )

            estado.ultima_analise = analise

            return analise

    async def obter(
        self,
        pessoa_id: str,
    ) -> PersonMovementAnalysis | None:
        async with self._lock:
            estado = self._estados.get(
                pessoa_id
            )

            if estado is None:
                return None

            return estado.ultima_analise

    async def finalizar(
        self,
        pessoa_id: str,
    ) -> PersonMovementAnalysis | None:
        async with self._lock:
            estado = self._estados.pop(
                pessoa_id,
                None,
            )

            if estado is None:
                return None

            return estado.ultima_analise

    async def limpar(self) -> None:
        async with self._lock:
            self._estados.clear()

    @property
    def quantidade_pessoas(self) -> int:
        return len(self._estados)

    def _criar_primeira_analise(
        self,
        pessoa_id: str,
    ) -> PersonMovementAnalysis:
        return PersonMovementAnalysis(
            pessoa_id=pessoa_id,
            movimento_horizontal="inicial",
            movimento_vertical="inicial",
            tendencia_distancia="inicial",
            deslocamento_x=0.0,
            deslocamento_y=0.0,
            distancia_pixels=0.0,
            velocidade_pixels_segundo=0.0,
            variacao_area_percentual=0.0,
            distancia_total_pixels=0.0,
            tempo_observado_segundos=0.0,
            quantidade_amostras=1,
            descricao=(
                "Primeira posição da pessoa "
                "registrada."
            ),
        )

    def _calcular_movimento(
        self,
        estado: _MovementState,
        bbox: DetectionBox,
        agora: float,
    ) -> PersonMovementAnalysis:
        bbox_anterior = estado.ultima_bbox

        deslocamento_x = (
            bbox.centro_x
            - bbox_anterior.centro_x
        )

        deslocamento_y = (
            bbox.centro_y
            - bbox_anterior.centro_y
        )

        distancia_pixels = math.hypot(
            deslocamento_x,
            deslocamento_y,
        )

        intervalo_segundos = max(
            0.001,
            agora
            - estado.ultima_amostra_em,
        )

        velocidade = (
            distancia_pixels
            / intervalo_segundos
        )

        limite_horizontal = max(
            self.limite_movimento_minimo_pixels,
            max(
                bbox.largura,
                bbox_anterior.largura,
            )
            * self.limite_movimento_percentual,
        )

        limite_vertical = max(
            self.limite_movimento_minimo_pixels,
            max(
                bbox.altura,
                bbox_anterior.altura,
            )
            * self.limite_movimento_percentual,
        )

        movimento_horizontal = (
            self._classificar_movimento_horizontal(
                deslocamento_x=deslocamento_x,
                limite=limite_horizontal,
            )
        )

        movimento_vertical = (
            self._classificar_movimento_vertical(
                deslocamento_y=deslocamento_y,
                limite=limite_vertical,
            )
        )

        area_anterior = max(
            1,
            bbox_anterior.area,
        )

        variacao_area_percentual = (
            (
                bbox.area
                - area_anterior
            )
            / area_anterior
            * 100
        )

        tendencia_distancia = (
            self._classificar_tendencia_distancia(
                variacao_area_percentual
            )
        )

        quantidade_amostras = (
            estado.quantidade_amostras
            + 1
        )

        distancia_total = (
            estado.distancia_total_pixels
            + distancia_pixels
        )

        tempo_observado = max(
            0.0,
            agora
            - estado.primeira_amostra_em,
        )

        descricao = self._montar_descricao(
            movimento_horizontal=(
                movimento_horizontal
            ),
            movimento_vertical=(
                movimento_vertical
            ),
            tendencia_distancia=(
                tendencia_distancia
            ),
            velocidade=velocidade,
        )

        return PersonMovementAnalysis(
            pessoa_id=estado.pessoa_id,
            movimento_horizontal=(
                movimento_horizontal
            ),
            movimento_vertical=(
                movimento_vertical
            ),
            tendencia_distancia=(
                tendencia_distancia
            ),
            deslocamento_x=round(
                deslocamento_x,
                2,
            ),
            deslocamento_y=round(
                deslocamento_y,
                2,
            ),
            distancia_pixels=round(
                distancia_pixels,
                2,
            ),
            velocidade_pixels_segundo=round(
                velocidade,
                2,
            ),
            variacao_area_percentual=round(
                variacao_area_percentual,
                2,
            ),
            distancia_total_pixels=round(
                distancia_total,
                2,
            ),
            tempo_observado_segundos=round(
                tempo_observado,
                2,
            ),
            quantidade_amostras=(
                quantidade_amostras
            ),
            descricao=descricao,
        )

    def _classificar_movimento_horizontal(
        self,
        deslocamento_x: float,
        limite: float,
    ) -> MovimentoHorizontal:
        if abs(deslocamento_x) < limite:
            return "parado"

        if deslocamento_x > 0:
            return "direita"

        return "esquerda"

    def _classificar_movimento_vertical(
        self,
        deslocamento_y: float,
        limite: float,
    ) -> MovimentoVertical:
        if abs(deslocamento_y) < limite:
            return "parado"

        if deslocamento_y > 0:
            return "baixo"

        return "cima"

    def _classificar_tendencia_distancia(
        self,
        variacao_area_percentual: float,
    ) -> TendenciaDistancia:
        limite = (
            self
            .limite_variacao_area_percentual
        )

        if variacao_area_percentual >= limite:
            return "aproximando"

        if variacao_area_percentual <= -limite:
            return "afastando"

        return "estavel"

    def _montar_descricao(
        self,
        movimento_horizontal: MovimentoHorizontal,
        movimento_vertical: MovimentoVertical,
        tendencia_distancia: TendenciaDistancia,
        velocidade: float,
    ) -> str:
        movimentos: list[str] = []

        if movimento_horizontal == "direita":
            movimentos.append(
                "para a direita"
            )

        elif movimento_horizontal == "esquerda":
            movimentos.append(
                "para a esquerda"
            )

        if movimento_vertical == "cima":
            movimentos.append(
                "para cima"
            )

        elif movimento_vertical == "baixo":
            movimentos.append(
                "para baixo"
            )

        if not movimentos:
            descricao_movimento = (
                "Pessoa praticamente parada"
            )

        else:
            descricao_movimento = (
                "Pessoa se movimentando "
                + " e ".join(movimentos)
            )

        tendencias = {
            "inicial": "",
            "estavel": (
                " mantendo distância "
                "aparente estável"
            ),
            "aproximando": (
                " e aparentemente se aproximando "
                "da câmera"
            ),
            "afastando": (
                " e aparentemente se afastando "
                "da câmera"
            ),
        }

        return (
            f"{descricao_movimento}"
            f"{tendencias[tendencia_distancia]}, "
            f"com velocidade aproximada de "
            f"{velocidade:.2f} pixels por segundo."
        )


person_movement_memory = (
    PersonMovementMemory()
)