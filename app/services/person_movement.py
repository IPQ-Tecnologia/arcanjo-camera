from __future__ import annotations

import asyncio
import math
import time
from dataclasses import asdict, dataclass
from typing import Literal

from app.services.person_tracker import DetectionBox


HorizontalMovement = Literal["inicial", "parado", "esquerda", "direita"]
VerticalMovement = Literal["inicial", "parado", "cima", "baixo"]
DistanceTrend = Literal["inicial", "estavel", "aproximando", "afastando"]


# NOTE: PersonMovementAnalysis is serialized as-is (via to_dict) into
# the panel/Kafka payload, so its field names are a wire contract and
# are intentionally kept in Portuguese, matching the frontend and any
# external consumer. Only the internal code around it is in English.
@dataclass(frozen=True)
class PersonMovementAnalysis:
    pessoa_id: str

    movimento_horizontal: HorizontalMovement
    movimento_vertical: VerticalMovement
    tendencia_distancia: DistanceTrend

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
    person_id: str

    first_sample_at: float
    last_sample_at: float

    last_bbox: DetectionBox

    sample_count: int
    total_distance_pixels: float

    last_analysis: PersonMovementAnalysis


class PersonMovementMemory:
    def __init__(
        self,
        movement_percent_threshold: float = 0.05,
        min_movement_pixels: float = 6.0,
        area_variation_percent_threshold: float = 12.0,
    ) -> None:
        if movement_percent_threshold < 0:
            raise ValueError("movement_percent_threshold cannot be negative")

        if min_movement_pixels < 0:
            raise ValueError("min_movement_pixels cannot be negative")

        if area_variation_percent_threshold < 0:
            raise ValueError("area_variation_percent_threshold cannot be negative")

        self.movement_percent_threshold = movement_percent_threshold
        self.min_movement_pixels = min_movement_pixels
        self.area_variation_percent_threshold = area_variation_percent_threshold

        self._states: dict[str, _MovementState] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        person_id: str,
        bbox: DetectionBox,
        now: float | None = None,
    ) -> PersonMovementAnalysis:
        if not person_id.strip():
            raise ValueError("person_id cannot be empty")

        if bbox.width <= 0 or bbox.height <= 0:
            raise ValueError("The bounding box must have positive width and height")

        moment = time.monotonic() if now is None else now

        async with self._lock:
            state = self._states.get(person_id)

            if state is None:
                analysis = self._create_first_analysis(person_id=person_id)
                self._states[person_id] = _MovementState(
                    person_id=person_id,
                    first_sample_at=moment,
                    last_sample_at=moment,
                    last_bbox=bbox,
                    sample_count=1,
                    total_distance_pixels=0.0,
                    last_analysis=analysis,
                )
                return analysis

            analysis = self._calculate_movement(state=state, bbox=bbox, now=moment)

            state.last_sample_at = moment
            state.last_bbox = bbox
            state.sample_count += 1
            state.total_distance_pixels = analysis.distancia_total_pixels
            state.last_analysis = analysis

            return analysis

    async def get(self, person_id: str) -> PersonMovementAnalysis | None:
        async with self._lock:
            state = self._states.get(person_id)
            if state is None:
                return None

            return state.last_analysis

    async def finalize(self, person_id: str) -> PersonMovementAnalysis | None:
        async with self._lock:
            state = self._states.pop(person_id, None)
            if state is None:
                return None

            return state.last_analysis

    async def clear(self) -> None:
        async with self._lock:
            self._states.clear()

    @property
    def person_count(self) -> int:
        return len(self._states)

    def _create_first_analysis(self, person_id: str) -> PersonMovementAnalysis:
        return PersonMovementAnalysis(
            pessoa_id=person_id,
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
            descricao="First position of the person recorded.",
        )

    def _calculate_movement(
        self,
        state: _MovementState,
        bbox: DetectionBox,
        now: float,
    ) -> PersonMovementAnalysis:
        previous_bbox = state.last_bbox

        displacement_x = bbox.center_x - previous_bbox.center_x
        displacement_y = bbox.center_y - previous_bbox.center_y
        distance_pixels = math.hypot(displacement_x, displacement_y)

        interval_seconds = max(0.001, now - state.last_sample_at)
        speed = distance_pixels / interval_seconds

        horizontal_threshold = max(
            self.min_movement_pixels,
            max(bbox.width, previous_bbox.width) * self.movement_percent_threshold,
        )
        vertical_threshold = max(
            self.min_movement_pixels,
            max(bbox.height, previous_bbox.height) * self.movement_percent_threshold,
        )

        horizontal_movement = self._classify_horizontal_movement(
            displacement_x=displacement_x,
            threshold=horizontal_threshold,
        )
        vertical_movement = self._classify_vertical_movement(
            displacement_y=displacement_y,
            threshold=vertical_threshold,
        )

        previous_area = max(1, previous_bbox.area)
        area_variation_percent = (bbox.area - previous_area) / previous_area * 100

        distance_trend = self._classify_distance_trend(area_variation_percent)

        sample_count = state.sample_count + 1
        total_distance = state.total_distance_pixels + distance_pixels
        observed_time = max(0.0, now - state.first_sample_at)

        description = self._build_description(
            horizontal_movement=horizontal_movement,
            vertical_movement=vertical_movement,
            distance_trend=distance_trend,
            speed=speed,
        )

        return PersonMovementAnalysis(
            pessoa_id=state.person_id,
            movimento_horizontal=horizontal_movement,
            movimento_vertical=vertical_movement,
            tendencia_distancia=distance_trend,
            deslocamento_x=round(displacement_x, 2),
            deslocamento_y=round(displacement_y, 2),
            distancia_pixels=round(distance_pixels, 2),
            velocidade_pixels_segundo=round(speed, 2),
            variacao_area_percentual=round(area_variation_percent, 2),
            distancia_total_pixels=round(total_distance, 2),
            tempo_observado_segundos=round(observed_time, 2),
            quantidade_amostras=sample_count,
            descricao=description,
        )

    def _classify_horizontal_movement(
        self,
        displacement_x: float,
        threshold: float,
    ) -> HorizontalMovement:
        if abs(displacement_x) < threshold:
            return "parado"

        if displacement_x > 0:
            return "direita"

        return "esquerda"

    def _classify_vertical_movement(
        self,
        displacement_y: float,
        threshold: float,
    ) -> VerticalMovement:
        if abs(displacement_y) < threshold:
            return "parado"

        if displacement_y > 0:
            return "baixo"

        return "cima"

    def _classify_distance_trend(self, area_variation_percent: float) -> DistanceTrend:
        threshold = self.area_variation_percent_threshold

        if area_variation_percent >= threshold:
            return "aproximando"

        if area_variation_percent <= -threshold:
            return "afastando"

        return "estavel"

    def _build_description(
        self,
        horizontal_movement: HorizontalMovement,
        vertical_movement: VerticalMovement,
        distance_trend: DistanceTrend,
        speed: float,
    ) -> str:
        movements: list[str] = []

        if horizontal_movement == "direita":
            movements.append("to the right")
        elif horizontal_movement == "esquerda":
            movements.append("to the left")

        if vertical_movement == "cima":
            movements.append("upward")
        elif vertical_movement == "baixo":
            movements.append("downward")

        if not movements:
            movement_description = "Person practically standing still"
        else:
            movement_description = "Person moving " + " and ".join(movements)

        trends = {
            "inicial": "",
            "estavel": ", keeping a stable apparent distance",
            "aproximando": " and apparently getting closer to the camera",
            "afastando": " and apparently moving away from the camera",
        }

        return (
            f"{movement_description}{trends[distance_trend]}, "
            f"at an approximate speed of {speed:.2f} pixels per second."
        )


person_movement_memory = PersonMovementMemory()
