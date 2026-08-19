import asyncio
from collections import Counter
from dataclasses import asdict, dataclass, field

from app.services.scene_analyzer import PersonVisualAnalysis


# NOTE: StableAppearance is serialized as-is (via to_dict) into the
# panel/Kafka payload, so its field names are a wire contract and are
# intentionally kept in Portuguese, matching the frontend and any
# external consumer. Only the internal code around it is in English.
@dataclass(frozen=True)
class StableAppearance:
    cor_roupa_predominante: str
    rgb_medio: tuple[int, int, int]
    posicao_atual: str
    tamanho_predominante: str
    percentual_medio_quadro: float
    quantidade_amostras: int
    descricao: str

    def to_dict(self) -> dict:
        dados = asdict(self)
        dados["rgb_medio"] = list(self.rgb_medio)

        return dados


@dataclass
class _AppearanceState:
    colors: Counter[str] = field(default_factory=Counter)
    sizes: Counter[str] = field(default_factory=Counter)

    red_sum: int = 0
    green_sum: int = 0
    blue_sum: int = 0

    percentage_sum: float = 0.0
    sample_count: int = 0

    last_color: str = "indefinida"

    # Color shown as the stable result. Kept separate from the latest
    # reading to avoid flip-flopping caused by lighting and shadows.
    stable_color: str = "indefinida"

    last_position: str = "centro"
    last_size: str = "medio"


class AppearanceMemory:
    def __init__(self) -> None:
        self._sessions: dict[str, _AppearanceState] = {}
        self._lock = asyncio.Lock()

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    async def register(
        self,
        person_id: str,
        analysis: PersonVisualAnalysis,
    ) -> StableAppearance:
        if not person_id:
            raise ValueError("person_id cannot be empty")

        async with self._lock:
            state = self._sessions.get(person_id)
            if state is None:
                state = _AppearanceState()
                self._sessions[person_id] = state

            state.colors[analysis.approximate_clothing_color] += 1
            state.sizes[analysis.size_in_frame] += 1

            red, green, blue = analysis.representative_rgb
            state.red_sum += red
            state.green_sum += green
            state.blue_sum += blue

            state.percentage_sum += analysis.frame_percentage
            state.sample_count += 1

            state.last_color = analysis.approximate_clothing_color
            state.last_position = analysis.horizontal_position
            state.last_size = analysis.size_in_frame

            return self._build_result(state)

    async def get(self, person_id: str) -> StableAppearance | None:
        async with self._lock:
            state = self._sessions.get(person_id)
            if state is None:
                return None

            return self._build_result(state)

    async def finalize(self, person_id: str) -> StableAppearance | None:
        """Returns the final appearance and removes that person's session from memory."""
        async with self._lock:
            state = self._sessions.pop(person_id, None)
            if state is None:
                return None

            return self._build_result(state)

    async def clear(self) -> None:
        async with self._lock:
            self._sessions.clear()

    def _build_result(self, state: _AppearanceState) -> StableAppearance:
        count = max(1, state.sample_count)
        computed_color = self._select_predominant_color(state)
        top_color_count = max(state.colors.values(), default=0)

        # For the first stable color, requires at least three
        # readings of the same color and at least 75% agreement. With
        # three samples, all three must match; with four, at least
        # three.
        color_consistent = (
            count >= 3
            and top_color_count >= 3
            and top_color_count * 4 >= count * 3
        )

        # Clothing doesn't change during a short passage. Once the
        # color becomes stable, isolated different readings don't
        # erase the result.
        if state.stable_color != "indefinida":
            predominant_color = state.stable_color
        elif color_consistent:
            predominant_color = computed_color
            state.stable_color = predominant_color
        else:
            predominant_color = "indefinida"

        predominant_size = self._select_predominant(
            counter=state.sizes,
            most_recent_value=state.last_size,
        )

        average_rgb = (
            round(state.red_sum / count),
            round(state.green_sum / count),
            round(state.blue_sum / count),
        )
        average_percentage = round(state.percentage_sum / count, 2)

        description = self._build_description(
            color=predominant_color,
            position=state.last_position,
            size=predominant_size,
            count=count,
        )

        return StableAppearance(
            cor_roupa_predominante=predominant_color,
            rgb_medio=average_rgb,
            posicao_atual=state.last_position,
            tamanho_predominante=predominant_size,
            percentual_medio_quadro=average_percentage,
            quantidade_amostras=count,
            descricao=description,
        )

    def _select_predominant_color(self, state: _AppearanceState) -> str:
        """
        Stabilizes the classification of dark clothing that flip-flops
        between black and a dark color due to lighting, distance or
        shadow.
        """
        colors = state.colors
        if not colors:
            return state.last_color

        default_result = self._select_predominant(
            counter=colors,
            most_recent_value=state.last_color,
        )

        dark_colors = ("azul-escura", "verde-escura", "vermelha-escura", "roxa-escura")
        black_count = colors.get("preta", 0)
        candidates = [
            (color, colors.get(color, 0)) for color in dark_colors if colors.get(color, 0) > 0
        ]

        if black_count == 0 or not candidates:
            return default_result

        top_dark_count = max(count for _, count in candidates)
        tied = [color for color, count in candidates if count == top_dark_count]

        if state.stable_color in tied:
            dark_color = state.stable_color
        elif state.last_color in tied:
            dark_color = state.last_color
        else:
            dark_color = tied[0]

        conflict_colors = set(dark_colors) | {"preta"}
        top_other_color_count = max(
            (count for color, count in colors.items() if color not in conflict_colors),
            default=0,
        )

        if top_other_color_count > max(black_count, top_dark_count):
            return default_result

        # Already stabilized as azul-escura, verde-escura or another
        # dark color. An isolated black frame shouldn't immediately
        # change the result.
        if state.stable_color in dark_colors:
            stable_color = state.stable_color
            stable_color_count = colors.get(stable_color, 0)

            # Allows switching between dark colors only when the new
            # color is two samples ahead of the current one.
            if dark_color != stable_color and top_dark_count >= stable_color_count + 2:
                stable_color = dark_color
                stable_color_count = top_dark_count

            # Returns to black only when the black readings are two
            # samples ahead of the stabilized dark color.
            if black_count >= stable_color_count + 2:
                return "preta"

            return stable_color

        # Was stabilized as black. Requires two readings of the dark
        # color and a tie or advantage to switch.
        if state.stable_color == "preta":
            if top_dark_count >= 2 and top_dark_count >= black_count:
                return dark_color

            return "preta"

        # Initial state, before a stable color.
        if top_dark_count >= 2 and top_dark_count >= black_count:
            return dark_color

        return default_result

    @staticmethod
    def _select_predominant(
        counter: Counter[str],
        most_recent_value: str,
    ) -> str:
        if not counter:
            return most_recent_value

        top_count = max(counter.values())
        tied = [value for value, count in counter.items() if count == top_count]

        # In case of a tie, uses the most recently observed value.
        if most_recent_value in tied:
            return most_recent_value

        return tied[0]

    @staticmethod
    def _build_description(color: str, position: str, size: str, count: int) -> str:
        positions = {
            "esquerda": "on the left",
            "centro": "in the center",
            "direita": "on the right",
        }
        sizes = {
            "pequeno": "small",
            "medio": "medium",
            "grande": "large",
        }

        formatted_position = positions.get(position, position)
        formatted_size = sizes.get(size, size)

        return (
            f"Person predominantly wearing {color}-colored clothing, currently located "
            f"{formatted_position} of the scene, with an apparent {formatted_size} size. "
            f"Result based on {count} sample(s)."
        )


appearance_memory = AppearanceMemory()
