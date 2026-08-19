from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from app.domain.models.camera_event import BoundingBox


# NOTE: ScenePerson, PersonProximity and SceneContextAnalysis are
# serialized as-is (via to_dict) into the panel/Kafka payload, so
# their field names are a wire contract and are intentionally kept in
# Portuguese, matching the frontend and any external consumer. Only
# the internal code around them is in English.
@dataclass(frozen=True)
class ScenePerson:
    """Represents a person identified in the scene."""

    indice: int
    origem: str

    x: int
    y: int
    largura: int
    altura: int

    centro_x: float
    centro_y: float

    posicao_horizontal: str
    posicao_vertical: str

    tamanho_no_quadro: str
    percentual_quadro: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PersonProximity:
    """Represents the distance between two people."""

    pessoa_a: int
    pessoa_b: int

    distancia_normalizada: float
    classificacao: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SceneContextAnalysis:
    """Full result of the scene context analysis."""

    quantidade_pessoas: int

    pessoas: list[ScenePerson]
    proximidades: list[PersonProximity]

    pessoas_esquerda: int
    pessoas_centro: int
    pessoas_direita: int

    pares_muito_proximos: int
    pares_proximos: int
    pares_separados: int

    descricao: str

    def to_dict(self) -> dict:
        return {
            "quantidade_pessoas": self.quantidade_pessoas,
            "pessoas": [person.to_dict() for person in self.pessoas],
            "proximidades": [proximity.to_dict() for proximity in self.proximidades],
            "pessoas_esquerda": self.pessoas_esquerda,
            "pessoas_centro": self.pessoas_centro,
            "pessoas_direita": self.pessoas_direita,
            "pares_muito_proximos": self.pares_muito_proximos,
            "pares_proximos": self.pares_proximos,
            "pares_separados": self.pares_separados,
            "descricao": self.descricao,
        }


def analyze_scene_context(
    image_width: int,
    image_height: int,
    bounding_boxes: list[BoundingBox],
) -> SceneContextAnalysis:
    """
    Analyzes how people are distributed across the scene.

    The analysis considers:

    - number of people;
    - horizontal position;
    - vertical position;
    - size in the frame;
    - proximity between people.
    """
    if image_width <= 0:
        raise ValueError("The image width must be positive")

    if image_height <= 0:
        raise ValueError("The image height must be positive")

    valid_boxes = _filter_valid_boxes(
        image_width=image_width,
        image_height=image_height,
        bounding_boxes=bounding_boxes,
    )

    people = [
        _create_person(
            index=index,
            bounding_box=bounding_box,
            image_width=image_width,
            image_height=image_height,
        )
        for index, bounding_box in enumerate(valid_boxes, start=1)
    ]

    proximities = _calculate_proximities(
        people=people,
        image_width=image_width,
        image_height=image_height,
    )

    people_left = sum(1 for person in people if person.posicao_horizontal == "esquerda")
    people_center = sum(1 for person in people if person.posicao_horizontal == "centro")
    people_right = sum(1 for person in people if person.posicao_horizontal == "direita")

    very_close_pairs = sum(
        1 for proximity in proximities if proximity.classificacao == "muito_proximas"
    )
    close_pairs = sum(1 for proximity in proximities if proximity.classificacao == "proximas")
    separate_pairs = sum(
        1 for proximity in proximities if proximity.classificacao == "separadas"
    )

    description = _build_description(
        person_count=len(people),
        people_left=people_left,
        people_center=people_center,
        people_right=people_right,
        very_close_pairs=very_close_pairs,
        close_pairs=close_pairs,
    )

    return SceneContextAnalysis(
        quantidade_pessoas=len(people),
        pessoas=people,
        proximidades=proximities,
        pessoas_esquerda=people_left,
        pessoas_centro=people_center,
        pessoas_direita=people_right,
        pares_muito_proximos=very_close_pairs,
        pares_proximos=close_pairs,
        pares_separados=separate_pairs,
        descricao=description,
    )


def _filter_valid_boxes(
    image_width: int,
    image_height: int,
    bounding_boxes: list[BoundingBox],
) -> list[BoundingBox]:
    """
    Removes invalid boxes, extremely small ones, or ones that
    represent practically the entire image.
    """
    valid_boxes: list[BoundingBox] = []
    image_area = image_width * image_height

    for bounding_box in bounding_boxes:
        if bounding_box.width <= 0 or bounding_box.height <= 0:
            continue

        if bounding_box.x2 <= bounding_box.x or bounding_box.y2 <= bounding_box.y:
            continue

        box_area = bounding_box.width * bounding_box.height
        percentage = box_area / image_area * 100

        # Some cameras send a box that represents practically the
        # entire image. That box doesn't correspond to a person.
        if percentage >= 60:
            continue

        # Ignores extremely small boxes.
        if percentage < 0.05:
            continue

        valid_boxes.append(bounding_box)

    return valid_boxes


def _create_person(
    index: int,
    bounding_box: BoundingBox,
    image_width: int,
    image_height: int,
) -> ScenePerson:
    center_x = bounding_box.x + bounding_box.width / 2
    center_y = bounding_box.y + bounding_box.height / 2

    frame_percentage = round(
        (bounding_box.width * bounding_box.height) / (image_width * image_height) * 100,
        2,
    )

    return ScenePerson(
        indice=index,
        origem=bounding_box.source,
        x=bounding_box.x,
        y=bounding_box.y,
        largura=bounding_box.width,
        altura=bounding_box.height,
        centro_x=round(center_x, 2),
        centro_y=round(center_y, 2),
        posicao_horizontal=_classify_horizontal_position(
            center_x=center_x, image_width=image_width
        ),
        posicao_vertical=_classify_vertical_position(
            center_y=center_y, image_height=image_height
        ),
        tamanho_no_quadro=_classify_size(frame_percentage),
        percentual_quadro=frame_percentage,
    )


def _classify_horizontal_position(center_x: float, image_width: int) -> str:
    ratio = center_x / image_width

    if ratio < 0.34:
        return "esquerda"

    if ratio < 0.67:
        return "centro"

    return "direita"


def _classify_vertical_position(center_y: float, image_height: int) -> str:
    ratio = center_y / image_height

    if ratio < 0.34:
        return "superior"

    if ratio < 0.67:
        return "central"

    return "inferior"


def _classify_size(frame_percentage: float) -> str:
    if frame_percentage < 3:
        return "pequeno"

    if frame_percentage < 12:
        return "medio"

    return "grande"


def _calculate_proximities(
    people: list[ScenePerson],
    image_width: int,
    image_height: int,
) -> list[PersonProximity]:
    proximities: list[PersonProximity] = []

    for index_a in range(len(people)):
        for index_b in range(index_a + 1, len(people)):
            person_a = people[index_a]
            person_b = people[index_b]

            diff_x = (person_a.centro_x - person_b.centro_x) / image_width
            diff_y = (person_a.centro_y - person_b.centro_y) / image_height
            distance = math.sqrt(diff_x**2 + diff_y**2)

            classification = _classify_proximity(distance)

            proximities.append(
                PersonProximity(
                    pessoa_a=person_a.indice,
                    pessoa_b=person_b.indice,
                    distancia_normalizada=round(distance, 3),
                    classificacao=classification,
                )
            )

    return proximities


def _classify_proximity(normalized_distance: float) -> str:
    """
    The distance is computed using the normalized dimensions of the
    image.

    Smaller values represent people who are closer together.
    """
    if normalized_distance <= 0.15:
        return "muito_proximas"

    if normalized_distance <= 0.30:
        return "proximas"

    return "separadas"


def _build_description(
    person_count: int,
    people_left: int,
    people_center: int,
    people_right: int,
    very_close_pairs: int,
    close_pairs: int,
) -> str:
    if person_count == 0:
        return "No person identified in the scene."

    if person_count == 1:
        if people_left == 1:
            position = "on the left"
        elif people_right == 1:
            position = "on the right"
        else:
            position = "in the center"

        return f"One person identified {position} of the scene."

    parts = [f"{person_count} people identified in the scene."]

    distribution: list[str] = []
    if people_left:
        distribution.append(f"{people_left} on the left")

    if people_center:
        distribution.append(f"{people_center} in the center")

    if people_right:
        distribution.append(f"{people_right} on the right")

    if distribution:
        parts.append("Distribution: " + ", ".join(distribution) + ".")

    if very_close_pairs == 1:
        parts.append("1 very close pair.")
    elif very_close_pairs > 1:
        parts.append(f"{very_close_pairs} very close pairs.")

    if close_pairs == 1:
        parts.append("1 close pair.")
    elif close_pairs > 1:
        parts.append(f"{close_pairs} close pairs.")

    if very_close_pairs == 0 and close_pairs == 0:
        parts.append("The people are spread apart.")

    return " ".join(parts)
