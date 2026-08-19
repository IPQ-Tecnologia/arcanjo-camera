import asyncio

from app.services.person_movement import PersonMovementMemory
from app.services.person_tracker import DetectionBox


def build_box(x: int, y: int, width: int, height: int) -> DetectionBox:
    return DetectionBox(x=x, y=y, width=width, height=height)


async def run_test() -> None:
    memory = PersonMovementMemory()

    person_id = "camera-01-teste"

    first = await memory.register(
        person_id=person_id,
        bbox=build_box(x=100, y=100, width=100, height=200),
        now=100.0,
    )

    assert first.movimento_horizontal == "inicial"
    assert first.quantidade_amostras == 1

    second = await memory.register(
        person_id=person_id,
        bbox=build_box(x=150, y=100, width=100, height=200),
        now=102.0,
    )

    assert second.movimento_horizontal == "direita"
    assert second.movimento_vertical == "parado"
    assert second.tendencia_distancia == "estavel"
    assert second.velocidade_pixels_segundo == 25.0

    third = await memory.register(
        person_id=person_id,
        bbox=build_box(x=135, y=60, width=130, height=240),
        now=104.0,
    )

    assert third.movimento_horizontal == "parado"
    assert third.movimento_vertical == "cima"
    assert third.tendencia_distancia == "aproximando"

    fourth = await memory.register(
        person_id=person_id,
        bbox=build_box(x=80, y=60, width=90, height=180),
        now=106.0,
    )

    assert fourth.movimento_horizontal == "esquerda"
    assert fourth.tendencia_distancia == "afastando"
    assert fourth.quantidade_amostras == 4
    assert fourth.tempo_observado_segundos == 6.0
    assert fourth.distancia_total_pixels > 0

    final = await memory.finalize(person_id)

    assert final is not None
    assert memory.person_count == 0

    print("===== MOVEMENT ANALYSIS =====")
    print("First:", first.to_dict())
    print("Second:", second.to_dict())
    print("Third:", third.to_dict())
    print("Fourth:", fourth.to_dict())
    print("\nTest completed successfully.")


if __name__ == "__main__":
    asyncio.run(run_test())
