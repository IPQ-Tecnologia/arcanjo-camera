import asyncio

from app.services.appearance_memory import AppearanceMemory
from app.services.scene_analyzer import PersonVisualAnalysis


def build_analysis(
    color: str,
    rgb: tuple[int, int, int],
    position: str = "centro",
) -> PersonVisualAnalysis:
    return PersonVisualAnalysis(
        approximate_clothing_color=color,
        representative_rgb=rgb,
        horizontal_position=position,
        size_in_frame="medio",
        frame_percentage=8.0,
        description="Memory test",
    )


async def main() -> None:
    memory = AppearanceMemory()

    samples = [
        ("preta", (25, 25, 25)),
        ("cinza", (85, 85, 85)),
        ("preta", (28, 28, 28)),
        ("preta", (24, 24, 24)),
        ("cinza", (75, 75, 75)),
    ]

    expected_results = [
        "indefinida",
        "indefinida",
        "indefinida",
        "preta",
        "preta",
    ]

    print("===== COLOR STABILIZATION =====")

    for index, ((color, rgb), expected) in enumerate(zip(samples, expected_results), start=1):
        result = await memory.register(
            person_id="pessoa-1",
            analysis=build_analysis(
                color=color,
                rgb=rgb,
                position="direita" if index == len(samples) else "centro",
            ),
        )

        print(f"Sample {index}: reading={color:<8} result={result.cor_roupa_predominante}")

        assert result.cor_roupa_predominante == expected

    obtained_result = await memory.get("pessoa-1")

    assert obtained_result is not None
    assert obtained_result.cor_roupa_predominante == "preta"
    assert obtained_result.quantidade_amostras == 5
    assert obtained_result.posicao_atual == "direita"

    final_result = await memory.finalize("pessoa-1")

    assert final_result is not None
    assert final_result.cor_roupa_predominante == "preta"

    assert await memory.get("pessoa-1") is None
    assert memory.session_count == 0

    await memory.register(
        person_id="pessoa-2",
        analysis=build_analysis(color="cinza", rgb=(80, 80, 80)),
    )

    assert memory.session_count == 1

    await memory.clear()

    assert memory.session_count == 0

    print("\nMemory test completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
