import json

from app.domain.models.camera_event import BoundingBox
from app.services.scene_context_analyzer import analyze_scene_context


def build_box(
    x: int,
    y: int,
    width: int,
    height: int,
    source: str = "targetrect",
) -> BoundingBox:
    return BoundingBox(
        source=source,
        x=x,
        y=y,
        width=width,
        height=height,
        x2=x + width,
        y2=y + height,
        image_ratio=False,
    )


def main() -> None:
    boxes = [
        # Person 1: left
        build_box(x=110, y=180, width=130, height=360),
        # Person 2: center
        build_box(x=510, y=170, width=135, height=370),
        # Person 3: center and close to person 2
        build_box(x=680, y=175, width=130, height=365),
        # Box representing almost the entire image. It should be ignored.
        build_box(x=0, y=0, width=1280, height=720, source="full_image"),
    ]

    result = analyze_scene_context(
        image_width=1280,
        image_height=720,
        bounding_boxes=boxes,
    )

    print("===== SCENE CONTEXT =====")

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

    assert result.quantidade_pessoas == 3, "Three people should be identified"
    assert result.pessoas_esquerda == 1, "There should be one person on the left"
    assert result.pessoas_centro == 2, "There should be two people in the center"
    assert result.pessoas_direita == 0, "There should be no person on the right"
    assert len(result.proximidades) == 3, "Three people should produce three comparisons"
    assert result.pares_muito_proximos == 1, "People 2 and 3 should be very close"

    print("\nNumber of people:", result.quantidade_pessoas)
    print("People on the left:", result.pessoas_esquerda)
    print("People in the center:", result.pessoas_centro)
    print("People on the right:", result.pessoas_direita)
    print("Very close pairs:", result.pares_muito_proximos)
    print("Close pairs:", result.pares_proximos)
    print("Description:", result.descricao)

    print("\nTest completed successfully.")


if __name__ == "__main__":
    main()
