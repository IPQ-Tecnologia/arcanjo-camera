import base64
from pathlib import Path

from PIL import Image

from app.domain.models.camera_event import BoundingBox
from app.services.scene_renderer import render_scene_with_boxes


def find_image() -> Path:
    """
    Looks for the most recent original image.

    Images that already have annotations are skipped to avoid drawing
    new bounding boxes on top of old rectangles.
    """
    folders = [
        Path("event_images"),
        Path("eventos_selecionados"),
    ]

    original_images: list[Path] = []

    for folder in folders:
        if not folder.exists():
            continue

        candidates = [*folder.rglob("*.jpg"), *folder.rglob("*.jpeg")]

        for path in candidates:
            name = path.name.lower()

            # Skips any image that already has bounding boxes drawn.
            if "_marcada" in name:
                continue

            # Skips the output produced by this test.
            if name == "teste_cena_marcada.jpg":
                continue

            original_images.append(path)

    if not original_images:
        raise FileNotFoundError(
            "No original JPG image was found in event_images or eventos_selecionados."
        )

    return max(original_images, key=lambda path: path.stat().st_mtime)


def build_bounding_box(x: int, y: int, width: int, height: int, source: str) -> BoundingBox:
    """Creates a bounding box for the test."""
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
    image_path = find_image()

    with Image.open(image_path) as image:
        image_width = image.width
        image_height = image.height

    person_1_x = int(image_width * 0.12)
    person_1_y = int(image_height * 0.18)
    person_1_width = int(image_width * 0.18)
    person_1_height = int(image_height * 0.65)

    boxes = [
        # Person 1.
        build_bounding_box(
            x=person_1_x,
            y=person_1_y,
            width=person_1_width,
            height=person_1_height,
            source="teste_pessoa_1",
        ),
        # Person 2.
        build_bounding_box(
            x=int(image_width * 0.48),
            y=int(image_height * 0.22),
            width=int(image_width * 0.17),
            height=int(image_height * 0.60),
            source="teste_pessoa_2",
        ),
        # Person 3.
        build_bounding_box(
            x=int(image_width * 0.70),
            y=int(image_height * 0.20),
            width=int(image_width * 0.16),
            height=int(image_height * 0.62),
            source="teste_pessoa_3",
        ),
        # This box represents the entire image. The renderer should
        # ignore it.
        build_bounding_box(
            x=0,
            y=0,
            width=image_width,
            height=image_height,
            source="full_image",
        ),
        # This box is identical to the first one. The renderer should
        # ignore it for being a duplicate.
        build_bounding_box(
            x=person_1_x,
            y=person_1_y,
            width=person_1_width,
            height=person_1_height,
            source="duplicated",
        ),
    ]

    result = render_scene_with_boxes(
        image_path=str(image_path),
        bounding_boxes=boxes,
    )

    output_path = Path("teste_cena_marcada.jpg")

    image_bytes = base64.b64decode(result.image_base64)

    output_path.write_bytes(image_bytes)

    assert result.box_count == 3, "Exactly three bounding boxes should have been drawn."
    assert output_path.is_file(), "The annotated image was not created."
    assert output_path.stat().st_size > 0, "The annotated image was created empty."
    assert result.image_width == image_width, "The image width was changed."
    assert result.image_height == image_height, "The image height was changed."

    print("===== RENDERER TEST =====")
    print("Original image:", image_path)
    print("Annotated image:", output_path)
    print("Dimensions:", f"{result.image_width}x{result.image_height}")
    print("Bounding boxes drawn:", result.box_count)
    print("Base64 image characters:", len(result.image_base64))
    print("File size:", output_path.stat().st_size, "bytes")

    print("\nTest completed successfully.")


if __name__ == "__main__":
    main()
