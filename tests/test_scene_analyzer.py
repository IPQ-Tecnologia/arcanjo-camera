import json
from pathlib import Path

from PIL import Image, ImageDraw

from app.services.scene_analyzer import analyze_person


def build_test_image(path: Path) -> None:
    image = Image.new(mode="RGB", size=(640, 480), color=(235, 235, 235))

    draw = ImageDraw.Draw(image)

    # Head
    draw.ellipse((290, 70, 350, 130), fill=(190, 145, 110))

    # Black shirt
    draw.rectangle((265, 125, 375, 300), fill=(20, 20, 20))

    # Pants
    draw.rectangle((280, 300, 360, 430), fill=(45, 65, 100))

    path.parent.mkdir(parents=True, exist_ok=True)

    image.save(path, format="JPEG", quality=95)


def main() -> None:
    path = Path("event_images/teste_scene_analyzer.jpg")

    build_test_image(path)

    result = analyze_person(image_path=path, x=250, y=60, width=140, height=380)

    print("===== VISUAL ANALYSIS =====")

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

    assert result.approximate_clothing_color == "preta"
    assert result.horizontal_position == "centro"

    print("\nTest completed successfully.")


if __name__ == "__main__":
    main()
