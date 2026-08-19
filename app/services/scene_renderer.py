from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.domain.models.camera_event import BoundingBox


@dataclass(frozen=True)
class DrawableBoundingBox:
    """Bounding box already converted to pixels and ready to be drawn on the image."""

    source: str

    x1: int
    y1: int
    x2: int
    y2: int

    width: int
    height: int

    frame_percentage: float


@dataclass(frozen=True)
class SceneRenderResult:
    """Result of rendering the scene."""

    image_base64: str
    box_count: int
    image_width: int
    image_height: int


def render_scene_with_boxes(
    image_path: str,
    bounding_boxes: list[BoundingBox],
) -> SceneRenderResult:
    """
    Opens the original image and draws every valid bounding box.

    Returns the annotated image in Base64.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as opened_image:
        image = opened_image.convert("RGB")

    image_width, image_height = image.size

    valid_boxes = _prepare_bounding_boxes(
        bounding_boxes=bounding_boxes,
        image_width=image_width,
        image_height=image_height,
    )

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    colors = [
        (239, 68, 68),
        (34, 197, 94),
        (59, 130, 246),
        (234, 179, 8),
        (168, 85, 247),
        (236, 72, 153),
        (6, 182, 212),
        (249, 115, 22),
    ]

    line_width = max(2, min(6, round(image_width / 450)))
    center_radius = max(3, line_width + 1)

    for index, box in enumerate(valid_boxes, start=1):
        color = colors[(index - 1) % len(colors)]

        draw.rectangle(
            (box.x1, box.y1, box.x2, box.y2),
            outline=color,
            width=line_width,
        )

        center_x = int((box.x1 + box.x2) / 2)
        center_y = int((box.y1 + box.y2) / 2)

        draw.ellipse(
            (
                center_x - center_radius,
                center_y - center_radius,
                center_x + center_radius,
                center_y + center_radius,
            ),
            fill=color,
        )

        text = f"Person {index} ({box.frame_percentage:.2f}%)"
        text_bounds = draw.textbbox((0, 0), text, font=font)
        text_width = text_bounds[2] - text_bounds[0]
        text_height = text_bounds[3] - text_bounds[1]

        text_margin = 5
        text_x1 = box.x1
        text_y1 = max(0, box.y1 - text_height - text_margin * 2)
        text_x2 = min(image_width - 1, text_x1 + text_width + text_margin * 2)
        text_y2 = min(image_height - 1, text_y1 + text_height + text_margin * 2)

        draw.rectangle((text_x1, text_y1, text_x2, text_y2), fill=color)
        draw.text(
            (text_x1 + text_margin, text_y1 + text_margin),
            text,
            fill=(255, 255, 255),
            font=font,
        )

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=88, optimize=True)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    return SceneRenderResult(
        image_base64=image_base64,
        box_count=len(valid_boxes),
        image_width=image_width,
        image_height=image_height,
    )


def _prepare_bounding_boxes(
    bounding_boxes: list[BoundingBox],
    image_width: int,
    image_height: int,
) -> list[DrawableBoundingBox]:
    """
    Converts coordinates to pixels, clamps the boxes to the image size
    and removes invalid or duplicate boxes.
    """
    valid_boxes: list[DrawableBoundingBox] = []
    seen_boxes: set[tuple[int, int, int, int]] = set()
    image_area = image_width * image_height

    for bounding_box in bounding_boxes:
        coordinates = _convert_to_pixels(
            bounding_box=bounding_box,
            image_width=image_width,
            image_height=image_height,
        )
        if coordinates is None:
            continue

        x1, y1, x2, y2 = coordinates

        x1 = max(0, min(image_width - 1, x1))
        y1 = max(0, min(image_height - 1, y1))
        x2 = max(0, min(image_width - 1, x2))
        y2 = max(0, min(image_height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        width = x2 - x1
        height = y2 - y1
        box_area = width * height
        frame_percentage = box_area / image_area * 100

        # Ignores the box that represents almost the entire image.
        if frame_percentage >= 60:
            continue

        # Ignores boxes too small to represent a useful detection.
        if frame_percentage < 0.05:
            continue

        key = (x1, y1, x2, y2)
        if key in seen_boxes:
            continue

        seen_boxes.add(key)

        valid_boxes.append(
            DrawableBoundingBox(
                source=bounding_box.source or "desconhecida",
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                width=width,
                height=height,
                frame_percentage=round(frame_percentage, 2),
            )
        )

    return valid_boxes


def _convert_to_pixels(
    bounding_box: BoundingBox,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    """Supports coordinates in pixels and coordinates normalized between zero and one."""
    try:
        x = float(bounding_box.x)
        y = float(bounding_box.y)
        width = float(bounding_box.width)
        height = float(bounding_box.height)
    except (TypeError, ValueError):
        return None

    if width <= 0 or height <= 0:
        return None

    normalized_values = 0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1

    if normalized_values:
        x1 = round(x * image_width)
        y1 = round(y * image_height)
        x2 = round((x + width) * image_width)
        y2 = round((y + height) * image_height)
    else:
        x1 = round(x)
        y1 = round(y)
        x2 = round(x + width)
        y2 = round(y + height)

    return (x1, y1, x2, y2)
