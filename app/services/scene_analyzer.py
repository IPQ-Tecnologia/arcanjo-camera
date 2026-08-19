from __future__ import annotations

import colorsys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Literal

from PIL import Image, ImageFilter

from app.services.person_detector_yolo import get_segmented_clothing_rgb


HorizontalPosition = Literal["esquerda", "centro", "direita"]
FrameSize = Literal["pequeno", "medio", "grande"]


@dataclass(frozen=True)
class PersonVisualAnalysis:
    approximate_clothing_color: str
    representative_rgb: tuple[int, int, int]
    horizontal_position: HorizontalPosition
    size_in_frame: FrameSize
    frame_percentage: float
    description: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["representative_rgb"] = list(self.representative_rgb)

        return data


def analyze_person(
    image_path: str | Path,
    x: int,
    y: int,
    width: int,
    height: int,
) -> PersonVisualAnalysis:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    if width <= 0 or height <= 0:
        raise ValueError("The bounding box must have positive width and height")

    with Image.open(path) as opened_image:
        image = opened_image.convert("RGB")
        image_width, image_height = image.size

        x1 = max(0, min(int(x), image_width - 1))
        y1 = max(0, min(int(y), image_height - 1))
        x2 = max(x1 + 1, min(int(x + width), image_width))
        y2 = max(y1 + 1, min(int(y + height), image_height))

        person_width = x2 - x1
        person_height = y2 - y1

        rgb = get_segmented_clothing_rgb(
            image_path=path,
            x=x1,
            y=y1,
            width=person_width,
            height=person_height,
        )

        # Fallback: keeps the old method when YOLO doesn't find a
        # matching mask.
        if rgb is None:
            clothing_region = _crop_clothing_region(
                image=image,
                x1=x1,
                y1=y1,
                width=person_width,
                height=person_height,
            )
            rgb = _get_representative_color(clothing_region)

        color = classify_rgb_color(rgb)
        position = _classify_position(
            center_x=x1 + person_width / 2,
            image_width=image_width,
        )
        frame_percentage = round(
            (person_width * person_height) / (image_width * image_height) * 100, 2
        )
        size = _classify_size(frame_percentage)
        description = _build_description(color=color, position=position, size=size)

        return PersonVisualAnalysis(
            approximate_clothing_color=color,
            representative_rgb=rgb,
            horizontal_position=position,
            size_in_frame=size,
            frame_percentage=frame_percentage,
            description=description,
        )


def _crop_clothing_region(
    image: Image.Image,
    x1: int,
    y1: int,
    width: int,
    height: int,
) -> Image.Image:
    """
    Crops the central region of the torso.

    The area is narrower than the bounding box to reduce background,
    arms, face, hands and pants.
    """
    clothing_x1 = int(x1 + width * 0.27)
    clothing_x2 = int(x1 + width * 0.73)
    clothing_y1 = int(y1 + height * 0.25)
    clothing_y2 = int(y1 + height * 0.58)

    clothing_x2 = max(clothing_x1 + 1, clothing_x2)
    clothing_y2 = max(clothing_y1 + 1, clothing_y2)

    return image.crop((clothing_x1, clothing_y1, clothing_x2, clothing_y2))


def _get_representative_color(region: Image.Image) -> tuple[int, int, int]:
    """
    Computes a robust color for the clothing.

    The previous method only discarded light pixels, leaving the
    result artificially dark. Now a small share of the darkest
    shadows and the brightest highlights is removed before computing
    the median RGB.
    """
    reduced_region = region.resize((64, 64), Image.Resampling.LANCZOS)
    smoothed_region = reduced_region.filter(ImageFilter.MedianFilter(size=3))
    pixels = list(smoothed_region.getdata())

    if not pixels:
        return (0, 0, 0)

    sorted_pixels = sorted(pixels, key=_calculate_luminance)
    count = len(sorted_pixels)
    trim = int(count * 0.08)

    if trim > 0 and count - trim > trim:
        candidates = sorted_pixels[trim : count - trim]
    else:
        candidates = sorted_pixels

    min_count = max(100, count // 3)
    if len(candidates) < min_count:
        candidates = pixels

    red = int(round(median(pixel[0] for pixel in candidates)))
    green = int(round(median(pixel[1] for pixel in candidates)))
    blue = int(round(median(pixel[2] for pixel in candidates)))

    return (red, green, blue)


def _calculate_luminance(rgb: tuple[int, int, int]) -> float:
    red, green, blue = rgb

    return red * 0.2126 + green * 0.7152 + blue * 0.0722


def classify_rgb_color(rgb: tuple[int, int, int]) -> str:
    """
    Classifies an RGB color using HSV, luminance and the difference
    between channels.

    The rule avoids calling any dark color "preta" (black). Dark
    tones with perceptible saturation are classified as
    azul-escura, verde-escura, vermelha-escura or roxa-escura
    (dark blue/green/red/purple) — the color names themselves stay in
    Portuguese, matching the panel/Kafka contract.
    """
    red, green, blue = (max(0, min(255, int(channel))) for channel in rgb)

    r = red / 255
    g = green / 255
    b = blue / 255

    hue, saturation, brightness = colorsys.rgb_to_hsv(r, g, b)
    hue_degrees = hue * 360
    luminance = _calculate_luminance((red, green, blue))

    max_channel = max(red, green, blue)
    min_channel = min(red, green, blue)
    channel_difference = max_channel - min_channel

    # Low-light zone where black and navy blue become visually
    # indistinguishable.
    #
    # In this range, small differences between channels can just be
    # noise, JPEG compression or camera lighting. It's safer not to
    # invent a color.
    if (
        brightness <= 0.20
        and max_channel <= 55
        and 0.16 < saturation < 0.38
        and 5 < channel_difference < 18
    ):
        return "escura-indefinida"

    sorted_channels = sorted((red, green, blue), reverse=True)
    second_highest_channel = sorted_channels[1]
    dominant_channel_edge = max_channel - second_highest_channel

    # In extremely dark regions, the HSV hue becomes unstable and
    # small camera noise can turn black into blue, green or red.
    #
    # Only keeps a very dark color when there is enough chromatic
    # signal in the pixels.
    if brightness <= 0.18:
        perceptible_dark_color = (
            max_channel >= 34
            and saturation >= 0.20
            and channel_difference >= 8
            and dominant_channel_edge >= 5
        )
        if not perceptible_dark_color:
            return "preta"

    # True black or lit black. In dark tones, a larger difference
    # between channels can still just be the tint of warm light
    # falling on black fabric.
    if (
        brightness <= 0.14
        or (brightness <= 0.28 and channel_difference <= 25)
        or (luminance <= 58 and saturation <= 0.32)
    ):
        return "preta"

    # White and neutral tones.
    if luminance >= 218 and saturation <= 0.18:
        return "branca"

    if saturation <= 0.16:
        if luminance < 135:
            return "cinza-escura"

        return "cinza"

    # Brown and beige sit in the orange hue range, separated by
    # brightness.
    if 15 <= hue_degrees < 48 and saturation >= 0.22:
        if brightness <= 0.62:
            return "marrom"

        if saturation <= 0.48 and brightness >= 0.66:
            return "bege"

        return "laranja"

    base_color = _classify_by_hue(hue_degrees)

    # Preserves the color of saturated dark clothing.
    if brightness <= 0.42:
        dark_colors = {
            "vermelha": "vermelha-escura",
            "verde": "verde-escura",
            "azul": "azul-escura",
            "roxa": "roxa-escura",
        }
        return dark_colors.get(base_color, base_color)

    return base_color


def _classify_by_hue(hue_degrees: float) -> str:
    if hue_degrees < 15 or hue_degrees >= 345:
        return "vermelha"

    if hue_degrees < 48:
        return "laranja"

    if hue_degrees < 72:
        return "amarela"

    if hue_degrees < 170:
        return "verde"

    if hue_degrees < 200:
        return "azul"

    if hue_degrees < 260:
        return "azul"

    if hue_degrees < 315:
        return "roxa"

    return "rosa"


def _classify_color(rgb: tuple[int, int, int]) -> str:
    """Kept for compatibility with old tests or imports that used the private function."""
    return classify_rgb_color(rgb)


def _classify_position(center_x: float, image_width: int) -> HorizontalPosition:
    ratio = center_x / image_width

    if ratio < 0.34:
        return "esquerda"

    if ratio < 0.67:
        return "centro"

    return "direita"


def _classify_size(frame_percentage: float) -> FrameSize:
    if frame_percentage < 3:
        return "pequeno"

    if frame_percentage < 12:
        return "medio"

    return "grande"


def _build_description(color: str, position: HorizontalPosition, size: FrameSize) -> str:
    position_descriptions = {
        "esquerda": "on the left",
        "centro": "in the center",
        "direita": "on the right",
    }
    size_descriptions = {
        "pequeno": "occupying a small part of the frame",
        "medio": "occupying an intermediate part of the frame",
        "grande": "occupying a large part of the frame",
    }

    return (
        f"Person predominantly wearing {color}-colored clothing, located "
        f"{position_descriptions[position]} of the scene, {size_descriptions[size]}."
    )
