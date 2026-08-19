from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class EventPoint(BaseModel):
    """
    Normalized geometry point.

    Values should remain between 0 and 1 whenever
    the manufacturer provides a known coordinate scale.
    """

    x: float
    y: float


class EventAttributes(BaseModel):
    """
    Relevant information received from the manufacturer
    that is not part of the main normalized contract.

    Some duplicated internal fields remain available
    for compatibility but are not serialized.
    """

    manufacturer: str = Field(exclude=True)

    category: str | None = Field(
        default=None,
        serialization_alias="event_category",
    )

    target_type: str | None = Field(
        default=None,
        exclude=True,
    )

    state: str | None = Field(
        default=None,
        exclude=True,
    )

    action: str | None = Field(
        default=None,
        serialization_alias="vendor_action",
    )

    vendor_event_type: str | None = None
    source_event_id: str | None = None

    rule_id: str | None = None
    rule_name: str | None = None

    object_id: str | None = None
    group_id: str | None = None

    sensitivity: float | None = None
    direction: str | None = None
    confidence: float | None = None

    geometry_type: str | None = None
    geometry: list[EventPoint] = Field(default_factory=list)

    raw_bounding_box: list[float] | None = None

    vendor_data: dict[str, Any] = Field(default_factory=dict)


class BoundingBox(BaseModel):
    source: str | None = None

    x: int
    y: int

    width: int
    height: int

    x2: int
    y2: int

    image_ratio: float | None = None


class ImageData(BaseModel):
    width: int | None = None
    height: int | None = None
    format: str | None = None

    original_path: str | None = None
    annotated_path: str | None = None
    face_crop_path: str | None = None


class CameraEvent(BaseModel):
    schema_version: str = "1.0"

    event_id: str
    manufacturer: str

    camera_model: str | None = None
    camera_id: str | None = None
    camera_name: str | None = None
    camera_ip: str | None = None

    event_type: str
    state: str | None = None
    timestamp: datetime

    target_type: str | None = None

    attributes: EventAttributes | None = None

    bounding_boxes: list[BoundingBox] = Field(default_factory=list)

    selected_bounding_box: BoundingBox | None = None

    image: ImageData | None = None

    # Kept internally for debugging and processing.
    # Not part of the external normalized contract.
    extra_data: dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
    )


class RawCameraPackage(BaseModel):
    event_id: str
    received_at: datetime

    content_type: str
    camera_ip: str | None = None

    package_path: str
