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
    geometry: list[EventPoint] = Field(
        default_factory=list
    )

    raw_bounding_box: list[float] | None = None

    vendor_data: dict[str, Any] = Field(
        default_factory=dict
    )


class BoundingBox(BaseModel):
    origem: str | None = Field(
        default=None,
        serialization_alias="source",
    )

    x: int
    y: int

    largura: int = Field(
        serialization_alias="width",
    )

    altura: int = Field(
        serialization_alias="height",
    )

    x2: int
    y2: int

    proporcao_imagem: float | None = Field(
        default=None,
        serialization_alias="image_ratio",
    )


class ImageData(BaseModel):
    largura: int | None = Field(
        default=None,
        serialization_alias="width",
    )

    altura: int | None = Field(
        default=None,
        serialization_alias="height",
    )

    formato: str | None = Field(
        default=None,
        serialization_alias="format",
    )

    caminho_original: str | None = Field(
        default=None,
        serialization_alias="original_path",
    )

    caminho_marcada: str | None = Field(
        default=None,
        serialization_alias="annotated_path",
    )


class CameraEvent(BaseModel):
    schema_version: str = "1.0"

    evento_id: str = Field(
        serialization_alias="event_id",
    )

    fabricante: str = Field(
        serialization_alias="manufacturer",
    )

    modelo_camera: str | None = Field(
        default=None,
        serialization_alias="camera_model",
    )

    camera_id: str | None = None

    nome_camera: str | None = Field(
        default=None,
        serialization_alias="camera_name",
    )

    ip_camera: str | None = Field(
        default=None,
        serialization_alias="camera_ip",
    )

    tipo_evento: str = Field(
        serialization_alias="event_type",
    )

    estado: str | None = Field(
        default=None,
        serialization_alias="state",
    )

    data_hora: datetime = Field(
        serialization_alias="timestamp",
    )

    alvo_detectado: str | None = Field(
        default=None,
        serialization_alias="target_type",
    )

    attributes: EventAttributes | None = None

    bounding_boxes: list[BoundingBox] = Field(
        default_factory=list
    )

    bounding_box_escolhida: BoundingBox | None = Field(
        default=None,
        serialization_alias="selected_bounding_box",
    )

    imagem: ImageData | None = Field(
        default=None,
        serialization_alias="image",
    )

    # Mantido internamente para depuração e processamento.
    # Não faz parte do contrato normalizado externo.
    dados_extras: dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
    )


class RawCameraPackage(BaseModel):
    evento_id: str
    recebido_em: datetime

    content_type: str
    ip_camera: str | None = None

    caminho_pacote: str
