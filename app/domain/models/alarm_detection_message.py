from pydantic import BaseModel

from app.domain.models.camera_event import (
    EventAttributes,
)


class AlarmDevice(BaseModel):
    name: str


class AlarmImages(BaseModel):
    background: str
    detection: str


class AlarmEventData(BaseModel):
    id: str
    type: str
    timestamp: int
    datetime: str

    attributes: EventAttributes | None = None

    images: AlarmImages


class AlarmDetectionMessage(BaseModel):
    device: AlarmDevice
    event: AlarmEventData
