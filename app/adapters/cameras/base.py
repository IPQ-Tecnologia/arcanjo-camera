from abc import ABC, abstractmethod

from app.domain.models.camera_event import CameraEvent, RawCameraPackage


class CameraAdapter(ABC):
    """
    Base class for camera adapters.

    Each manufacturer must create a class that inherits from
    CameraAdapter and implements the methods below.
    """

    manufacturer: str

    @abstractmethod
    def can_handle(self, content_type: str, body: bytes) -> bool:
        """
        Checks whether the adapter recognizes the received package.

        Returns True when the package belongs to the manufacturer
        supported by this adapter.
        """
        raise NotImplementedError

    @abstractmethod
    def normalize(self, package: RawCameraPackage, body: bytes) -> CameraEvent:
        """Converts the manufacturer-specific package into the universal CameraEvent model."""
        raise NotImplementedError
