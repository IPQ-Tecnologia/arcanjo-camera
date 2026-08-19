from app.adapters.cameras.base import CameraAdapter
from app.adapters.cameras.dahua import DahuaAdapter
from app.adapters.cameras.hikvision import HikvisionAdapter


class CameraAdapterFactory:
    """
    Central registry of camera adapters.

    Each manufacturer implements CameraAdapter and must be registered
    only in this class.
    """

    def __init__(self) -> None:
        self._adapters: list[CameraAdapter] = [
            HikvisionAdapter(),
            DahuaAdapter(),
        ]

    def register_adapter(self, adapter: CameraAdapter) -> None:
        manufacturers = self.list_manufacturers()
        if adapter.manufacturer in manufacturers:
            raise ValueError(
                f"Adapter already registered for manufacturer: {adapter.manufacturer}"
            )

        self._adapters.append(adapter)

    def find_adapter(self, content_type: str, body: bytes) -> CameraAdapter:
        for adapter in self._adapters:
            if adapter.can_handle(content_type=content_type, body=body):
                return adapter

        manufacturers = ", ".join(self.list_manufacturers())
        raise ValueError(
            "No adapter available for this package. "
            f"Content-Type={content_type!r}; registered adapters=[{manufacturers}]"
        )

    def list_manufacturers(self) -> list[str]:
        return [adapter.manufacturer for adapter in self._adapters]


camera_adapter_factory = CameraAdapterFactory()
