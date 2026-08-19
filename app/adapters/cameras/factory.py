from app.adapters.cameras.base import CameraAdapter
from app.adapters.cameras.dahua import DahuaAdapter
from app.adapters.cameras.hikvision import HikvisionAdapter


class CameraAdapterFactory:
    """
    Registro central dos adapters de câmera.

    Cada fabricante implementa CameraAdapter e deve ser
    registrado apenas nesta classe.
    """

    def __init__(self) -> None:
        self._adapters: list[CameraAdapter] = [
            HikvisionAdapter(),
            DahuaAdapter(),
        ]

    def registrar_adapter(self, adapter: CameraAdapter) -> None:
        fabricantes = self.listar_fabricantes()
        if adapter.fabricante in fabricantes:
            raise ValueError(
                f"Adapter já registrado para o fabricante: {adapter.fabricante}"
            )

        self._adapters.append(adapter)

    def encontrar_adapter(self, content_type: str, body: bytes) -> CameraAdapter:
        for adapter in self._adapters:
            if adapter.consegue_processar(content_type=content_type, body=body):
                return adapter

        fabricantes = ", ".join(self.listar_fabricantes())
        raise ValueError(
            "Nenhum adaptador disponível para o pacote. "
            f"Content-Type={content_type!r}; adapters registrados=[{fabricantes}]"
        )

    def listar_fabricantes(self) -> list[str]:
        return [adapter.fabricante for adapter in self._adapters]


camera_adapter_factory = CameraAdapterFactory()
