from abc import ABC, abstractmethod

from app.domain.models.camera_event import (
    CameraEvent,
    RawCameraPackage
)


class CameraAdapter(ABC):
    """
    Classe base para os adaptadores de câmeras.

    Cada fabricante deverá criar uma classe que herda
    CameraAdapter e implementa os métodos abaixo.
    """

    fabricante: str

    @abstractmethod
    def consegue_processar(
        self,
        content_type: str,
        body: bytes
    ) -> bool:
        """
        Verifica se o adaptador reconhece o pacote recebido.

        Retorna True quando o pacote pertence ao fabricante
        suportado pelo adaptador.
        """

        raise NotImplementedError

    @abstractmethod
    def normalizar(
        self,
        pacote: RawCameraPackage,
        body: bytes
    ) -> CameraEvent:
        """
        Converte o pacote específico do fabricante
        para o modelo universal CameraEvent.
        """

        raise NotImplementedError