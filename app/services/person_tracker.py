import asyncio
import math
import time
import uuid
from dataclasses import dataclass
from typing import Literal


TrackingStatus = Literal["entered", "updated", "suppressed", "exited"]


@dataclass(frozen=True)
class DetectionBox:
    x: int
    y: int
    largura: int
    altura: int

    @property
    def x2(self) -> int:
        return self.x + self.largura

    @property
    def y2(self) -> int:
        return self.y + self.altura

    @property
    def centro_x(self) -> float:
        return self.x + self.largura / 2

    @property
    def centro_y(self) -> float:
        return self.y + self.altura / 2

    @property
    def area(self) -> int:
        return max(0, self.largura) * max(0, self.altura)

    def iou(self, outra: "DetectionBox") -> float:
        """
        Calcula a sobreposição entre duas caixas.

        0.0 significa nenhuma sobreposição.
        1.0 significa caixas iguais.
        """
        intersecao_x1 = max(self.x, outra.x)
        intersecao_y1 = max(self.y, outra.y)
        intersecao_x2 = min(self.x2, outra.x2)
        intersecao_y2 = min(self.y2, outra.y2)

        largura_intersecao = max(0, intersecao_x2 - intersecao_x1)
        altura_intersecao = max(0, intersecao_y2 - intersecao_y1)
        area_intersecao = largura_intersecao * altura_intersecao
        area_uniao = self.area + outra.area - area_intersecao

        if area_uniao <= 0:
            return 0.0

        return area_intersecao / area_uniao

    def distancia_centros(self, outra: "DetectionBox") -> float:
        diferenca_x = self.centro_x - outra.centro_x
        diferenca_y = self.centro_y - outra.centro_y

        return math.hypot(diferenca_x, diferenca_y)


@dataclass
class TrackedPerson:
    pessoa_id: str
    camera: str

    primeira_deteccao: float
    ultima_deteccao: float
    ultimo_processamento: float

    bbox: DetectionBox
    quantidade_deteccoes: int
    ultimo_evento_id: str


@dataclass(frozen=True)
class TrackingDecision:
    pessoa_id: str
    camera: str
    evento_id: str

    status: TrackingStatus
    deve_processar: bool

    quantidade_deteccoes: int
    bbox: DetectionBox


class PersonTracker:
    def __init__(
        self,
        intervalo_reprocessamento: float = 5.0,
        tempo_para_saida: float = 15.0,
        limite_iou: float = 0.25,
        limite_distancia: float = 0.75,
        janela_continuidade_flexivel: float = 4.0,
        limite_distancia_flexivel: float = 2.5,
        limite_razao_area_flexivel: float = 3.0,
        limite_velocidade_normalizada: float = 1.25,
    ) -> None:
        if intervalo_reprocessamento <= 0:
            raise ValueError("intervalo_reprocessamento deve ser positivo")

        if tempo_para_saida <= 0:
            raise ValueError("tempo_para_saida deve ser positivo")

        if limite_iou < 0:
            raise ValueError("limite_iou não pode ser negativo")

        if limite_distancia < 0:
            raise ValueError("limite_distancia não pode ser negativo")

        if janela_continuidade_flexivel <= 0:
            raise ValueError("janela_continuidade_flexivel deve ser positiva")

        if limite_distancia_flexivel < limite_distancia:
            raise ValueError(
                "limite_distancia_flexivel deve ser maior ou igual ao limite_distancia"
            )

        if limite_razao_area_flexivel < 1:
            raise ValueError("limite_razao_area_flexivel deve ser maior ou igual a 1")

        if limite_velocidade_normalizada <= 0:
            raise ValueError("limite_velocidade_normalizada deve ser positivo")

        self.intervalo_reprocessamento = intervalo_reprocessamento
        self.tempo_para_saida = tempo_para_saida
        self.limite_iou = limite_iou
        self.limite_distancia = limite_distancia
        self.janela_continuidade_flexivel = janela_continuidade_flexivel
        self.limite_distancia_flexivel = limite_distancia_flexivel
        self.limite_razao_area_flexivel = limite_razao_area_flexivel
        self.limite_velocidade_normalizada = limite_velocidade_normalizada

        self._pessoas: dict[str, TrackedPerson] = {}
        self._lock = asyncio.Lock()

    @property
    def quantidade_ativas(self) -> int:
        return len(self._pessoas)

    async def registrar(
        self,
        camera: str,
        evento_id: str,
        bbox: DetectionBox,
        agora: float | None = None,
    ) -> TrackingDecision:
        """Mantém compatibilidade com chamadas que enviam somente uma bounding box."""
        decisoes = await self.registrar_lote(
            camera=camera,
            evento_id=evento_id,
            bboxes=[bbox],
            agora=agora,
        )

        return decisoes[0]

    async def registrar_lote(
        self,
        camera: str,
        evento_id: str,
        bboxes: list[DetectionBox],
        agora: float | None = None,
    ) -> list[TrackingDecision]:
        """
        Registra todas as pessoas encontradas no mesmo quadro.

        Cada pessoa ativa pode ser associada somente a uma bounding box
        do quadro.
        """
        if not bboxes:
            return []

        for bbox in bboxes:
            if bbox.largura <= 0 or bbox.altura <= 0:
                raise ValueError(
                    "Todas as bounding boxes devem possuir largura e altura positivas"
                )

        momento = time.monotonic() if agora is None else agora

        async with self._lock:
            associacoes = self._associar_deteccoes_em_lote(camera=camera, bboxes=bboxes)
            self._aplicar_continuidade_flexivel(
                camera=camera,
                bboxes=bboxes,
                associacoes=associacoes,
                agora=momento,
            )

            total_deteccoes = len(bboxes)
            decisoes: list[TrackingDecision] = []

            for indice, bbox in enumerate(bboxes):
                evento_individual_id = self._criar_evento_individual_id(
                    evento_id=evento_id,
                    indice=indice,
                    total=total_deteccoes,
                )

                pessoa = associacoes.get(indice)

                if pessoa is None:
                    pessoa = self._criar_pessoa(
                        camera=camera,
                        evento_id=evento_individual_id,
                        bbox=bbox,
                        agora=momento,
                    )
                    decisoes.append(
                        TrackingDecision(
                            pessoa_id=pessoa.pessoa_id,
                            camera=camera,
                            evento_id=evento_individual_id,
                            status="entered",
                            deve_processar=True,
                            quantidade_deteccoes=1,
                            bbox=bbox,
                        )
                    )
                    continue

                decisao = self._atualizar_pessoa(
                    pessoa=pessoa,
                    evento_id=evento_individual_id,
                    bbox=bbox,
                    agora=momento,
                )
                decisoes.append(decisao)

            return decisoes

    async def coletar_saidas(self, agora: float | None = None) -> list[TrackingDecision]:
        momento = time.monotonic() if agora is None else agora

        async with self._lock:
            pessoas_encerradas: list[TrackingDecision] = []
            ids_para_remover: list[str] = []

            for pessoa_id, pessoa in self._pessoas.items():
                tempo_sem_deteccao = momento - pessoa.ultima_deteccao
                if tempo_sem_deteccao < self.tempo_para_saida:
                    continue

                pessoas_encerradas.append(
                    TrackingDecision(
                        pessoa_id=pessoa.pessoa_id,
                        camera=pessoa.camera,
                        evento_id=pessoa.ultimo_evento_id,
                        status="exited",
                        deve_processar=False,
                        quantidade_deteccoes=pessoa.quantidade_deteccoes,
                        bbox=pessoa.bbox,
                    )
                )
                ids_para_remover.append(pessoa_id)

            for pessoa_id in ids_para_remover:
                self._pessoas.pop(pessoa_id, None)

            return pessoas_encerradas

    async def limpar(self) -> None:
        async with self._lock:
            self._pessoas.clear()

    def _associar_deteccoes_em_lote(
        self,
        camera: str,
        bboxes: list[DetectionBox],
    ) -> dict[int, TrackedPerson]:
        """
        Busca as melhores associações entre as bounding boxes e as
        pessoas ativas.

        Uma pessoa não pode ser usada duas vezes dentro do mesmo
        quadro.
        """
        candidatos: list[tuple[float, int, str]] = []

        for indice, bbox in enumerate(bboxes):
            for pessoa in self._pessoas.values():
                if pessoa.camera != camera:
                    continue

                pontuacao = self._calcular_pontuacao(bbox=bbox, pessoa=pessoa)
                if pontuacao is None:
                    continue

                candidatos.append((pontuacao, indice, pessoa.pessoa_id))

        candidatos.sort(key=lambda item: (-item[0], item[1], item[2]))

        deteccoes_utilizadas: set[int] = set()
        pessoas_utilizadas: set[str] = set()
        associacoes: dict[int, TrackedPerson] = {}

        for _, indice, pessoa_id in candidatos:
            if indice in deteccoes_utilizadas:
                continue

            if pessoa_id in pessoas_utilizadas:
                continue

            pessoa = self._pessoas.get(pessoa_id)
            if pessoa is None:
                continue

            associacoes[indice] = pessoa
            deteccoes_utilizadas.add(indice)
            pessoas_utilizadas.add(pessoa_id)

        return associacoes

    def _aplicar_continuidade_flexivel(
        self,
        camera: str,
        bboxes: list[DetectionBox],
        associacoes: dict[int, TrackedPerson],
        agora: float,
    ) -> None:
        """
        Corrige o caso em que uma pessoa atravessa rapidamente a
        imagem e a nova caixa fica distante da caixa anterior.

        Esta regra somente é aplicada quando:

        - existe uma única bounding box;
        - existe uma única pessoa ativa na câmera;
        - a associação normal falhou;
        - o intervalo entre as imagens é curto;
        - o tamanho das caixas continua plausível.

        Assim, a regra não interfere no rastreamento em lote de várias
        pessoas.
        """
        if len(bboxes) != 1:
            return

        if associacoes:
            return

        pessoas_camera = [
            pessoa for pessoa in self._pessoas.values() if pessoa.camera == camera
        ]
        if len(pessoas_camera) != 1:
            return

        pessoa = pessoas_camera[0]
        bbox = bboxes[0]

        continuidade_valida = self._pode_usar_continuidade_flexivel(
            pessoa=pessoa,
            bbox=bbox,
            agora=agora,
        )
        if continuidade_valida:
            associacoes[0] = pessoa

    def _pode_usar_continuidade_flexivel(
        self,
        pessoa: TrackedPerson,
        bbox: DetectionBox,
        agora: float,
    ) -> bool:
        tempo_sem_deteccao = max(0.0, agora - pessoa.ultima_deteccao)
        if tempo_sem_deteccao > self.janela_continuidade_flexivel:
            return False

        menor_area = max(1, min(bbox.area, pessoa.bbox.area))
        maior_area = max(bbox.area, pessoa.bbox.area)
        razao_area = maior_area / menor_area
        if razao_area > self.limite_razao_area_flexivel:
            return False

        distancia = bbox.distancia_centros(pessoa.bbox)
        maior_dimensao = max(
            bbox.largura, bbox.altura, pessoa.bbox.largura, pessoa.bbox.altura, 1
        )
        distancia_normalizada = distancia / maior_dimensao

        limite_dinamico = min(
            self.limite_distancia_flexivel,
            max(self.limite_distancia, 1.25 + tempo_sem_deteccao * 0.45),
        )
        if distancia_normalizada > limite_dinamico:
            return False

        velocidade_normalizada = distancia_normalizada / max(tempo_sem_deteccao, 0.25)
        if velocidade_normalizada > self.limite_velocidade_normalizada:
            return False

        return True

    def _calcular_pontuacao(
        self,
        bbox: DetectionBox,
        pessoa: TrackedPerson,
    ) -> float | None:
        sobreposicao = bbox.iou(pessoa.bbox)
        distancia = bbox.distancia_centros(pessoa.bbox)
        maior_dimensao = max(
            bbox.largura, bbox.altura, pessoa.bbox.largura, pessoa.bbox.altura, 1
        )
        distancia_normalizada = distancia / maior_dimensao

        caixas_proximas = distancia_normalizada <= self.limite_distancia
        caixas_sobrepostas = sobreposicao >= self.limite_iou
        if not (caixas_proximas or caixas_sobrepostas):
            return None

        pontuacao_distancia = max(0.0, 1.0 - distancia_normalizada)

        return sobreposicao + pontuacao_distancia

    def _atualizar_pessoa(
        self,
        pessoa: TrackedPerson,
        evento_id: str,
        bbox: DetectionBox,
        agora: float,
    ) -> TrackingDecision:
        pessoa.ultima_deteccao = agora
        pessoa.bbox = bbox
        pessoa.ultimo_evento_id = evento_id
        pessoa.quantidade_deteccoes += 1

        tempo_desde_processamento = agora - pessoa.ultimo_processamento

        if tempo_desde_processamento >= self.intervalo_reprocessamento:
            pessoa.ultimo_processamento = agora

            return TrackingDecision(
                pessoa_id=pessoa.pessoa_id,
                camera=pessoa.camera,
                evento_id=evento_id,
                status="updated",
                deve_processar=True,
                quantidade_deteccoes=pessoa.quantidade_deteccoes,
                bbox=bbox,
            )

        return TrackingDecision(
            pessoa_id=pessoa.pessoa_id,
            camera=pessoa.camera,
            evento_id=evento_id,
            status="suppressed",
            deve_processar=False,
            quantidade_deteccoes=pessoa.quantidade_deteccoes,
            bbox=bbox,
        )

    def _encontrar_pessoa(
        self,
        camera: str,
        bbox: DetectionBox,
    ) -> TrackedPerson | None:
        """Mantido para compatibilidade com testes individuais."""
        melhor_pessoa: TrackedPerson | None = None
        melhor_pontuacao = -1.0

        for pessoa in self._pessoas.values():
            if pessoa.camera != camera:
                continue

            pontuacao = self._calcular_pontuacao(bbox=bbox, pessoa=pessoa)
            if pontuacao is None:
                continue

            if pontuacao > melhor_pontuacao:
                melhor_pontuacao = pontuacao
                melhor_pessoa = pessoa

        return melhor_pessoa

    def _criar_pessoa(
        self,
        camera: str,
        evento_id: str,
        bbox: DetectionBox,
        agora: float,
    ) -> TrackedPerson:
        camera_normalizada = "".join(
            caractere.lower() if caractere.isalnum() else "-" for caractere in camera
        ).strip("-")

        if not camera_normalizada:
            camera_normalizada = "camera"

        pessoa_id = f"{camera_normalizada}-{uuid.uuid4().hex[:8]}"

        pessoa = TrackedPerson(
            pessoa_id=pessoa_id,
            camera=camera,
            primeira_deteccao=agora,
            ultima_deteccao=agora,
            ultimo_processamento=agora,
            bbox=bbox,
            quantidade_deteccoes=1,
            ultimo_evento_id=evento_id,
        )

        self._pessoas[pessoa_id] = pessoa

        return pessoa

    def _criar_evento_individual_id(self, evento_id: str, indice: int, total: int) -> str:
        """
        Um quadro com uma pessoa mantém o ID original.

        Quadros com várias pessoas recebem IDs como:

        evento123-01
        evento123-02
        """
        if total <= 1:
            return evento_id

        return f"{evento_id}-{indice + 1:02d}"


person_tracker = PersonTracker(
    intervalo_reprocessamento=5.0,
    tempo_para_saida=15.0,
    limite_iou=0.25,
    limite_distancia=0.75,
    janela_continuidade_flexivel=12.0,
    limite_distancia_flexivel=2.5,
    limite_razao_area_flexivel=12.0,
    limite_velocidade_normalizada=1.25,
)
