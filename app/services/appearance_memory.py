import asyncio
from collections import Counter
from dataclasses import asdict, dataclass, field

from app.services.scene_analyzer import (
    PersonVisualAnalysis,
)


@dataclass(frozen=True)
class StableAppearance:
    cor_roupa_predominante: str

    rgb_medio: tuple[
        int,
        int,
        int,
    ]

    posicao_atual: str
    tamanho_predominante: str
    percentual_medio_quadro: float
    quantidade_amostras: int
    descricao: str

    def to_dict(self) -> dict:
        dados = asdict(self)

        dados["rgb_medio"] = list(
            self.rgb_medio
        )

        return dados


@dataclass
class _AppearanceState:
    cores: Counter[str] = field(
        default_factory=Counter
    )

    tamanhos: Counter[str] = field(
        default_factory=Counter
    )

    soma_vermelho: int = 0
    soma_verde: int = 0
    soma_azul: int = 0

    soma_percentual: float = 0.0
    quantidade_amostras: int = 0

    ultima_cor: str = "indefinida"

    # Cor apresentada como resultado estável.
    # É separada da última leitura para evitar
    # alternância causada por iluminação e sombra.
    cor_estavel: str = "indefinida"

    ultima_posicao: str = "centro"
    ultimo_tamanho: str = "medio"


class AppearanceMemory:
    def __init__(self) -> None:
        self._sessoes: dict[
            str,
            _AppearanceState,
        ] = {}

        self._lock = asyncio.Lock()

    @property
    def quantidade_sessoes(self) -> int:
        return len(self._sessoes)

    async def registrar(
        self,
        pessoa_id: str,
        analise: PersonVisualAnalysis,
    ) -> StableAppearance:
        if not pessoa_id:
            raise ValueError(
                "pessoa_id não pode ser vazio"
            )

        async with self._lock:
            estado = self._sessoes.get(
                pessoa_id
            )

            if estado is None:
                estado = _AppearanceState()

                self._sessoes[
                    pessoa_id
                ] = estado

            estado.cores[
                analise.cor_roupa_aproximada
            ] += 1

            estado.tamanhos[
                analise.tamanho_no_quadro
            ] += 1

            vermelho, verde, azul = (
                analise.rgb_representativo
            )

            estado.soma_vermelho += vermelho
            estado.soma_verde += verde
            estado.soma_azul += azul

            estado.soma_percentual += (
                analise.percentual_quadro
            )

            estado.quantidade_amostras += 1

            estado.ultima_cor = (
                analise.cor_roupa_aproximada
            )

            estado.ultima_posicao = (
                analise.posicao_horizontal
            )

            estado.ultimo_tamanho = (
                analise.tamanho_no_quadro
            )

            return self._montar_resultado(
                estado
            )

    async def obter(
        self,
        pessoa_id: str,
    ) -> StableAppearance | None:
        async with self._lock:
            estado = self._sessoes.get(
                pessoa_id
            )

            if estado is None:
                return None

            return self._montar_resultado(
                estado
            )

    async def finalizar(
        self,
        pessoa_id: str,
    ) -> StableAppearance | None:
        """
        Retorna a aparência final e remove a sessão
        daquela pessoa da memória.
        """

        async with self._lock:
            estado = self._sessoes.pop(
                pessoa_id,
                None,
            )

            if estado is None:
                return None

            return self._montar_resultado(
                estado
            )

    async def limpar(self) -> None:
        async with self._lock:
            self._sessoes.clear()

    def _montar_resultado(
        self,
        estado: _AppearanceState,
    ) -> StableAppearance:
        quantidade = max(
            1,
            estado.quantidade_amostras,
        )

        cor_calculada = (
            self._selecionar_cor_predominante(
                estado
            )
        )

        maior_quantidade_cor = max(
            estado.cores.values(),
            default=0,
        )

        # Confirma a cor depois de duas leituras
        # iguais. Com mais amostras, exige pelo menos
        # 60% de concordância.
        cor_consistente = (
            quantidade >= 2
            and maior_quantidade_cor >= 2
            and (
                maior_quantidade_cor * 5
                >= quantidade * 3
            )
        )

        # A roupa não muda durante uma passagem curta.
        # Depois que a cor fica estável, leituras
        # isoladas diferentes não apagam o resultado.
        if (
            estado.cor_estavel
            != "indefinida"
        ):
            cor_predominante = (
                estado.cor_estavel
            )

        elif cor_consistente:
            cor_predominante = (
                cor_calculada
            )

            estado.cor_estavel = (
                cor_predominante
            )

        else:
            cor_predominante = (
                "indefinida"
            )

        tamanho_predominante = (
            self._selecionar_predominante(
                contagem=estado.tamanhos,
                valor_mais_recente=(
                    estado.ultimo_tamanho
                ),
            )
        )

        rgb_medio = (
            round(
                estado.soma_vermelho
                / quantidade
            ),
            round(
                estado.soma_verde
                / quantidade
            ),
            round(
                estado.soma_azul
                / quantidade
            ),
        )

        percentual_medio = round(
            estado.soma_percentual
            / quantidade,
            2,
        )

        descricao = self._montar_descricao(
            cor=cor_predominante,
            posicao=estado.ultima_posicao,
            tamanho=tamanho_predominante,
            quantidade=quantidade,
        )

        return StableAppearance(
            cor_roupa_predominante=(
                cor_predominante
            ),
            rgb_medio=rgb_medio,
            posicao_atual=(
                estado.ultima_posicao
            ),
            tamanho_predominante=(
                tamanho_predominante
            ),
            percentual_medio_quadro=(
                percentual_medio
            ),
            quantidade_amostras=quantidade,
            descricao=descricao,
        )

    def _selecionar_cor_predominante(
        self,
        estado: _AppearanceState,
    ) -> str:
        """
        Estabiliza a classificação de roupas escuras
        que alternam entre preta e uma cor escura
        devido à iluminação, distância ou sombra.
        """

        contagem = estado.cores

        if not contagem:
            return estado.ultima_cor

        resultado_padrao = (
            self._selecionar_predominante(
                contagem=contagem,
                valor_mais_recente=(
                    estado.ultima_cor
                ),
            )
        )

        cores_escuras = (
            "azul-escura",
            "verde-escura",
            "vermelha-escura",
            "roxa-escura",
        )

        quantidade_preta = contagem.get(
            "preta",
            0,
        )

        candidatas = [
            (
                cor,
                contagem.get(cor, 0),
            )
            for cor in cores_escuras
            if contagem.get(cor, 0) > 0
        ]

        if (
            quantidade_preta == 0
            or not candidatas
        ):
            return resultado_padrao

        maior_quantidade_escura = max(
            quantidade
            for _, quantidade in candidatas
        )

        empatadas = [
            cor
            for cor, quantidade in candidatas
            if (
                quantidade
                == maior_quantidade_escura
            )
        ]

        if estado.cor_estavel in empatadas:
            cor_escura = estado.cor_estavel
        elif estado.ultima_cor in empatadas:
            cor_escura = estado.ultima_cor
        else:
            cor_escura = empatadas[0]

        cores_do_conflito = (
            set(cores_escuras)
            | {"preta"}
        )

        maior_outra_cor = max(
            (
                quantidade
                for cor, quantidade
                in contagem.items()
                if cor not in cores_do_conflito
            ),
            default=0,
        )

        if maior_outra_cor > max(
            quantidade_preta,
            maior_quantidade_escura,
        ):
            return resultado_padrao

        # Já estabilizou em azul-escura, verde-escura
        # ou outra cor escura. Um quadro preto isolado
        # não deve alterar imediatamente o resultado.
        if estado.cor_estavel in cores_escuras:
            cor_estavel = estado.cor_estavel

            quantidade_cor_estavel = (
                contagem.get(
                    cor_estavel,
                    0,
                )
            )

            # Permite mudar entre cores escuras apenas
            # quando a nova cor estiver duas amostras
            # à frente da atual.
            if (
                cor_escura != cor_estavel
                and maior_quantidade_escura
                >= quantidade_cor_estavel + 2
            ):
                cor_estavel = cor_escura
                quantidade_cor_estavel = (
                    maior_quantidade_escura
                )

            # Volta para preta somente quando as
            # leituras pretas estiverem duas amostras
            # à frente da cor escura estabilizada.
            if (
                quantidade_preta
                >= quantidade_cor_estavel + 2
            ):
                return "preta"

            return cor_estavel

        # Estava estabilizado como preto. Exige duas
        # leituras da cor escura e empate ou vantagem
        # para realizar a mudança.
        if estado.cor_estavel == "preta":
            if (
                maior_quantidade_escura >= 2
                and maior_quantidade_escura
                >= quantidade_preta
            ):
                return cor_escura

            return "preta"

        # Estado inicial, antes de uma cor estável.
        if (
            maior_quantidade_escura >= 2
            and maior_quantidade_escura
            >= quantidade_preta
        ):
            return cor_escura

        return resultado_padrao


    @staticmethod
    def _selecionar_predominante(
        contagem: Counter[str],
        valor_mais_recente: str,
    ) -> str:
        if not contagem:
            return valor_mais_recente

        maior_quantidade = max(
            contagem.values()
        )

        empatados = [
            valor
            for valor, quantidade
            in contagem.items()
            if quantidade == maior_quantidade
        ]

        # Em caso de empate, utiliza o valor
        # observado mais recentemente.
        if valor_mais_recente in empatados:
            return valor_mais_recente

        return empatados[0]

    @staticmethod
    def _montar_descricao(
        cor: str,
        posicao: str,
        tamanho: str,
        quantidade: int,
    ) -> str:
        posicoes = {
            "esquerda": "à esquerda",
            "centro": "no centro",
            "direita": "à direita",
        }

        tamanhos = {
            "pequeno": "pequeno",
            "medio": "médio",
            "grande": "grande",
        }

        posicao_formatada = posicoes.get(
            posicao,
            posicao,
        )

        tamanho_formatado = tamanhos.get(
            tamanho,
            tamanho,
        )

        return (
            "Pessoa com roupa predominantemente "
            f"{cor}, localizada atualmente "
            f"{posicao_formatada} da cena, "
            f"com tamanho aparente "
            f"{tamanho_formatado}. "
            f"Resultado baseado em "
            f"{quantidade} amostra(s)."
        )


appearance_memory = AppearanceMemory()