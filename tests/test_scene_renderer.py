import base64
from pathlib import Path

from PIL import Image

from app.domain.models.camera_event import (
    BoundingBox,
)
from app.services.scene_renderer import (
    renderizar_cena_com_boxes,
)


def localizar_imagem() -> Path:
    """
    Procura a imagem original mais recente.

    Imagens que já possuem marcações são ignoradas
    para evitar desenhar novas bounding boxes sobre
    retângulos antigos.
    """

    pastas = [
        Path("imagens_eventos"),
        Path("eventos_selecionados"),
    ]

    imagens_originais: list[Path] = []

    for pasta in pastas:
        if not pasta.exists():
            continue

        candidatos = [
            *pasta.rglob("*.jpg"),
            *pasta.rglob("*.jpeg"),
        ]

        for caminho in candidatos:
            nome = caminho.name.lower()

            # Ignora qualquer imagem que já tenha
            # bounding boxes desenhadas.
            if "_marcada" in nome:
                continue

            # Ignora a saída produzida por este teste.
            if nome == "teste_cena_marcada.jpg":
                continue

            imagens_originais.append(
                caminho
            )

    if not imagens_originais:
        raise FileNotFoundError(
            "Nenhuma imagem original JPG foi "
            "encontrada em imagens_eventos ou "
            "eventos_selecionados."
        )

    return max(
        imagens_originais,
        key=lambda caminho: (
            caminho.stat().st_mtime
        ),
    )


def criar_bounding_box(
    x: int,
    y: int,
    largura: int,
    altura: int,
    origem: str,
) -> BoundingBox:
    """
    Cria uma bounding box para o teste.
    """

    return BoundingBox(
        origem=origem,
        x=x,
        y=y,
        largura=largura,
        altura=altura,
        x2=x + largura,
        y2=y + altura,
        proporcao_imagem=False,
    )


def main() -> None:
    caminho_imagem = localizar_imagem()

    with Image.open(
        caminho_imagem
    ) as imagem:
        largura_imagem = imagem.width
        altura_imagem = imagem.height

    pessoa_1_x = int(
        largura_imagem * 0.12
    )

    pessoa_1_y = int(
        altura_imagem * 0.18
    )

    pessoa_1_largura = int(
        largura_imagem * 0.18
    )

    pessoa_1_altura = int(
        altura_imagem * 0.65
    )

    caixas = [
        # Pessoa 1.
        criar_bounding_box(
            x=pessoa_1_x,
            y=pessoa_1_y,
            largura=pessoa_1_largura,
            altura=pessoa_1_altura,
            origem="teste_pessoa_1",
        ),

        # Pessoa 2.
        criar_bounding_box(
            x=int(
                largura_imagem * 0.48
            ),
            y=int(
                altura_imagem * 0.22
            ),
            largura=int(
                largura_imagem * 0.17
            ),
            altura=int(
                altura_imagem * 0.60
            ),
            origem="teste_pessoa_2",
        ),

        # Pessoa 3.
        criar_bounding_box(
            x=int(
                largura_imagem * 0.70
            ),
            y=int(
                altura_imagem * 0.20
            ),
            largura=int(
                largura_imagem * 0.16
            ),
            altura=int(
                altura_imagem * 0.62
            ),
            origem="teste_pessoa_3",
        ),

        # Esta caixa representa a imagem inteira.
        # O renderizador deverá ignorá-la.
        criar_bounding_box(
            x=0,
            y=0,
            largura=largura_imagem,
            altura=altura_imagem,
            origem="imagem_inteira",
        ),

        # Esta caixa é igual à primeira.
        # O renderizador deverá ignorá-la por ser
        # uma duplicata.
        criar_bounding_box(
            x=pessoa_1_x,
            y=pessoa_1_y,
            largura=pessoa_1_largura,
            altura=pessoa_1_altura,
            origem="duplicada",
        ),
    ]

    resultado = (
        renderizar_cena_com_boxes(
            caminho_imagem=str(
                caminho_imagem
            ),
            bounding_boxes=caixas,
        )
    )

    caminho_saida = Path(
        "teste_cena_marcada.jpg"
    )

    bytes_imagem = base64.b64decode(
        resultado.imagem_base64
    )

    caminho_saida.write_bytes(
        bytes_imagem
    )

    assert (
        resultado.quantidade_boxes == 3
    ), (
        "Deveriam ser desenhadas exatamente "
        "três bounding boxes."
    )

    assert caminho_saida.is_file(), (
        "A imagem marcada não foi criada."
    )

    assert caminho_saida.stat().st_size > 0, (
        "A imagem marcada foi criada vazia."
    )

    assert (
        resultado.largura_imagem
        == largura_imagem
    ), (
        "A largura da imagem foi alterada."
    )

    assert (
        resultado.altura_imagem
        == altura_imagem
    ), (
        "A altura da imagem foi alterada."
    )

    print(
        "===== TESTE DO RENDERIZADOR ====="
    )

    print(
        "Imagem original:",
        caminho_imagem,
    )

    print(
        "Imagem marcada:",
        caminho_saida,
    )

    print(
        "Dimensões:",
        (
            f"{resultado.largura_imagem}"
            f"x{resultado.altura_imagem}"
        ),
    )

    print(
        "Bounding boxes desenhadas:",
        resultado.quantidade_boxes,
    )

    print(
        "Caracteres da imagem Base64:",
        len(
            resultado.imagem_base64
        ),
    )

    print(
        "Tamanho do arquivo:",
        caminho_saida.stat().st_size,
        "bytes",
    )

    print(
        "\nTeste concluído com sucesso."
    )


if __name__ == "__main__":
    main()