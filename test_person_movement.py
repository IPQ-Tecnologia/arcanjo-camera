import asyncio

from app.services.person_movement import (
    PersonMovementMemory,
)
from app.services.person_tracker import (
    DetectionBox,
)


def criar_box(
    x: int,
    y: int,
    largura: int,
    altura: int,
) -> DetectionBox:
    return DetectionBox(
        x=x,
        y=y,
        largura=largura,
        altura=altura,
    )


async def executar_teste() -> None:
    memoria = PersonMovementMemory()

    pessoa_id = "camera-01-teste"

    primeira = await memoria.registrar(
        pessoa_id=pessoa_id,
        bbox=criar_box(
            x=100,
            y=100,
            largura=100,
            altura=200,
        ),
        agora=100.0,
    )

    assert (
        primeira.movimento_horizontal
        == "inicial"
    )

    assert (
        primeira.quantidade_amostras
        == 1
    )

    segunda = await memoria.registrar(
        pessoa_id=pessoa_id,
        bbox=criar_box(
            x=150,
            y=100,
            largura=100,
            altura=200,
        ),
        agora=102.0,
    )

    assert (
        segunda.movimento_horizontal
        == "direita"
    )

    assert (
        segunda.movimento_vertical
        == "parado"
    )

    assert (
        segunda.tendencia_distancia
        == "estavel"
    )

    assert (
        segunda.velocidade_pixels_segundo
        == 25.0
    )

    terceira = await memoria.registrar(
        pessoa_id=pessoa_id,
        bbox=criar_box(
            x=135,
            y=60,
            largura=130,
            altura=240,
        ),
        agora=104.0,
    )

    assert (
        terceira.movimento_horizontal
        == "parado"
    )

    assert (
        terceira.movimento_vertical
        == "cima"
    )

    assert (
        terceira.tendencia_distancia
        == "aproximando"
    )

    quarta = await memoria.registrar(
        pessoa_id=pessoa_id,
        bbox=criar_box(
            x=80,
            y=60,
            largura=90,
            altura=180,
        ),
        agora=106.0,
    )

    assert (
        quarta.movimento_horizontal
        == "esquerda"
    )

    assert (
        quarta.tendencia_distancia
        == "afastando"
    )

    assert (
        quarta.quantidade_amostras
        == 4
    )

    assert (
        quarta.tempo_observado_segundos
        == 6.0
    )

    assert (
        quarta.distancia_total_pixels
        > 0
    )

    final = await memoria.finalizar(
        pessoa_id
    )

    assert final is not None

    assert (
        memoria.quantidade_pessoas
        == 0
    )

    print(
        "===== ANÁLISE DE MOVIMENTO ====="
    )

    print(
        "Primeira:",
        primeira.to_dict(),
    )

    print(
        "Segunda:",
        segunda.to_dict(),
    )

    print(
        "Terceira:",
        terceira.to_dict(),
    )

    print(
        "Quarta:",
        quarta.to_dict(),
    )

    print(
        "\nTeste concluído com sucesso."
    )


if __name__ == "__main__":
    asyncio.run(
        executar_teste()
    )