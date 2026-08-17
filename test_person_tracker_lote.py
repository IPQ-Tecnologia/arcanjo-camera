import asyncio

from app.services.person_tracker import (
    DetectionBox,
    PersonTracker,
)


def box(
    x: int,
    y: int,
    largura: int = 100,
    altura: int = 220,
) -> DetectionBox:
    return DetectionBox(
        x=x,
        y=y,
        largura=largura,
        altura=altura,
    )


async def executar_teste() -> None:
    tracker = PersonTracker(
        intervalo_reprocessamento=5.0,
        tempo_para_saida=15.0,
        limite_iou=0.25,
        limite_distancia=0.75,
    )

    # Primeiro quadro: duas pessoas novas.
    primeiro_quadro = (
        await tracker.registrar_lote(
            camera="Camera 01",
            evento_id="quadro-001",
            bboxes=[
                box(100, 120),
                box(600, 120),
            ],
            agora=100.0,
        )
    )

    assert len(primeiro_quadro) == 2

    assert all(
        decisao.status == "entered"
        for decisao in primeiro_quadro
    )

    pessoa_esquerda_id = (
        primeiro_quadro[0].pessoa_id
    )

    pessoa_direita_id = (
        primeiro_quadro[1].pessoa_id
    )

    assert (
        pessoa_esquerda_id
        != pessoa_direita_id
    )

    # Segundo quadro: ordem invertida.
    #
    # A pessoa da direita aparece primeiro na lista,
    # mas deverá manter seu ID correto.
    segundo_quadro = (
        await tracker.registrar_lote(
            camera="Camera 01",
            evento_id="quadro-002",
            bboxes=[
                box(610, 125),
                box(110, 125),
            ],
            agora=101.0,
        )
    )

    assert len(segundo_quadro) == 2

    assert (
        segundo_quadro[0].pessoa_id
        == pessoa_direita_id
    )

    assert (
        segundo_quadro[1].pessoa_id
        == pessoa_esquerda_id
    )

    assert all(
        decisao.status == "suppressed"
        for decisao in segundo_quadro
    )

    assert (
        segundo_quadro[0].pessoa_id
        != segundo_quadro[1].pessoa_id
    )

    # Terceiro quadro: passaram mais de cinco
    # segundos desde o último processamento.
    terceiro_quadro = (
        await tracker.registrar_lote(
            camera="Camera 01",
            evento_id="quadro-003",
            bboxes=[
                box(120, 130),
                box(620, 130),
            ],
            agora=106.5,
        )
    )

    assert all(
        decisao.status == "updated"
        for decisao in terceiro_quadro
    )

    assert (
        terceiro_quadro[0].pessoa_id
        == pessoa_esquerda_id
    )

    assert (
        terceiro_quadro[1].pessoa_id
        == pessoa_direita_id
    )

    assert all(
        decisao.quantidade_deteccoes == 3
        for decisao in terceiro_quadro
    )

    # Após mais de 15 segundos, as duas pessoas
    # deverão sair.
    saidas = await tracker.coletar_saidas(
        agora=122.0
    )

    assert len(saidas) == 2

    assert all(
        decisao.status == "exited"
        for decisao in saidas
    )

    ids_saida = {
        decisao.pessoa_id
        for decisao in saidas
    }

    assert ids_saida == {
        pessoa_esquerda_id,
        pessoa_direita_id,
    }

    assert tracker.quantidade_ativas == 0

    print(
        "===== RASTREAMENTO EM LOTE ====="
    )

    print(
        "Pessoa esquerda:",
        pessoa_esquerda_id,
    )

    print(
        "Pessoa direita:",
        pessoa_direita_id,
    )

    print(
        "Primeiro quadro:",
        [
            decisao.status
            for decisao in primeiro_quadro
        ],
    )

    print(
        "Segundo quadro:",
        [
            decisao.status
            for decisao in segundo_quadro
        ],
    )

    print(
        "Terceiro quadro:",
        [
            decisao.status
            for decisao in terceiro_quadro
        ],
    )

    print(
        "Saídas:",
        len(saidas),
    )

    print(
        "\nTeste concluído com sucesso."
    )


if __name__ == "__main__":
    asyncio.run(
        executar_teste()
    )