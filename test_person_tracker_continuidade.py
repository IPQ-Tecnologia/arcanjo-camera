import asyncio

from app.services.person_tracker import (
    DetectionBox,
    PersonTracker,
)


def box(
    x: int,
    y: int,
    largura: int = 120,
    altura: int = 260,
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
        janela_continuidade_flexivel=4.0,
        limite_distancia_flexivel=2.5,
        limite_razao_area_flexivel=3.0,
        limite_velocidade_normalizada=1.25,
    )

    primeira = await tracker.registrar_lote(
        camera="Camera 01",
        evento_id="evento-001",
        bboxes=[
            box(
                x=520,
                y=180,
            ),
        ],
        agora=100.0,
    )

    pessoa_id = primeira[0].pessoa_id

    assert primeira[0].status == "entered"

    segunda = await tracker.registrar_lote(
        camera="Camera 01",
        evento_id="evento-002",
        bboxes=[
            box(
                x=100,
                y=185,
                largura=115,
                altura=250,
            ),
        ],
        agora=103.0,
    )

    assert (
        segunda[0].pessoa_id
        == pessoa_id
    )

    assert (
        segunda[0].status
        == "suppressed"
    )

    assert (
        tracker.quantidade_ativas
        == 1
    )

    terceira = await tracker.registrar_lote(
        camera="Camera 01",
        evento_id="evento-003",
        bboxes=[
            box(
                x=135,
                y=190,
                largura=118,
                altura=255,
            ),
        ],
        agora=106.0,
    )

    assert (
        terceira[0].pessoa_id
        == pessoa_id
    )

    assert (
        terceira[0].status
        == "updated"
    )

    assert (
        terceira[0].quantidade_deteccoes
        == 3
    )

    tracker_lote = PersonTracker()

    duas_pessoas = (
        await tracker_lote.registrar_lote(
            camera="Camera 02",
            evento_id="evento-lote",
            bboxes=[
                box(100, 100),
                box(700, 100),
            ],
            agora=200.0,
        )
    )

    assert len(duas_pessoas) == 2

    assert (
        duas_pessoas[0].pessoa_id
        != duas_pessoas[1].pessoa_id
    )

    print(
        "===== CONTINUIDADE DO ID ====="
    )

    print(
        "ID inicial:",
        pessoa_id,
    )

    print(
        "ID após movimento longo:",
        segunda[0].pessoa_id,
    )

    print(
        "ID após terceira detecção:",
        terceira[0].pessoa_id,
    )

    print(
        "Quantidade de detecções:",
        terceira[0].quantidade_deteccoes,
    )

    print(
        "Pessoas ativas:",
        tracker.quantidade_ativas,
    )

    print(
        "IDs diferentes no lote:",
        (
            duas_pessoas[0].pessoa_id
            != duas_pessoas[1].pessoa_id
        ),
    )

    print(
        "\nTeste concluído com sucesso."
    )


if __name__ == "__main__":
    asyncio.run(
        executar_teste()
    )