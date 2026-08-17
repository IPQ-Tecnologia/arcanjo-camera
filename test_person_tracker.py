import asyncio

from app.services.person_tracker import (
    DetectionBox,
    PersonTracker,
)


async def main() -> None:
    tracker = PersonTracker(
        intervalo_reprocessamento=5.0,
        tempo_para_saida=8.0,
    )

    primeira_caixa = DetectionBox(
        x=600,
        y=250,
        largura=100,
        altura=190,
    )

    caixa_parecida = DetectionBox(
        x=608,
        y=254,
        largura=102,
        altura=188,
    )

    primeira = await tracker.registrar(
        camera="Camera 01",
        evento_id="evento-001",
        bbox=primeira_caixa,
        agora=0.0,
    )

    repetida = await tracker.registrar(
        camera="Camera 01",
        evento_id="evento-002",
        bbox=caixa_parecida,
        agora=1.0,
    )

    atualizada = await tracker.registrar(
        camera="Camera 01",
        evento_id="evento-003",
        bbox=caixa_parecida,
        agora=6.0,
    )

    saidas = await tracker.coletar_saidas(
        agora=15.0,
    )

    print(
        "1:",
        primeira.status,
        primeira.deve_processar,
        primeira.pessoa_id,
    )

    print(
        "2:",
        repetida.status,
        repetida.deve_processar,
        repetida.pessoa_id,
    )

    print(
        "3:",
        atualizada.status,
        atualizada.deve_processar,
        atualizada.pessoa_id,
    )

    print(
        "4:",
        saidas[0].status,
        saidas[0].pessoa_id,
    )

    print(
        "Mesma pessoa:",
        primeira.pessoa_id
        == repetida.pessoa_id
        == atualizada.pessoa_id
        == saidas[0].pessoa_id,
    )


if __name__ == "__main__":
    asyncio.run(main())