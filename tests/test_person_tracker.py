import asyncio

from app.services.person_tracker import DetectionBox, PersonTracker


async def main() -> None:
    tracker = PersonTracker(reprocessing_interval=5.0, exit_timeout=8.0)

    first_box = DetectionBox(x=600, y=250, width=100, height=190)
    similar_box = DetectionBox(x=608, y=254, width=102, height=188)

    first = await tracker.register(
        camera="Camera 01",
        event_id="evento-001",
        bbox=first_box,
        now=0.0,
    )

    repeated = await tracker.register(
        camera="Camera 01",
        event_id="evento-002",
        bbox=similar_box,
        now=1.0,
    )

    updated = await tracker.register(
        camera="Camera 01",
        event_id="evento-003",
        bbox=similar_box,
        now=6.0,
    )

    exits = await tracker.collect_exits(now=15.0)

    print("1:", first.status, first.should_process, first.person_id)
    print("2:", repeated.status, repeated.should_process, repeated.person_id)
    print("3:", updated.status, updated.should_process, updated.person_id)
    print("4:", exits[0].status, exits[0].person_id)

    print(
        "Same person:",
        first.person_id == repeated.person_id == updated.person_id == exits[0].person_id,
    )


if __name__ == "__main__":
    asyncio.run(main())
