import asyncio

from app.services.person_tracker import DetectionBox, PersonTracker


def box(x: int, y: int, width: int = 120, height: int = 260) -> DetectionBox:
    return DetectionBox(x=x, y=y, width=width, height=height)


async def run_test() -> None:
    tracker = PersonTracker(
        reprocessing_interval=5.0,
        exit_timeout=15.0,
        iou_threshold=0.25,
        distance_threshold=0.75,
        flexible_continuity_window=4.0,
        flexible_distance_threshold=2.5,
        flexible_area_ratio_threshold=3.0,
        normalized_speed_threshold=1.25,
    )

    first = await tracker.register_batch(
        camera="Camera 01",
        event_id="evento-001",
        bboxes=[box(x=520, y=180)],
        now=100.0,
    )

    person_id = first[0].person_id

    assert first[0].status == "entered"

    second = await tracker.register_batch(
        camera="Camera 01",
        event_id="evento-002",
        bboxes=[box(x=100, y=185, width=115, height=250)],
        now=103.0,
    )

    assert second[0].person_id == person_id
    assert second[0].status == "suppressed"
    assert tracker.active_count == 1

    third = await tracker.register_batch(
        camera="Camera 01",
        event_id="evento-003",
        bboxes=[box(x=135, y=190, width=118, height=255)],
        now=106.0,
    )

    assert third[0].person_id == person_id
    assert third[0].status == "updated"
    assert third[0].detection_count == 3

    batch_tracker = PersonTracker()

    two_people = await batch_tracker.register_batch(
        camera="Camera 02",
        event_id="evento-lote",
        bboxes=[box(100, 100), box(700, 100)],
        now=200.0,
    )

    assert len(two_people) == 2
    assert two_people[0].person_id != two_people[1].person_id

    print("===== ID CONTINUITY =====")
    print("Initial ID:", person_id)
    print("ID after long movement:", second[0].person_id)
    print("ID after third detection:", third[0].person_id)
    print("Detection count:", third[0].detection_count)
    print("Active people:", tracker.active_count)
    print("Different IDs in batch:", two_people[0].person_id != two_people[1].person_id)
    print("\nTest completed successfully.")


if __name__ == "__main__":
    asyncio.run(run_test())
