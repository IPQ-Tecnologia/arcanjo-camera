import asyncio

from app.services.person_tracker import DetectionBox, PersonTracker


def box(x: int, y: int, width: int = 100, height: int = 220) -> DetectionBox:
    return DetectionBox(x=x, y=y, width=width, height=height)


async def run_test() -> None:
    tracker = PersonTracker(
        reprocessing_interval=5.0,
        exit_timeout=15.0,
        iou_threshold=0.25,
        distance_threshold=0.75,
    )

    # First frame: two new people.
    first_frame = await tracker.register_batch(
        camera="Camera 01",
        event_id="quadro-001",
        bboxes=[box(100, 120), box(600, 120)],
        now=100.0,
    )

    assert len(first_frame) == 2
    assert all(decision.status == "entered" for decision in first_frame)

    left_person_id = first_frame[0].person_id
    right_person_id = first_frame[1].person_id

    assert left_person_id != right_person_id

    # Second frame: reversed order.
    #
    # The person on the right shows up first in the list, but should
    # keep their correct ID.
    second_frame = await tracker.register_batch(
        camera="Camera 01",
        event_id="quadro-002",
        bboxes=[box(610, 125), box(110, 125)],
        now=101.0,
    )

    assert len(second_frame) == 2
    assert second_frame[0].person_id == right_person_id
    assert second_frame[1].person_id == left_person_id
    assert all(decision.status == "suppressed" for decision in second_frame)
    assert second_frame[0].person_id != second_frame[1].person_id

    # Third frame: more than five seconds have passed since the last
    # processing.
    third_frame = await tracker.register_batch(
        camera="Camera 01",
        event_id="quadro-003",
        bboxes=[box(120, 130), box(620, 130)],
        now=106.5,
    )

    assert all(decision.status == "updated" for decision in third_frame)
    assert third_frame[0].person_id == left_person_id
    assert third_frame[1].person_id == right_person_id
    assert all(decision.detection_count == 3 for decision in third_frame)

    # After more than 15 seconds, both people should exit.
    exits = await tracker.collect_exits(now=122.0)

    assert len(exits) == 2
    assert all(decision.status == "exited" for decision in exits)

    exit_ids = {decision.person_id for decision in exits}

    assert exit_ids == {left_person_id, right_person_id}
    assert tracker.active_count == 0

    print("===== BATCH TRACKING =====")
    print("Left person:", left_person_id)
    print("Right person:", right_person_id)
    print("First frame:", [decision.status for decision in first_frame])
    print("Second frame:", [decision.status for decision in second_frame])
    print("Third frame:", [decision.status for decision in third_frame])
    print("Exits:", len(exits))
    print("\nTest completed successfully.")


if __name__ == "__main__":
    asyncio.run(run_test())
