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
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def iou(self, other: "DetectionBox") -> float:
        """
        Computes the overlap between two boxes.

        0.0 means no overlap.
        1.0 means identical boxes.
        """
        intersection_x1 = max(self.x, other.x)
        intersection_y1 = max(self.y, other.y)
        intersection_x2 = min(self.x2, other.x2)
        intersection_y2 = min(self.y2, other.y2)

        intersection_width = max(0, intersection_x2 - intersection_x1)
        intersection_height = max(0, intersection_y2 - intersection_y1)
        intersection_area = intersection_width * intersection_height
        union_area = self.area + other.area - intersection_area

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    def center_distance(self, other: "DetectionBox") -> float:
        diff_x = self.center_x - other.center_x
        diff_y = self.center_y - other.center_y

        return math.hypot(diff_x, diff_y)


@dataclass
class TrackedPerson:
    person_id: str
    camera: str

    first_detected_at: float
    last_detected_at: float
    last_processed_at: float

    bbox: DetectionBox
    detection_count: int
    last_event_id: str


@dataclass(frozen=True)
class TrackingDecision:
    person_id: str
    camera: str
    event_id: str

    status: TrackingStatus
    should_process: bool

    detection_count: int
    bbox: DetectionBox


class PersonTracker:
    def __init__(
        self,
        reprocessing_interval: float = 5.0,
        exit_timeout: float = 15.0,
        iou_threshold: float = 0.25,
        distance_threshold: float = 0.75,
        flexible_continuity_window: float = 4.0,
        flexible_distance_threshold: float = 2.5,
        flexible_area_ratio_threshold: float = 3.0,
        normalized_speed_threshold: float = 1.25,
    ) -> None:
        if reprocessing_interval <= 0:
            raise ValueError("reprocessing_interval must be positive")

        if exit_timeout <= 0:
            raise ValueError("exit_timeout must be positive")

        if iou_threshold < 0:
            raise ValueError("iou_threshold cannot be negative")

        if distance_threshold < 0:
            raise ValueError("distance_threshold cannot be negative")

        if flexible_continuity_window <= 0:
            raise ValueError("flexible_continuity_window must be positive")

        if flexible_distance_threshold < distance_threshold:
            raise ValueError(
                "flexible_distance_threshold must be greater than or equal to "
                "distance_threshold"
            )

        if flexible_area_ratio_threshold < 1:
            raise ValueError("flexible_area_ratio_threshold must be greater than or equal to 1")

        if normalized_speed_threshold <= 0:
            raise ValueError("normalized_speed_threshold must be positive")

        self.reprocessing_interval = reprocessing_interval
        self.exit_timeout = exit_timeout
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.flexible_continuity_window = flexible_continuity_window
        self.flexible_distance_threshold = flexible_distance_threshold
        self.flexible_area_ratio_threshold = flexible_area_ratio_threshold
        self.normalized_speed_threshold = normalized_speed_threshold

        self._people: dict[str, TrackedPerson] = {}
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        return len(self._people)

    async def register(
        self,
        camera: str,
        event_id: str,
        bbox: DetectionBox,
        now: float | None = None,
    ) -> TrackingDecision:
        """Kept for compatibility with calls that send a single bounding box."""
        decisions = await self.register_batch(
            camera=camera,
            event_id=event_id,
            bboxes=[bbox],
            now=now,
        )

        return decisions[0]

    async def register_batch(
        self,
        camera: str,
        event_id: str,
        bboxes: list[DetectionBox],
        now: float | None = None,
    ) -> list[TrackingDecision]:
        """
        Registers every person found in the same frame.

        Each active person can be matched to at most one bounding box
        of the frame.
        """
        if not bboxes:
            return []

        for bbox in bboxes:
            if bbox.width <= 0 or bbox.height <= 0:
                raise ValueError("All bounding boxes must have positive width and height")

        moment = time.monotonic() if now is None else now

        async with self._lock:
            matches = self._match_detections_batch(camera=camera, bboxes=bboxes)
            self._apply_flexible_continuity(
                camera=camera,
                bboxes=bboxes,
                matches=matches,
                now=moment,
            )

            total_detections = len(bboxes)
            decisions: list[TrackingDecision] = []

            for index, bbox in enumerate(bboxes):
                individual_event_id = self._create_individual_event_id(
                    event_id=event_id,
                    index=index,
                    total=total_detections,
                )

                person = matches.get(index)

                if person is None:
                    person = self._create_person(
                        camera=camera,
                        event_id=individual_event_id,
                        bbox=bbox,
                        now=moment,
                    )
                    decisions.append(
                        TrackingDecision(
                            person_id=person.person_id,
                            camera=camera,
                            event_id=individual_event_id,
                            status="entered",
                            should_process=True,
                            detection_count=1,
                            bbox=bbox,
                        )
                    )
                    continue

                decision = self._update_person(
                    person=person,
                    event_id=individual_event_id,
                    bbox=bbox,
                    now=moment,
                )
                decisions.append(decision)

            return decisions

    async def collect_exits(self, now: float | None = None) -> list[TrackingDecision]:
        moment = time.monotonic() if now is None else now

        async with self._lock:
            ended_people: list[TrackingDecision] = []
            ids_to_remove: list[str] = []

            for person_id, person in self._people.items():
                time_since_detection = moment - person.last_detected_at
                if time_since_detection < self.exit_timeout:
                    continue

                ended_people.append(
                    TrackingDecision(
                        person_id=person.person_id,
                        camera=person.camera,
                        event_id=person.last_event_id,
                        status="exited",
                        should_process=False,
                        detection_count=person.detection_count,
                        bbox=person.bbox,
                    )
                )
                ids_to_remove.append(person_id)

            for person_id in ids_to_remove:
                self._people.pop(person_id, None)

            return ended_people

    async def clear(self) -> None:
        async with self._lock:
            self._people.clear()

    def _match_detections_batch(
        self,
        camera: str,
        bboxes: list[DetectionBox],
    ) -> dict[int, TrackedPerson]:
        """
        Finds the best matches between the bounding boxes and the
        active people.

        A person cannot be matched twice within the same frame.
        """
        candidates: list[tuple[float, int, str]] = []

        for index, bbox in enumerate(bboxes):
            for person in self._people.values():
                if person.camera != camera:
                    continue

                score = self._calculate_score(bbox=bbox, person=person)
                if score is None:
                    continue

                candidates.append((score, index, person.person_id))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

        used_detections: set[int] = set()
        used_people: set[str] = set()
        matches: dict[int, TrackedPerson] = {}

        for _, index, person_id in candidates:
            if index in used_detections:
                continue

            if person_id in used_people:
                continue

            person = self._people.get(person_id)
            if person is None:
                continue

            matches[index] = person
            used_detections.add(index)
            used_people.add(person_id)

        return matches

    def _apply_flexible_continuity(
        self,
        camera: str,
        bboxes: list[DetectionBox],
        matches: dict[int, TrackedPerson],
        now: float,
    ) -> None:
        """
        Fixes the case where a person crosses the frame quickly and
        the new box ends up far from the previous box.

        This rule only applies when:

        - there is a single bounding box;
        - there is a single active person on the camera;
        - the normal match failed;
        - the interval between frames is short;
        - the box size is still plausible.

        This way, the rule doesn't interfere with batch tracking of
        several people.
        """
        if len(bboxes) != 1:
            return

        if matches:
            return

        camera_people = [person for person in self._people.values() if person.camera == camera]
        if len(camera_people) != 1:
            return

        person = camera_people[0]
        bbox = bboxes[0]

        continuity_valid = self._can_use_flexible_continuity(
            person=person,
            bbox=bbox,
            now=now,
        )
        if continuity_valid:
            matches[0] = person

    def _can_use_flexible_continuity(
        self,
        person: TrackedPerson,
        bbox: DetectionBox,
        now: float,
    ) -> bool:
        time_since_detection = max(0.0, now - person.last_detected_at)
        if time_since_detection > self.flexible_continuity_window:
            return False

        smaller_area = max(1, min(bbox.area, person.bbox.area))
        larger_area = max(bbox.area, person.bbox.area)
        area_ratio = larger_area / smaller_area
        if area_ratio > self.flexible_area_ratio_threshold:
            return False

        distance = bbox.center_distance(person.bbox)
        largest_dimension = max(bbox.width, bbox.height, person.bbox.width, person.bbox.height, 1)
        normalized_distance = distance / largest_dimension

        dynamic_threshold = min(
            self.flexible_distance_threshold,
            max(self.distance_threshold, 1.25 + time_since_detection * 0.45),
        )
        if normalized_distance > dynamic_threshold:
            return False

        normalized_speed = normalized_distance / max(time_since_detection, 0.25)
        if normalized_speed > self.normalized_speed_threshold:
            return False

        return True

    def _calculate_score(
        self,
        bbox: DetectionBox,
        person: TrackedPerson,
    ) -> float | None:
        overlap = bbox.iou(person.bbox)
        distance = bbox.center_distance(person.bbox)
        largest_dimension = max(bbox.width, bbox.height, person.bbox.width, person.bbox.height, 1)
        normalized_distance = distance / largest_dimension

        boxes_close = normalized_distance <= self.distance_threshold
        boxes_overlapping = overlap >= self.iou_threshold
        if not (boxes_close or boxes_overlapping):
            return None

        distance_score = max(0.0, 1.0 - normalized_distance)

        return overlap + distance_score

    def _update_person(
        self,
        person: TrackedPerson,
        event_id: str,
        bbox: DetectionBox,
        now: float,
    ) -> TrackingDecision:
        person.last_detected_at = now
        person.bbox = bbox
        person.last_event_id = event_id
        person.detection_count += 1

        time_since_processed = now - person.last_processed_at

        if time_since_processed >= self.reprocessing_interval:
            person.last_processed_at = now

            return TrackingDecision(
                person_id=person.person_id,
                camera=person.camera,
                event_id=event_id,
                status="updated",
                should_process=True,
                detection_count=person.detection_count,
                bbox=bbox,
            )

        return TrackingDecision(
            person_id=person.person_id,
            camera=person.camera,
            event_id=event_id,
            status="suppressed",
            should_process=False,
            detection_count=person.detection_count,
            bbox=bbox,
        )

    def _find_person(
        self,
        camera: str,
        bbox: DetectionBox,
    ) -> TrackedPerson | None:
        """Kept for compatibility with individual tests."""
        best_person: TrackedPerson | None = None
        best_score = -1.0

        for person in self._people.values():
            if person.camera != camera:
                continue

            score = self._calculate_score(bbox=bbox, person=person)
            if score is None:
                continue

            if score > best_score:
                best_score = score
                best_person = person

        return best_person

    def _create_person(
        self,
        camera: str,
        event_id: str,
        bbox: DetectionBox,
        now: float,
    ) -> TrackedPerson:
        normalized_camera = "".join(
            character.lower() if character.isalnum() else "-" for character in camera
        ).strip("-")

        if not normalized_camera:
            normalized_camera = "camera"

        person_id = f"{normalized_camera}-{uuid.uuid4().hex[:8]}"

        person = TrackedPerson(
            person_id=person_id,
            camera=camera,
            first_detected_at=now,
            last_detected_at=now,
            last_processed_at=now,
            bbox=bbox,
            detection_count=1,
            last_event_id=event_id,
        )

        self._people[person_id] = person

        return person

    def _create_individual_event_id(self, event_id: str, index: int, total: int) -> str:
        """
        A frame with a single person keeps the original ID.

        Frames with several people receive IDs like:

        event123-01
        event123-02
        """
        if total <= 1:
            return event_id

        return f"{event_id}-{index + 1:02d}"


person_tracker = PersonTracker(
    reprocessing_interval=5.0,
    exit_timeout=15.0,
    iou_threshold=0.25,
    distance_threshold=0.75,
    flexible_continuity_window=12.0,
    flexible_distance_threshold=2.5,
    flexible_area_ratio_threshold=12.0,
    normalized_speed_threshold=1.25,
)
