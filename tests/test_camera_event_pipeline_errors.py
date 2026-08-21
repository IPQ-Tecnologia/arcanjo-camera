import asyncio
from datetime import datetime, timezone

import pytest

import app.services.camera_event_pipeline as pipeline_module
from app.core.config import settings
from app.domain.models.camera_event import RawCameraPackage
from app.services.camera_event_pipeline import CameraEventPipeline


class FakePublisher:
    started = True


class FailingAdapter:
    def normalize(self, package, body):
        raise RuntimeError("falha simulada no processamento")


def build_package() -> RawCameraPackage:
    return RawCameraPackage(
        event_id="package-error-1",
        received_at=datetime.now(timezone.utc),
        content_type="application/octet-stream",
        camera_ip="192.168.1.100",
        package_path="/tmp/package-error-1",
    )


async def stop_worker(task: asyncio.Task) -> None:
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_worker_publishes_processing_error(monkeypatch):
    publisher = FakePublisher()
    published = {}

    monkeypatch.setattr(
        settings,
        "kafka_enabled",
        True,
    )

    monkeypatch.setattr(
        pipeline_module.camera_adapter_factory,
        "find_adapter",
        lambda **kwargs: FailingAdapter(),
    )

    async def fake_publish_processing_error(
        publisher,
        event_id,
        data,
    ):
        published["publisher"] = publisher
        published["event_id"] = event_id
        published["data"] = data

        return {
            "topic": settings.kafka_topic_errors,
            "partition": 0,
            "offset": 1,
            "timestamp": None,
        }

    monkeypatch.setattr(
        pipeline_module,
        "publish_processing_error",
        fake_publish_processing_error,
    )

    pipeline = CameraEventPipeline(
        publisher=publisher,
        maxsize=10,
        worker_count=1,
    )

    worker = asyncio.create_task(pipeline._worker(1))

    package = build_package()

    await pipeline.queue.put(
        (
            package,
            b"payload-invalido",
        )
    )

    await asyncio.wait_for(
        pipeline.queue.join(),
        timeout=2,
    )

    assert published["publisher"] is publisher
    assert published["event_id"] == "package-error-1"

    payload = published["data"]

    assert payload["event_id"] == "package-error-1"
    assert payload["type"] == "camera_processing_error"

    assert payload["error"]["type"] == "RuntimeError"
    assert payload["error"]["message"] == (
        "falha simulada no processamento"
    )

    assert payload["source"]["camera_ip"] == "192.168.1.100"
    assert payload["source"]["content_type"] == (
        "application/octet-stream"
    )
    assert payload["source"]["package_path"] == (
        "/tmp/package-error-1"
    )

    assert payload["worker"] == 1
    assert "occurred_at" in payload

    await stop_worker(worker)


@pytest.mark.asyncio
async def test_worker_survives_error_publication_failure(
    monkeypatch,
):
    publisher = FakePublisher()

    monkeypatch.setattr(
        settings,
        "kafka_enabled",
        True,
    )

    monkeypatch.setattr(
        pipeline_module.camera_adapter_factory,
        "find_adapter",
        lambda **kwargs: FailingAdapter(),
    )

    async def failing_publish_processing_error(
        publisher,
        event_id,
        data,
    ):
        raise RuntimeError("kafka indisponivel")

    monkeypatch.setattr(
        pipeline_module,
        "publish_processing_error",
        failing_publish_processing_error,
    )

    pipeline = CameraEventPipeline(
        publisher=publisher,
        maxsize=10,
        worker_count=1,
    )

    worker = asyncio.create_task(pipeline._worker(1))

    await pipeline.queue.put(
        (
            build_package(),
            b"payload-invalido",
        )
    )

    await asyncio.wait_for(
        pipeline.queue.join(),
        timeout=2,
    )

    # O erro ao publicar no Kafka não deve derrubar o worker.
    assert not worker.done()

    await stop_worker(worker)
