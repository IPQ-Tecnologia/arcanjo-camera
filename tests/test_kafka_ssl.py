import pytest

import app.messaging.kafka_producer as kafka_module
from app.core.config import settings
from app.messaging.kafka_producer import KafkaPublisher


class FakeSSLContext:
    def __init__(self) -> None:
        self.cert_chain = None

    def load_cert_chain(
        self,
        certfile,
        keyfile,
        password=None,
    ) -> None:
        self.cert_chain = {
            "certfile": certfile,
            "keyfile": keyfile,
            "password": password,
        }


def test_plaintext_does_not_create_ssl_context(monkeypatch):
    monkeypatch.setattr(
        settings,
        "kafka_security_protocol",
        "PLAINTEXT",
    )

    publisher = KafkaPublisher()
    config = publisher._build_config()

    assert config["security_protocol"] == "PLAINTEXT"
    assert "ssl_context" not in config


def test_ssl_uses_ca_file(monkeypatch):
    fake_context = FakeSSLContext()
    received = {}

    def fake_create_ssl_context(cafile=None):
        received["cafile"] = cafile
        return fake_context

    monkeypatch.setattr(
        kafka_module,
        "create_ssl_context",
        fake_create_ssl_context,
    )

    monkeypatch.setattr(
        settings,
        "kafka_ssl_cafile",
        "/tmp/ca.crt",
    )
    monkeypatch.setattr(
        settings,
        "kafka_ssl_certfile",
        "",
    )
    monkeypatch.setattr(
        settings,
        "kafka_ssl_keyfile",
        "",
    )
    monkeypatch.setattr(
        settings,
        "kafka_ssl_key_password",
        "",
    )

    context = KafkaPublisher._create_ssl_context()

    assert context is fake_context
    assert received["cafile"] == "/tmp/ca.crt"
    assert fake_context.cert_chain is None


def test_ssl_loads_client_certificate_and_key(monkeypatch):
    fake_context = FakeSSLContext()

    monkeypatch.setattr(
        kafka_module,
        "create_ssl_context",
        lambda cafile=None: fake_context,
    )

    monkeypatch.setattr(
        settings,
        "kafka_ssl_cafile",
        "/tmp/ca.crt",
    )
    monkeypatch.setattr(
        settings,
        "kafka_ssl_certfile",
        "/tmp/client.crt",
    )
    monkeypatch.setattr(
        settings,
        "kafka_ssl_keyfile",
        "/tmp/client.key",
    )
    monkeypatch.setattr(
        settings,
        "kafka_ssl_key_password",
        "secret",
    )

    context = KafkaPublisher._create_ssl_context()

    assert context is fake_context
    assert fake_context.cert_chain == {
        "certfile": "/tmp/client.crt",
        "keyfile": "/tmp/client.key",
        "password": "secret",
    }


@pytest.mark.parametrize(
    ("certfile", "keyfile"),
    [
        ("/tmp/client.crt", ""),
        ("", "/tmp/client.key"),
    ],
)
def test_ssl_requires_certificate_and_key_together(
    monkeypatch,
    certfile,
    keyfile,
):
    monkeypatch.setattr(
        kafka_module,
        "create_ssl_context",
        lambda cafile=None: FakeSSLContext(),
    )

    monkeypatch.setattr(
        settings,
        "kafka_ssl_cafile",
        "",
    )
    monkeypatch.setattr(
        settings,
        "kafka_ssl_certfile",
        certfile,
    )
    monkeypatch.setattr(
        settings,
        "kafka_ssl_keyfile",
        keyfile,
    )
    monkeypatch.setattr(
        settings,
        "kafka_ssl_key_password",
        "",
    )

    with pytest.raises(
        ValueError,
        match="KAFKA_SSL_CERTFILE.*KAFKA_SSL_KEYFILE",
    ):
        KafkaPublisher._create_ssl_context()
