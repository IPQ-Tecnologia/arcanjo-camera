from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    kafka_enabled: bool = False

    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_normalized: str = "camera.events.normalized"
    kafka_topic_errors: str = "camera.events.errors"
    kafka_client_id: str = "camera-ingestion-api"

    kafka_security_protocol: str = "PLAINTEXT"
    kafka_sasl_mechanism: str = ""
    kafka_sasl_username: str = ""
    kafka_sasl_password: str = ""
    kafka_ssl_cafile: str = ""
    kafka_request_timeout_ms: int = 10000

    ingestion_queue_size: int = 1000
    spool_directory: str = "spool"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
