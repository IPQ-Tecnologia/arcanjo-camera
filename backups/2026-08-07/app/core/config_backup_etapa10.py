from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    kafka_enabled: bool = False

    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_normalized: str = "camera.events.normalized"
    kafka_topic_errors: str = "camera.events.errors"
    kafka_client_id: str = "camera-ingestion-api"

    ingestion_queue_size: int = 1000
    spool_directory: str = "spool"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
