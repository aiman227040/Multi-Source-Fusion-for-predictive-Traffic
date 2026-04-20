from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    google_maps_api_key: str = ""
    yolo_weights: str = "yolov8n.pt"
    artifact_dir: Path = Path(__file__).resolve().parent.parent / "artifacts"
    lstm_checkpoint: str = "lstm_forecaster.pt"
    default_seed: int = 42


settings = Settings()
