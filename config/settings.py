from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Object Detection Training System"

    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/detection_db"

    # GPU Management
    TRAINING_GPU_ID: int = 0
    INFERENCE_GPU_ID: int = 0

    # File Storage
    UPLOAD_DIR: str = "uploads"
    THUMBNAIL_SIZE: int = 256
    MODELS_DIR: str = "trained_models"
    NETWORK_DIR: str = "network"
    RESULTS_DIR: str = "results"
    REPORTS_DIR: str = "reports"

    # Training Defaults
    DEFAULT_EPOCHS: int = 50
    DEFAULT_BATCH_SIZE: int = 16
    DEFAULT_IMG_SIZE: int = 640

    # Inference Defaults
    DEFAULT_CONF_THRESHOLD: float = 0.25
    DEFAULT_IOU_THRESHOLD: float = 0.7

    # WebSocket
    WS_MESSAGE_QUEUE_SIZE: int = 100

    # CORS
    BACKEND_CORS_ORIGINS: list = ["http://localhost:5173", "http://localhost:3000"]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
