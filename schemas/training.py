from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional, Dict, Any
from models.database.training_session import TrainingStatus


class TrainingConfigSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    epochs: int = Field(default=50, ge=1, le=1000)
    batch: int = Field(default=16, ge=1, le=128)
    imgsz: int = Field(default=640, ge=320, le=1280)
    lr0: float = Field(default=0.01, gt=0)
    patience: int = Field(default=50, ge=0)
    device: Optional[int] = None  # Will be set by TrainingService
    cache: bool = Field(default=True)
    workers: int = Field(default=8, ge=0)
    optimizer: str = Field(default="auto")
    pretrained: bool = Field(default=True)
    verbose: bool = Field(default=True)


class TrainingSessionCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    project_id: int
    name: Optional[str] = None
    model_type: str = Field(..., pattern=r'^yolov[0-9]+[nsmxl]?$')  # yolov8n, yolov8s, etc.
    config: TrainingConfigSchema


class TrainingSessionUpdate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Optional[TrainingStatus] = None


class TrainingSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    project_id: int
    name: Optional[str]
    model_type: str
    config: Dict[str, Any]
    status: TrainingStatus
    progress: float
    best_weights_path: Optional[str]
    last_weights_path: Optional[str]
    final_map50: Optional[float]
    final_map50_95: Optional[float]
    final_precision: Optional[float]
    final_recall: Optional[float]
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime

class TrainingMetricsLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    training_session_id: int
    epoch: int
    train_box_loss: Optional[float]
    train_cls_loss: Optional[float]
    train_dfl_loss: Optional[float]
    train_total_loss: Optional[float]
    val_box_loss: Optional[float]
    val_cls_loss: Optional[float]
    val_dfl_loss: Optional[float]
    val_total_loss: Optional[float]
    map50: Optional[float]
    map50_95: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    learning_rate: Optional[float]
    logged_at: datetime

