from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Enum, JSON, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from config.database import Base


class TrainingStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TrainingSession(Base):
    __tablename__ = "training_sessions"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)

    name = Column(String(255))
    model_type = Column(String(50), nullable=False)  # yolov8n, yolov8s, yolov8m, etc.

    # Training Configuration (stored as JSON)
    config = Column(JSON, nullable=False)  # epochs, batch, imgsz, etc.

    # Status
    status = Column(Enum(TrainingStatus), default=TrainingStatus.PENDING)
    progress = Column(Float, default=0.0)  # 0-100%

    # Results
    best_weights_path = Column(String(500))
    last_weights_path = Column(String(500))

    # Final Metrics
    final_map50 = Column(Float)
    final_map50_95 = Column(Float)
    final_precision = Column(Float)
    final_recall = Column(Float)

    # Error handling
    error_message = Column(Text)

    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    project = relationship("Project", back_populates="training_sessions")
    metrics_logs = relationship("TrainingMetricsLog", back_populates="training_session", cascade="all, delete-orphan")
    models = relationship("Model", back_populates="training_session", cascade="all, delete-orphan")
