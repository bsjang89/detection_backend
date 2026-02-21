from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from config.database import Base


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    training_session_id = Column(Integer, ForeignKey("training_sessions.id", ondelete="CASCADE"), nullable=False)

    name = Column(String(255), nullable=False)
    weights_path = Column(String(500), nullable=False)
    is_best = Column(Boolean, default=False)  # best.pt vs last.pt

    # Metrics at checkpoint
    map50 = Column(Float)
    map50_95 = Column(Float)
    precision = Column(Float)
    recall = Column(Float)

    # Model metadata
    model_type = Column(String(50))  # yolov8n, yolov8s, etc.
    task_type = Column(String(20))  # bbox, obb
    img_size = Column(Integer)
    num_classes = Column(Integer)

    # Additional info
    model_metadata = Column("metadata", JSON)  # Store class names, etc.

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    training_session = relationship("TrainingSession", back_populates="models")
    inference_results = relationship("InferenceResult", back_populates="model", cascade="all, delete-orphan")
