from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from config.database import Base


class InferenceResult(Base):
    __tablename__ = "inference_results"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id", ondelete="CASCADE"), nullable=False)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=False)

    # Detection results (JSON array)
    # Each detection: {class_id, confidence, bbox: [x1,y1,x2,y2] or obb: [x1,y1,x2,y2,x3,y3,x4,y4]}
    detections = Column(JSON, nullable=False)

    # Result image path (with drawn boxes)
    result_image_path = Column(String(500))

    # Inference settings
    conf_threshold = Column(Float)
    iou_threshold = Column(Float)

    # Performance
    inference_time_ms = Column(Float)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    model = relationship("Model", back_populates="inference_results")
    image = relationship("Image", back_populates="inference_results")
