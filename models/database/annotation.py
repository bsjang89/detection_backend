from sqlalchemy import Column, Integer, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from config.database import Base


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=False)
    class_id = Column(Integer, ForeignKey("classes.id", ondelete="CASCADE"), nullable=False)

    # Normalized coordinates (0-1) for YOLO
    cx = Column(Float, nullable=False)
    cy = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    rotation = Column(Float, default=0.0)  # For OBB, in radians

    # Pixel coordinates (absolute)
    px_cx = Column(Float, nullable=True)
    px_cy = Column(Float, nullable=True)
    px_width = Column(Float, nullable=True)
    px_height = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    image = relationship("Image", back_populates="annotations")
    class_obj = relationship("Class", back_populates="annotations")
