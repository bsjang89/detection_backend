from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Enum, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from config.database import Base


class SplitType(str, enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    UNLABELED = "unlabeled"


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)

    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)  # Full path to image
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)

    split_type = Column(Enum(SplitType), default=SplitType.UNLABELED)
    annotation_version = Column(Integer, nullable=False, default=0, server_default="0")

    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    project = relationship("Project", back_populates="images")
    annotations = relationship("Annotation", back_populates="image", cascade="all, delete-orphan")
    inference_results = relationship("InferenceResult", back_populates="image", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("project_id", "filename", name="uq_images_project_filename"),
    )
