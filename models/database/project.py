from sqlalchemy import Column, Integer, String, DateTime, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from config.database import Base


class TaskType(str, enum.Enum):
    BBOX = "bbox"
    OBB = "obb"


class ProjectStatus(str, enum.Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(String(1000))
    task_type = Column(Enum(TaskType), nullable=False, default=TaskType.BBOX)
    status = Column(Enum(ProjectStatus), nullable=False, default=ProjectStatus.ACTIVE)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    images = relationship("Image", back_populates="project", cascade="all, delete-orphan")
    classes = relationship("Class", back_populates="project", cascade="all, delete-orphan")
    training_sessions = relationship("TrainingSession", back_populates="project", cascade="all, delete-orphan")
