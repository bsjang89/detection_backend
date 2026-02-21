from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from config.database import Base


class Class(Base):
    __tablename__ = "classes"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    class_id = Column(Integer, nullable=False)  # 0, 1, 2, ... for YOLO
    class_name = Column(String(100), nullable=False)
    color = Column(String(7))  # Hex color like #FF0000

    project = relationship("Project", back_populates="classes")
    annotations = relationship("Annotation", back_populates="class_obj", cascade="all, delete-orphan")

    __table_args__ = (
        # Unique constraint: same class_id within a project
        # Index for performance
    )
