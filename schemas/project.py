from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
from models.database.project import TaskType, ProjectStatus


# Base schemas
class ProjectBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    task_type: TaskType = TaskType.BBOX


class ProjectCreate(ProjectBase):
    pass


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[ProjectStatus] = None


class ProjectResponse(ProjectBase):
    id: int
    status: ProjectStatus
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


# With statistics
class ProjectWithStats(ProjectResponse):
    total_images: int = 0
    labeled_images: int = 0
    train_images: int = 0
    val_images: int = 0
    test_images: int = 0
    total_classes: int = 0
    total_annotations: int = 0
