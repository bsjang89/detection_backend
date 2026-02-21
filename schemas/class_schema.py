from pydantic import BaseModel, Field
from typing import Optional


class ClassBase(BaseModel):
    class_id: int = Field(..., ge=0)
    class_name: str = Field(..., min_length=1, max_length=100)
    color: Optional[str] = Field(None, pattern=r'^#[0-9A-Fa-f]{6}$')


class ClassCreate(ClassBase):
    project_id: int


class ClassUpdate(BaseModel):
    class_name: Optional[str] = Field(None, min_length=1, max_length=100)
    color: Optional[str] = Field(None, pattern=r'^#[0-9A-Fa-f]{6}$')


class ClassResponse(ClassBase):
    id: int
    project_id: int

    class Config:
        from_attributes = True
