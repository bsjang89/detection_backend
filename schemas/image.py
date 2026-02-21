from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from models.database.image import SplitType


class ImageBase(BaseModel):
    filename: str
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)


class ImageCreate(ImageBase):
    project_id: int
    file_path: str


class ImageUpdate(BaseModel):
    split_type: Optional[SplitType] = None


class ImageResponse(ImageBase):
    id: int
    project_id: int
    file_path: str
    thumbnail_path: Optional[str] = None
    split_type: SplitType
    uploaded_at: datetime

    class Config:
        from_attributes = True


class ImageWithAnnotations(ImageResponse):
    annotation_count: int = 0
