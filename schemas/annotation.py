from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional


class AnnotationBase(BaseModel):
    class_id: int = Field(..., ge=0)
    # Normalized (0-1)
    cx: float = Field(..., ge=0, le=1)
    cy: float = Field(..., ge=0, le=1)
    width: float = Field(..., ge=0, le=1)
    height: float = Field(..., ge=0, le=1)
    rotation: float = Field(default=0.0)
    # Pixel (absolute)
    px_cx: Optional[float] = None
    px_cy: Optional[float] = None
    px_width: Optional[float] = None
    px_height: Optional[float] = None

    @validator('rotation')
    def validate_rotation(cls, v):
        import math
        while v > math.pi:
            v -= 2 * math.pi
        while v < -math.pi:
            v += 2 * math.pi
        return v


class AnnotationCreate(AnnotationBase):
    image_id: int


class AnnotationUpdate(BaseModel):
    class_id: Optional[int] = Field(None, ge=0)
    cx: Optional[float] = Field(None, ge=0, le=1)
    cy: Optional[float] = Field(None, ge=0, le=1)
    width: Optional[float] = Field(None, ge=0, le=1)
    height: Optional[float] = Field(None, ge=0, le=1)
    rotation: Optional[float] = None
    px_cx: Optional[float] = None
    px_cy: Optional[float] = None
    px_width: Optional[float] = None
    px_height: Optional[float] = None


class AnnotationResponse(AnnotationBase):
    id: int
    image_id: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


# Batch operations
class AnnotationBatchCreate(BaseModel):
    image_id: int
    annotations: list[AnnotationBase]
