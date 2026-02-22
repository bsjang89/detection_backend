from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
from pathlib import Path
import os

from config.database import get_db
from config.settings import settings
from models.database import Image, Annotation
from schemas import ImageResponse, ImageWithAnnotations
from utils.file_handler import create_thumbnail, create_viewer_image, get_thumbnail_path, get_viewer_path

router = APIRouter()


def resolve_thumbnail_path_fast(image: Image) -> str:
    """Fast path resolver for list APIs without expensive image processing."""
    thumbnail_path = get_thumbnail_path(image.project_id, image.filename)
    if thumbnail_path.exists():
        return str(thumbnail_path)
    return image.file_path


def resolve_thumbnail_path(image: Image) -> str:
    """Return thumbnail path, creating it lazily for older uploads if missing."""
    thumbnail_path = get_thumbnail_path(image.project_id, image.filename)
    if thumbnail_path.exists():
        return str(thumbnail_path)

    source_path = Path(image.file_path)
    if not source_path.exists():
        return image.file_path

    try:
        create_thumbnail(source_path, thumbnail_path, settings.THUMBNAIL_SIZE)
        return str(thumbnail_path)
    except Exception:
        return image.file_path


def resolve_viewer_path_fast(image: Image) -> str:
    """Fast path resolver for list APIs without expensive image processing."""
    viewer_path = get_viewer_path(image.project_id, image.filename)
    if viewer_path.exists():
        return str(viewer_path)
    return image.file_path


def resolve_viewer_path(image: Image) -> str:
    """Return viewer-optimized path, creating it lazily for older uploads if missing."""
    viewer_path = get_viewer_path(image.project_id, image.filename)
    if viewer_path.exists():
        return str(viewer_path)

    source_path = Path(image.file_path)
    if not source_path.exists():
        return image.file_path

    try:
        create_viewer_image(source_path, viewer_path)
        return str(viewer_path)
    except Exception:
        return image.file_path


@router.get("/", response_model=List[ImageWithAnnotations])
def list_images(
    project_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List images for a project"""
    images = db.query(Image).filter(
        Image.project_id == project_id
    ).offset(skip).limit(limit).all()

    image_ids = [img.id for img in images]
    annotation_counts = {}
    if image_ids:
        rows = (
            db.query(Annotation.image_id, func.count(Annotation.id))
            .filter(Annotation.image_id.in_(image_ids))
            .group_by(Annotation.image_id)
            .all()
        )
        annotation_counts = {image_id: count for image_id, count in rows}

    results = []
    for img in images:
        annotation_count = annotation_counts.get(img.id, 0)
        img_dict = ImageResponse.model_validate(img).model_dump()
        img_dict["thumbnail_path"] = resolve_thumbnail_path_fast(img)
        img_dict["viewer_path"] = resolve_viewer_path_fast(img)
        img_dict["annotation_count"] = annotation_count
        results.append(ImageWithAnnotations(**img_dict))

    return results


@router.get("/{image_id}", response_model=ImageWithAnnotations)
def get_image(image_id: int, db: Session = Depends(get_db)):
    """Get image details"""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {image_id} not found"
        )

    annotation_count = db.query(Annotation).filter(Annotation.image_id == image_id).count()
    img_dict = ImageResponse.model_validate(image).model_dump()
    img_dict["thumbnail_path"] = resolve_thumbnail_path(image)
    img_dict["viewer_path"] = resolve_viewer_path(image)
    img_dict["annotation_count"] = annotation_count

    return ImageWithAnnotations(**img_dict)


@router.delete("/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_image(image_id: int, db: Session = Depends(get_db)):
    """Delete an image and its annotations"""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {image_id} not found"
        )

    # Delete files
    if os.path.exists(image.file_path):
        os.remove(image.file_path)

    thumbnail_path = get_thumbnail_path(image.project_id, image.filename)
    if thumbnail_path.exists():
        thumbnail_path.unlink()
    viewer_path = get_viewer_path(image.project_id, image.filename)
    if viewer_path.exists():
        viewer_path.unlink()

    db.delete(image)
    db.commit()

    return None
