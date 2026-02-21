from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from config.database import get_db
from models.database import Image, Annotation, Project
from schemas import ImageResponse, ImageWithAnnotations

router = APIRouter()


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

    results = []
    for img in images:
        annotation_count = db.query(Annotation).filter(Annotation.image_id == img.id).count()
        img_dict = ImageResponse.model_validate(img).model_dump()
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

    # Delete file
    import os
    if os.path.exists(image.file_path):
        os.remove(image.file_path)

    db.delete(image)
    db.commit()

    return None
