from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from config.database import get_db
from models.database import Annotation, Image, Class
from schemas import (
    AnnotationCreate,
    AnnotationUpdate,
    AnnotationResponse,
    AnnotationBatchCreate,
    AnnotationBatchResponse,
)

router = APIRouter()


@router.post("/", response_model=AnnotationResponse, status_code=status.HTTP_201_CREATED)
def create_annotation(
    annotation: AnnotationCreate,
    db: Session = Depends(get_db)
):
    """Create a new annotation"""
    # Verify image exists
    image = db.query(Image).filter(Image.id == annotation.image_id).with_for_update().first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {annotation.image_id} not found"
        )

    # Verify class exists in the same project
    class_obj = db.query(Class).filter(Class.id == annotation.class_id).first()
    if not class_obj or class_obj.project_id != image.project_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid class_id {annotation.class_id} for this project"
        )

    db_annotation = Annotation(**annotation.model_dump())
    db.add(db_annotation)
    image.annotation_version = (image.annotation_version or 0) + 1
    db.commit()
    db.refresh(db_annotation)

    return db_annotation


@router.post("/batch", response_model=AnnotationBatchResponse)
def create_annotations_batch(
    batch: AnnotationBatchCreate,
    db: Session = Depends(get_db)
):
    """Create multiple annotations for an image (replaces existing)"""
    # Verify image exists
    image = db.query(Image).filter(Image.id == batch.image_id).with_for_update().first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {batch.image_id} not found"
        )
    current_version = image.annotation_version or 0
    if batch.base_annotation_version is not None and batch.base_annotation_version != current_version:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Annotation version conflict",
                "current_annotation_version": current_version,
            },
        )

    # Delete existing annotations for this image
    db.query(Annotation).filter(Annotation.image_id == batch.image_id).delete()

    # Create new annotations
    created_annotations = []
    for ann_data in batch.annotations:
        # Verify class exists
        class_obj = db.query(Class).filter(Class.id == ann_data.class_id).first()
        if not class_obj or class_obj.project_id != image.project_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid class_id {ann_data.class_id} for this project"
            )

        db_annotation = Annotation(
            image_id=batch.image_id,
            **ann_data.model_dump()
        )
        db.add(db_annotation)
        created_annotations.append(db_annotation)

    image.annotation_version = current_version + 1
    db.commit()

    # Refresh all
    for ann in created_annotations:
        db.refresh(ann)
    db.refresh(image)

    return AnnotationBatchResponse(
        annotations=created_annotations,
        annotation_version=image.annotation_version,
    )


@router.get("/image/{image_id}", response_model=List[AnnotationResponse])
def list_annotations_for_image(
    image_id: int,
    db: Session = Depends(get_db)
):
    """Get all annotations for an image"""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {image_id} not found"
        )

    annotations = db.query(Annotation).filter(Annotation.image_id == image_id).all()
    return annotations


@router.put("/{annotation_id}", response_model=AnnotationResponse)
def update_annotation(
    annotation_id: int,
    annotation_update: AnnotationUpdate,
    db: Session = Depends(get_db)
):
    """Update an annotation"""
    annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if not annotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Annotation {annotation_id} not found"
        )

    update_data = annotation_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(annotation, field, value)
    if annotation.image:
        annotation.image.annotation_version = (annotation.image.annotation_version or 0) + 1

    db.commit()
    db.refresh(annotation)

    return annotation


@router.delete("/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_annotation(annotation_id: int, db: Session = Depends(get_db)):
    """Delete an annotation"""
    annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if not annotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Annotation {annotation_id} not found"
        )

    if annotation.image:
        annotation.image.annotation_version = (annotation.image.annotation_version or 0) + 1
    db.delete(annotation)
    db.commit()

    return None
