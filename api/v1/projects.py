from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
import shutil
from pathlib import Path
from PIL import Image as PILImage

from config.database import get_db
from config.settings import settings
from models.database import Project, Image, Annotation, Class, SplitType
from utils.file_handler import create_thumbnail, get_project_images_dir, get_thumbnail_path, safe_filename
from schemas import (
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
    ProjectWithStats,
    ClassCreate,
    ClassResponse,
)

router = APIRouter()


@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    """Create a new project"""
    # Check if name already exists
    existing = db.query(Project).filter(Project.name == project.name).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Project with name '{project.name}' already exists"
        )

    db_project = Project(**project.model_dump())
    db.add(db_project)
    db.commit()
    db.refresh(db_project)

    # Create upload directory for this project
    project_dir = Path(settings.UPLOAD_DIR) / str(db_project.id)
    project_dir.mkdir(parents=True, exist_ok=True)

    return db_project


@router.get("/", response_model=List[ProjectWithStats])
def list_projects(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all projects with statistics"""
    projects = db.query(Project).offset(skip).limit(limit).all()

    results = []
    for project in projects:
        # Get statistics
        total_images = db.query(func.count(Image.id)).filter(Image.project_id == project.id).scalar() or 0

        labeled_images = db.query(func.count(func.distinct(Annotation.image_id))).join(
            Image
        ).filter(Image.project_id == project.id).scalar() or 0

        train_images = db.query(func.count(Image.id)).filter(
            Image.project_id == project.id,
            Image.split_type == SplitType.TRAIN
        ).scalar() or 0

        val_images = db.query(func.count(Image.id)).filter(
            Image.project_id == project.id,
            Image.split_type == SplitType.VAL
        ).scalar() or 0

        test_images = db.query(func.count(Image.id)).filter(
            Image.project_id == project.id,
            Image.split_type == SplitType.TEST
        ).scalar() or 0

        total_classes = db.query(func.count(Class.id)).filter(Class.project_id == project.id).scalar() or 0

        total_annotations = db.query(func.count(Annotation.id)).join(
            Image
        ).filter(Image.project_id == project.id).scalar() or 0

        project_dict = ProjectResponse.model_validate(project).model_dump()
        project_dict.update({
            "total_images": total_images,
            "labeled_images": labeled_images,
            "train_images": train_images,
            "val_images": val_images,
            "test_images": test_images,
            "total_classes": total_classes,
            "total_annotations": total_annotations,
        })
        results.append(ProjectWithStats(**project_dict))

    return results


@router.get("/{project_id}", response_model=ProjectWithStats)
def get_project(project_id: int, db: Session = Depends(get_db)):
    """Get project details with statistics"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found"
        )

    # Get statistics (same as list_projects)
    total_images = db.query(func.count(Image.id)).filter(Image.project_id == project.id).scalar() or 0
    labeled_images = db.query(func.count(func.distinct(Annotation.image_id))).join(
        Image
    ).filter(Image.project_id == project.id).scalar() or 0
    train_images = db.query(func.count(Image.id)).filter(
        Image.project_id == project.id, Image.split_type == SplitType.TRAIN
    ).scalar() or 0
    val_images = db.query(func.count(Image.id)).filter(
        Image.project_id == project.id, Image.split_type == SplitType.VAL
    ).scalar() or 0
    test_images = db.query(func.count(Image.id)).filter(
        Image.project_id == project.id, Image.split_type == SplitType.TEST
    ).scalar() or 0
    total_classes = db.query(func.count(Class.id)).filter(Class.project_id == project.id).scalar() or 0
    total_annotations = db.query(func.count(Annotation.id)).join(
        Image
    ).filter(Image.project_id == project.id).scalar() or 0

    project_dict = ProjectResponse.model_validate(project).model_dump()
    project_dict.update({
        "total_images": total_images,
        "labeled_images": labeled_images,
        "train_images": train_images,
        "val_images": val_images,
        "test_images": test_images,
        "total_classes": total_classes,
        "total_annotations": total_annotations,
    })

    return ProjectWithStats(**project_dict)


@router.put("/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: int,
    project_update: ProjectUpdate,
    db: Session = Depends(get_db)
):
    """Update project details"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found"
        )

    update_data = project_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(project, field, value)

    db.commit()
    db.refresh(project)
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: int, db: Session = Depends(get_db)):
    """Delete a project and all associated data"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found"
        )

    # Delete project directory
    project_dir = Path(settings.UPLOAD_DIR) / str(project_id)
    if project_dir.exists():
        shutil.rmtree(project_dir)

    db.delete(project)
    db.commit()

    return None


@router.post("/{project_id}/images/upload", response_model=List[dict])
async def upload_images(
    project_id: int,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Upload multiple images to a project"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found"
        )

    project_images_dir = get_project_images_dir(project_id)
    project_images_dir.mkdir(parents=True, exist_ok=True)

    uploaded_images = []

    for file in files:
        normalized_name = safe_filename(file.filename or "")
        if not normalized_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image filename"
            )

        file_path = project_images_dir / normalized_name
        thumbnail_path = get_thumbnail_path(project_id, normalized_name)

        # Save original file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Get image dimensions
        try:
            with PILImage.open(file_path) as img:
                width, height = img.size
            create_thumbnail(file_path, thumbnail_path, settings.THUMBNAIL_SIZE)
        except Exception as e:
            if file_path.exists():
                file_path.unlink()
            if thumbnail_path.exists():
                thumbnail_path.unlink()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image file: {normalized_name}"
            )

        # Create database entry
        db_image = Image(
            project_id=project_id,
            filename=normalized_name,
            file_path=str(file_path),
            width=width,
            height=height,
            split_type=SplitType.UNLABELED
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        uploaded_images.append({
            "id": db_image.id,
            "filename": normalized_name,
            "file_path": str(file_path),
            "thumbnail_path": str(thumbnail_path),
            "width": width,
            "height": height
        })

    return uploaded_images


@router.post("/{project_id}/classes", response_model=ClassResponse, status_code=status.HTTP_201_CREATED)
def create_class(
    project_id: int,
    class_data: ClassCreate,
    db: Session = Depends(get_db)
):
    """Create a new class for the project"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found"
        )

    # Check if class_id already exists in this project
    existing = db.query(Class).filter(
        Class.project_id == project_id,
        Class.class_id == class_data.class_id
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Class ID {class_data.class_id} already exists in this project"
        )

    db_class = Class(**class_data.model_dump())
    db.add(db_class)
    db.commit()
    db.refresh(db_class)

    return db_class


@router.get("/{project_id}/classes", response_model=List[ClassResponse])
def list_classes(project_id: int, db: Session = Depends(get_db)):
    """List all classes for a project"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found"
        )

    classes = db.query(Class).filter(Class.project_id == project_id).order_by(Class.class_id).all()
    return classes
