from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from config.database import get_db
from models.database import TrainingSession, Project
from schemas import (
    TrainingSessionCreate,
    TrainingSessionResponse,
    TrainingMetricsLogResponse,
)
from models.database import TrainingMetricsLog
from services.training_service import TrainingService
from services.dataset_service import DatasetService
from api.v1.websocket import (
    broadcast_training_update_sync,
    broadcast_status_update_sync,
)

router = APIRouter()


@router.post("/sessions", response_model=TrainingSessionResponse, status_code=status.HTTP_201_CREATED)
def create_training_session(
    session_data: TrainingSessionCreate,
    db: Session = Depends(get_db)
):
    """Create a new training session"""
    # Verify project exists
    project = db.query(Project).filter(Project.id == session_data.project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {session_data.project_id} not found"
        )

    # Create session
    db_session = TrainingSession(
        project_id=session_data.project_id,
        name=session_data.name,
        model_type=session_data.model_type,
        config=session_data.config.model_dump()
    )

    db.add(db_session)
    db.commit()
    db.refresh(db_session)

    return db_session


@router.post("/sessions/{session_id}/start")
def start_training(
    session_id: int,
    db: Session = Depends(get_db)
):
    """Start training for a session"""
    try:
        def on_epoch_callback(training_session_id: int, epoch: int, metrics: dict):
            broadcast_training_update_sync(
                training_session_id=training_session_id,
                epoch=epoch,
                metrics=metrics,
            )

        result = TrainingService.start_training(
            db=db,
            training_session_id=session_id,
            on_epoch_callback=on_epoch_callback,
        )
        broadcast_status_update_sync(
            training_session_id=session_id,
            status="running",
            message="Training started",
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start training: {str(e)}"
        )


@router.post("/sessions/{session_id}/stop")
def stop_training(
    session_id: int,
    db: Session = Depends(get_db)
):
    """Stop a running training session"""
    try:
        result = TrainingService.stop_training(
            db=db,
            training_session_id=session_id
        )
        broadcast_status_update_sync(
            training_session_id=session_id,
            status="stopped",
            message="Training stopped",
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.delete("/sessions/{session_id}")
def delete_training_session(
    session_id: int,
    db: Session = Depends(get_db)
):
    """Delete a training session"""
    try:
        return TrainingService.delete_training_session(
            db=db,
            training_session_id=session_id
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete training session: {str(e)}"
        )


@router.get("/sessions/{session_id}", response_model=TrainingSessionResponse)
def get_training_session(
    session_id: int,
    db: Session = Depends(get_db)
):
    """Get training session details"""
    session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training session {session_id} not found"
        )

    return session


@router.get("/sessions/{session_id}/status")
def get_training_status(
    session_id: int,
    db: Session = Depends(get_db)
):
    """Get training status"""
    try:
        status_info = TrainingService.get_training_status(
            db=db,
            training_session_id=session_id
        )
        return status_info
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.get("/sessions/{session_id}/metrics", response_model=List[TrainingMetricsLogResponse])
def get_training_metrics(
    session_id: int,
    db: Session = Depends(get_db)
):
    """Get training metrics logs"""
    session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training session {session_id} not found"
        )

    metrics = db.query(TrainingMetricsLog).filter(
        TrainingMetricsLog.training_session_id == session_id
    ).order_by(TrainingMetricsLog.epoch).all()

    return metrics


@router.get("/projects/{project_id}/sessions", response_model=List[TrainingSessionResponse])
def list_project_training_sessions(
    project_id: int,
    db: Session = Depends(get_db)
):
    """List all training sessions for a project"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found"
        )

    sessions = db.query(TrainingSession).filter(
        TrainingSession.project_id == project_id
    ).order_by(TrainingSession.created_at.desc()).all()

    return sessions


@router.post("/projects/{project_id}/dataset/split")
def split_dataset(
    project_id: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    db: Session = Depends(get_db)
):
    """Split project dataset into train/val/test"""
    try:
        result = DatasetService.split_dataset(
            db=db,
            project_id=project_id,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/projects/{project_id}/dataset/stats")
def get_dataset_stats(
    project_id: int,
    db: Session = Depends(get_db)
):
    """Get dataset statistics"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found"
        )

    stats = DatasetService.get_dataset_stats(db=db, project_id=project_id)
    return stats
