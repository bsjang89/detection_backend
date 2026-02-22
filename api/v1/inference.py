from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field, ConfigDict
from typing import List

from config.database import get_db
from models.database import Model, Image
from services.inference_service import InferenceService
from services.report_service import ReportService

router = APIRouter()


class InferenceRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: int
    image_id: int
    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    save_result_image: bool = True
    save_to_db: bool = True


class CompareRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id_1: int
    model_id_2: int
    image_id: int
    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class InferenceBatchRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: int
    image_ids: List[int] = Field(min_length=1)
    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    batch_size: int = Field(default=10, ge=1, le=64)
    save_result_image: bool = False
    save_to_db: bool = False


class CompareBatchRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id_1: int
    model_id_2: int
    image_ids: List[int] = Field(min_length=1)
    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    batch_size: int = Field(default=10, ge=1, le=64)
    save_result_image: bool = False
    save_to_db: bool = False
    include_detections: bool = False


@router.post("/predict")
def predict(
    request: InferenceRequest,
    db: Session = Depends(get_db)
):
    """
    Run inference on a single image.

    Returns:
        - detections: List of detected objects
        - result_image_path: Path to result image (if saved)
        - inference_time_ms: Inference time in milliseconds
        - detection_count: Number of detections
    """
    # Verify model exists
    model = db.query(Model).filter(Model.id == request.model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {request.model_id} not found"
        )

    # Verify image exists
    image = db.query(Image).filter(Image.id == request.image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {request.image_id} not found"
        )

    try:
        result = InferenceService.predict_single(
            db=db,
            model_id=request.model_id,
            image_id=request.image_id,
            conf_threshold=request.conf_threshold,
            iou_threshold=request.iou_threshold,
            save_result_image=request.save_result_image,
            save_to_db=request.save_to_db,
        )
        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


@router.post("/predict/batch")
def predict_batch(
    request: InferenceBatchRequest,
    db: Session = Depends(get_db)
):
    """Run batched inference on multiple images."""
    model = db.query(Model).filter(Model.id == request.model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {request.model_id} not found"
        )

    try:
        results = InferenceService.predict_batch(
            db=db,
            model_id=request.model_id,
            image_ids=request.image_ids,
            conf_threshold=request.conf_threshold,
            iou_threshold=request.iou_threshold,
            batch_size=request.batch_size,
            save_result_image=request.save_result_image,
            save_to_db=request.save_to_db,
        )
        return {"results": results}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}"
        )


@router.post("/compare")
def compare_models(
    request: CompareRequest,
    db: Session = Depends(get_db)
):
    """
    Compare two models on the same image.

    Returns:
        - model_1: Results from first model
        - model_2: Results from second model
    """
    # Verify models exist
    model1 = db.query(Model).filter(Model.id == request.model_id_1).first()
    if not model1:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {request.model_id_1} not found"
        )

    model2 = db.query(Model).filter(Model.id == request.model_id_2).first()
    if not model2:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {request.model_id_2} not found"
        )

    # Verify image exists
    image = db.query(Image).filter(Image.id == request.image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {request.image_id} not found"
        )

    try:
        result = InferenceService.compare_models(
            db=db,
            model_id_1=request.model_id_1,
            model_id_2=request.model_id_2,
            image_id=request.image_id,
            conf_threshold=request.conf_threshold,
            iou_threshold=request.iou_threshold
        )
        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model comparison failed: {str(e)}"
        )


@router.post("/compare/batch")
def compare_models_batch(
    request: CompareBatchRequest,
    db: Session = Depends(get_db)
):
    """Compare two models over multiple images in batch."""
    model1 = db.query(Model).filter(Model.id == request.model_id_1).first()
    if not model1:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {request.model_id_1} not found"
        )

    model2 = db.query(Model).filter(Model.id == request.model_id_2).first()
    if not model2:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {request.model_id_2} not found"
        )

    try:
        result = InferenceService.compare_models_batch(
            db=db,
            model_id_1=request.model_id_1,
            model_id_2=request.model_id_2,
            image_ids=request.image_ids,
            conf_threshold=request.conf_threshold,
            iou_threshold=request.iou_threshold,
            batch_size=request.batch_size,
            save_result_image=request.save_result_image,
            save_to_db=request.save_to_db,
            include_detections=request.include_detections,
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
            detail=f"Batch model comparison failed: {str(e)}"
        )


@router.get("/models")
def list_models(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all available models"""
    models = db.query(Model).offset(skip).limit(limit).all()
    return models


@router.get("/models/{model_id}")
def get_model(
    model_id: int,
    db: Session = Depends(get_db)
):
    """Get model details"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    return model


@router.get("/models/{model_id}/report", response_class=HTMLResponse)
def generate_report(
    model_id: int,
    db: Session = Depends(get_db)
):
    """Generate HTML report for a model"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

    try:
        html = ReportService.generate_report(db=db, model_id=model_id)
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}"
        )
