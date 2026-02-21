from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Polygon
import time

from models.database import Model, Image, InferenceResult
from models.ml import ModelRegistry
from utils.gpu_manager import GPUManager
from config.settings import settings


class InferenceService:
    """
    Service for running inference on images.
    Uses GPU 1 exclusively for inference.
    Implements caching for loaded models.
    """

    # Model cache: model_id -> (model_instance, last_used_time)
    _model_cache: Dict[int, Tuple[Any, float]] = {}
    _cache_max_size = 3

    @classmethod
    def _load_model(cls, model_id: int, db: Session):
        """
        Load model from cache or disk.

        Args:
            model_id: Model ID
            db: Database session

        Returns:
            Model instance
        """
        # Check cache
        if model_id in cls._model_cache:
            model_instance, _ = cls._model_cache[model_id]
            cls._model_cache[model_id] = (model_instance, time.time())
            return model_instance

        # Load from database
        model_record = db.query(Model).filter(Model.id == model_id).first()
        if not model_record:
            raise ValueError(f"Model {model_id} not found")

        # Create model instance
        model_instance = ModelRegistry.create_model(
            task_type=model_record.task_type,
            model_type=model_record.model_type,
            device=GPUManager.get_inference_gpu()
        )

        # Load weights
        model_instance.load_model(model_record.weights_path)

        # Add to cache
        cls._model_cache[model_id] = (model_instance, time.time())

        # Clean cache if too large
        if len(cls._model_cache) > cls._cache_max_size:
            # Remove oldest
            oldest_id = min(cls._model_cache.keys(), key=lambda k: cls._model_cache[k][1])
            del cls._model_cache[oldest_id]

        return model_instance

    @classmethod
    def predict_single(
        cls,
        db: Session,
        model_id: int,
        image_id: int,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        save_result_image: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference on a single image.

        Args:
            db: Database session
            model_id: Model ID
            image_id: Image ID
            conf_threshold: Confidence threshold
            iou_threshold: IOU threshold for NMS
            save_result_image: Whether to save result image with drawn detections

        Returns:
            Inference results dictionary
        """
        # Validate GPU
        GPUManager.validate_inference_gpu()

        # Get image
        image_record = db.query(Image).filter(Image.id == image_id).first()
        if not image_record:
            raise ValueError(f"Image {image_id} not found")

        # Load model
        model = cls._load_model(model_id, db)

        # Run inference
        start_time = time.time()
        detections = model.predict(
            image_path=image_record.file_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        inference_time_ms = (time.time() - start_time) * 1000

        # Save result image if requested
        result_image_path = None
        if save_result_image:
            model_record = db.query(Model).filter(Model.id == model_id).first()

            if model_record.task_type == "obb":
                result_image_path = cls._draw_obb_detections(
                    image_path=image_record.file_path,
                    detections=detections,
                    output_dir=Path(settings.RESULTS_DIR) / "inference" / str(model_id),
                    filename=f"{image_record.id}_{Path(image_record.filename).stem}.jpg",
                    conf_threshold=conf_threshold,
                    overlap_iou=0.5
                )
            else:
                result_image_path = cls._draw_bbox_detections(
                    image_path=image_record.file_path,
                    detections=detections,
                    output_dir=Path(settings.RESULTS_DIR) / "inference" / str(model_id),
                    filename=f"{image_record.id}_{Path(image_record.filename).stem}.jpg"
                )

        # Save inference result to database
        inference_result = InferenceResult(
            model_id=model_id,
            image_id=image_id,
            detections=detections,
            result_image_path=result_image_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            inference_time_ms=inference_time_ms
        )

        db.add(inference_result)
        db.commit()
        db.refresh(inference_result)

        return {
            "inference_result_id": inference_result.id,
            "detections": detections,
            "result_image_path": result_image_path,
            "inference_time_ms": inference_time_ms,
            "detection_count": len(detections)
        }

    @classmethod
    def compare_models(
        cls,
        db: Session,
        model_id_1: int,
        model_id_2: int,
        image_id: int,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Compare two models on the same image.

        Args:
            db: Database session
            model_id_1: First model ID
            model_id_2: Second model ID
            image_id: Image ID
            conf_threshold: Confidence threshold
            iou_threshold: IOU threshold

        Returns:
            Comparison results
        """
        result1 = cls.predict_single(
            db=db,
            model_id=model_id_1,
            image_id=image_id,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            save_result_image=True
        )

        result2 = cls.predict_single(
            db=db,
            model_id=model_id_2,
            image_id=image_id,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            save_result_image=True
        )

        return {
            "model_1": {
                "model_id": model_id_1,
                "detections": result1["detections"],
                "result_image_path": result1["result_image_path"],
                "inference_time_ms": result1["inference_time_ms"],
                "detection_count": result1["detection_count"]
            },
            "model_2": {
                "model_id": model_id_2,
                "detections": result2["detections"],
                "result_image_path": result2["result_image_path"],
                "inference_time_ms": result2["inference_time_ms"],
                "detection_count": result2["detection_count"]
            }
        }

    @classmethod
    def _draw_bbox_detections(
        cls,
        image_path: str,
        detections: List[Dict],
        output_dir: Path,
        filename: str
    ) -> str:
        """
        Draw bounding box detections on image.

        Args:
            image_path: Path to input image
            detections: List of detection dictionaries
            output_dir: Output directory
            filename: Output filename

        Returns:
            Path to result image
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        # Read image
        img = cv2.imread(image_path)

        # Draw each detection
        for det in detections:
            bbox = det["bbox"]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, bbox)
            conf = det["confidence"]
            class_id = det["class_id"]

            # Draw rectangle
            color = cls._get_class_color(class_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"Class {class_id}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save
        cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        return str(output_path)

    @classmethod
    def _draw_obb_detections(
        cls,
        image_path: str,
        detections: List[Dict],
        output_dir: Path,
        filename: str,
        conf_threshold: float = 0.25,
        overlap_iou: float = 0.5
    ) -> str:
        """
        Draw OBB detections on image with custom NMS.
        Integrates logic from Test_Otoki.py

        Args:
            image_path: Path to input image
            detections: List of OBB detection dictionaries
            output_dir: Output directory
            filename: Output filename
            conf_threshold: Confidence threshold for drawing
            overlap_iou: IOU threshold for custom NMS

        Returns:
            Path to result image
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        # Read image
        img = cv2.imread(image_path)

        if not detections:
            cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return str(output_path)

        # Convert to arrays for NMS
        polys = []
        confs = []
        class_ids = []

        for det in detections:
            obb = det["obb"]  # [x1,y1,x2,y2,x3,y3,x4,y4]
            poly = np.array(obb).reshape(4, 2)
            polys.append(poly)
            confs.append(det["confidence"])
            class_ids.append(det["class_id"])

        polys = np.array(polys)
        confs = np.array(confs)
        class_ids = np.array(class_ids)

        # Filter by confidence
        valid = confs >= conf_threshold
        polys = polys[valid]
        confs = confs[valid]
        class_ids = class_ids[valid]

        if len(polys) == 0:
            cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return str(output_path)

        # Custom NMS for OBB
        keep_idxs = cls._suppress_obb(polys, confs, iou_th=overlap_iou)

        # Draw
        for i in keep_idxs:
            pts = polys[i].astype(int)
            cls_id = int(class_ids[i])
            conf = float(confs[i])

            color = cls._get_class_color(cls_id)
            label = f"Class {cls_id}: {conf:.2f}"

            # Draw polygon
            cv2.polylines(img, [pts], True, color, 4)

            # Draw label at center
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            cv2.putText(
                img, label, (cx - 40, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA
            )

        cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return str(output_path)

    @staticmethod
    def _polygon_iou(p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Calculate IoU between two polygons.
        From Test_Otoki.py

        Args:
            p1: (4,2) numpy array
            p2: (4,2) numpy array

        Returns:
            IoU value
        """
        poly1 = Polygon(p1)
        poly2 = Polygon(p2)

        if not poly1.is_valid or not poly2.is_valid:
            return 0.0

        inter = poly1.intersection(poly2).area
        union = poly1.union(poly2).area

        return float(inter / union) if union > 0 else 0.0

    @classmethod
    def _suppress_obb(cls, polys: np.ndarray, confs: np.ndarray, iou_th: float = 0.5) -> List[int]:
        """
        Custom NMS for OBB using polygon IoU.
        From Test_Otoki.py

        Args:
            polys: (N, 4, 2) polygon coordinates
            confs: (N,) confidence scores
            iou_th: IoU threshold

        Returns:
            List of indices to keep
        """
        if len(polys) == 0:
            return []

        idxs = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)
        keep = []

        while idxs:
            i = idxs.pop(0)
            keep.append(i)

            rest = []
            for j in idxs:
                if cls._polygon_iou(polys[i], polys[j]) < iou_th:
                    rest.append(j)
            idxs = rest

        return keep

    @staticmethod
    def _get_class_color(class_id: int) -> Tuple[int, int, int]:
        """Get color for a class"""
        colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
        ]
        return colors[class_id % len(colors)]
