from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
from uuid import uuid4
import cv2
import numpy as np
from shapely.geometry import Polygon
import time
import torch

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
    _cache_lock = threading.RLock()
    _cache_max_size = 3
    _result_box_thickness = 5
    _result_text_scale_bbox = 0.92
    _result_text_scale_obb = 0.98
    _result_text_thickness = 2
    _result_jpeg_quality = 90
    _result_review_size = 1024

    @classmethod
    def _build_result_filename(cls, image_record: Image, model_id: int) -> str:
        stem = Path(image_record.filename).stem
        return f"{image_record.id}_{model_id}_{stem}_{uuid4().hex[:8]}.jpg"

    @classmethod
    def _load_model(
        cls,
        model_id: int,
        db: Session,
        model_record: Optional[Model] = None
    ):
        """
        Load model from cache or disk.

        Args:
            model_id: Model ID
            db: Database session

        Returns:
            Model instance
        """
        # Check cache first
        with cls._cache_lock:
            if model_id in cls._model_cache:
                model_instance, _ = cls._model_cache[model_id]
                cls._model_cache[model_id] = (model_instance, time.time())
                return model_instance

        # Load from database
        if model_record is None:
            model_record = db.query(Model).filter(Model.id == model_id).first()
        if not model_record:
            raise ValueError(f"Model {model_id} not found")

        # Create model instance
        model_instance = ModelRegistry.create_model(
            task_type=model_record.task_type,
            model_type=model_record.model_type,
            device=GPUManager.get_inference_device()
        )

        # Load weights
        model_instance.load_model(model_record.weights_path)

        # Add to cache
        with cls._cache_lock:
            cls._model_cache[model_id] = (model_instance, time.time())

            # Clean cache if too large
            while len(cls._model_cache) > cls._cache_max_size:
                oldest_id = min(cls._model_cache.keys(), key=lambda k: cls._model_cache[k][1])
                del cls._model_cache[oldest_id]

        return model_instance

    @classmethod
    def _preload_models_parallel(cls, db: Session, model_ids: List[int]) -> None:
        """
        Preload models concurrently into cache to reduce compare startup latency.
        Inference execution remains sequential on the inference device.
        """
        unique_ids = list(dict.fromkeys(model_ids))
        if not unique_ids:
            return

        records = db.query(Model).filter(Model.id.in_(unique_ids)).all()
        record_map = {m.id: m for m in records}
        missing = [mid for mid in unique_ids if mid not in record_map]
        if missing:
            raise ValueError(f"Model(s) not found: {missing}")

        with cls._cache_lock:
            to_load = [mid for mid in unique_ids if mid not in cls._model_cache]
            for mid in unique_ids:
                if mid in cls._model_cache:
                    model_instance, _ = cls._model_cache[mid]
                    cls._model_cache[mid] = (model_instance, time.time())

        if not to_load:
            return

        workers = min(2, len(to_load))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(cls._load_model, mid, db, record_map[mid])
                for mid in to_load
            ]
            for f in futures:
                f.result()

    @classmethod
    def predict_single(
        cls,
        db: Session,
        model_id: int,
        image_id: int,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        save_result_image: bool = True,
        save_to_db: bool = True
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
            save_to_db: Whether to persist inference result row

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
                    filename=cls._build_result_filename(image_record, model_id),
                    conf_threshold=conf_threshold,
                    overlap_iou=0.5
                )
            else:
                result_image_path = cls._draw_bbox_detections(
                    image_path=image_record.file_path,
                    detections=detections,
                    output_dir=Path(settings.RESULTS_DIR) / "inference" / str(model_id),
                    filename=cls._build_result_filename(image_record, model_id)
                )

        inference_result_id = None
        if save_to_db:
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
            inference_result_id = inference_result.id

        return {
            "inference_result_id": inference_result_id,
            "detections": detections,
            "result_image_path": result_image_path,
            "inference_time_ms": inference_time_ms,
            "detection_count": len(detections)
        }

    @staticmethod
    def _extract_detections_from_result(task_type: str, result: Any) -> List[Dict[str, Any]]:
        """Extract normalized detection payload from a single Ultralytics result."""
        detections: List[Dict[str, Any]] = []

        if task_type == "obb":
            if hasattr(result, "obb") and result.obb is not None:
                obbs = result.obb
                for i in range(len(obbs)):
                    detections.append({
                        "class_id": int(obbs.cls[i]),
                        "confidence": float(obbs.conf[i]),
                        "obb": obbs.xyxyxyxy[i].cpu().numpy().flatten().tolist(),
                    })
            return detections

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        for i in range(len(boxes)):
            detections.append({
                "class_id": int(boxes.cls[i]),
                "confidence": float(boxes.conf[i]),
                "bbox": boxes.xyxy[i].cpu().numpy().tolist(),
            })
        return detections

    @classmethod
    def predict_batch(
        cls,
        db: Session,
        model_id: int,
        image_ids: List[int],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        batch_size: int = 10,
        save_result_image: bool = False,
        save_to_db: bool = False,
        ordered_images: Optional[List[Image]] = None,
        source_inputs: Optional[List[Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run inference on multiple images in one batched call.
        """
        if not image_ids:
            return []

        GPUManager.validate_inference_gpu()

        model_record = db.query(Model).filter(Model.id == model_id).first()
        if not model_record:
            raise ValueError(f"Model {model_id} not found")

        if ordered_images is None:
            image_records = db.query(Image).filter(Image.id.in_(image_ids)).all()
            image_map = {img.id: img for img in image_records}
            missing_ids = [img_id for img_id in image_ids if img_id not in image_map]
            if missing_ids:
                raise ValueError(f"Images not found: {missing_ids}")
            ordered_images = [image_map[img_id] for img_id in image_ids]
        else:
            provided_map = {img.id: img for img in ordered_images}
            missing_ids = [img_id for img_id in image_ids if img_id not in provided_map]
            if missing_ids:
                raise ValueError(f"Images not found: {missing_ids}")
            ordered_images = [provided_map[img_id] for img_id in image_ids]

        predict_source: List[Any]
        if source_inputs is not None:
            if len(source_inputs) != len(ordered_images):
                raise ValueError(
                    f"source_inputs length mismatch: expected {len(ordered_images)}, got {len(source_inputs)}"
                )
            predict_source = source_inputs
        else:
            # Decode once per request/chunk to reduce file-loader overhead in Ultralytics.
            predict_source = []
            for image_record in ordered_images:
                src = cv2.imread(image_record.file_path)
                if src is None:
                    raise ValueError(f"Failed to read image: {image_record.file_path}")
                predict_source.append(src)
        model = cls._load_model(model_id, db)

        raw_results = model.model.predict(
            source=predict_source,
            conf=conf_threshold,
            iou=iou_threshold,
            device=model.device,
            batch=batch_size,
            half=(torch.cuda.is_available() and model.device != "cpu"),
            verbose=False
        )
        raw_results = list(raw_results)

        if len(raw_results) != len(ordered_images):
            raise RuntimeError(
                f"Batch inference output mismatch: expected {len(ordered_images)}, got {len(raw_results)}"
            )

        response_rows: List[Dict[str, Any]] = []
        db_rows: List[InferenceResult] = []
        draw_tasks: List[Tuple[int, Any]] = []

        for image_record, raw_result in zip(ordered_images, raw_results):
            detections = cls._extract_detections_from_result(model_record.task_type, raw_result)

            speed = getattr(raw_result, "speed", {}) or {}
            inference_time_ms = float(
                speed.get("preprocess", 0.0) + speed.get("inference", 0.0) + speed.get("postprocess", 0.0)
            )

            result_image_path = None

            row = {
                "image_id": image_record.id,
                "inference_result_id": None,
                "detections": detections,
                "result_image_path": result_image_path,
                "inference_time_ms": inference_time_ms,
                "detection_count": len(detections),
            }
            response_rows.append(row)

            if save_result_image:
                source_img = getattr(raw_result, "orig_img", None)
                if model_record.task_type == "obb":
                    draw_tasks.append((
                        len(response_rows) - 1,
                        (
                            cls._draw_obb_detections,
                            {
                                "image_path": image_record.file_path,
                                "detections": detections,
                                "output_dir": Path(settings.RESULTS_DIR) / "inference" / str(model_id),
                                "filename": cls._build_result_filename(image_record, model_id),
                                "conf_threshold": conf_threshold,
                                "overlap_iou": None,
                                "image_array": source_img,
                            },
                        ),
                    ))
                else:
                    draw_tasks.append((
                        len(response_rows) - 1,
                        (
                            cls._draw_bbox_detections,
                            {
                                "image_path": image_record.file_path,
                                "detections": detections,
                                "output_dir": Path(settings.RESULTS_DIR) / "inference" / str(model_id),
                                "filename": cls._build_result_filename(image_record, model_id),
                                "image_array": source_img,
                            },
                        ),
                    ))

            if save_to_db:
                db_rows.append(
                    InferenceResult(
                        model_id=model_id,
                        image_id=image_record.id,
                        detections=detections,
                        result_image_path=result_image_path,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        inference_time_ms=inference_time_ms
                    )
                )

        if draw_tasks:
            workers = min(4, len(draw_tasks))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures: List[Tuple[int, Any]] = []
                for row_index, (draw_fn, kwargs) in draw_tasks:
                    futures.append((row_index, pool.submit(draw_fn, **kwargs)))
                for row_index, future in futures:
                    response_rows[row_index]["result_image_path"] = future.result()

        if save_to_db and db_rows:
            db.add_all(db_rows)
            db.commit()
            for saved_row, response_row in zip(db_rows, response_rows):
                db.refresh(saved_row)
                response_row["inference_result_id"] = saved_row.id

        return response_rows

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
        cls._preload_models_parallel(db, [model_id_1, model_id_2])

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
    def compare_models_batch(
        cls,
        db: Session,
        model_id_1: int,
        model_id_2: int,
        image_ids: List[int],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        batch_size: int = 10,
        save_result_image: bool = False,
        save_to_db: bool = False,
        include_detections: bool = False
    ) -> Dict[str, Any]:
        """
        Compare two models across multiple images.
        """
        model_1_record = db.query(Model).filter(Model.id == model_id_1).first()
        if not model_1_record:
            raise ValueError(f"Model {model_id_1} not found")
        model_2_record = db.query(Model).filter(Model.id == model_id_2).first()
        if not model_2_record:
            raise ValueError(f"Model {model_id_2} not found")

        image_records = db.query(Image).filter(Image.id.in_(image_ids)).all()
        image_map = {img.id: img for img in image_records}
        missing_ids = [img_id for img_id in image_ids if img_id not in image_map]
        if missing_ids:
            raise ValueError(f"Images not found: {missing_ids}")
        ordered_images = [image_map[img_id] for img_id in image_ids]

        # Decode source images once and share for both model runs.
        source_images: List[np.ndarray] = []
        for image_record in ordered_images:
            src = cv2.imread(image_record.file_path)
            if src is None:
                raise ValueError(f"Failed to read image: {image_record.file_path}")
            source_images.append(src)

        cls._preload_models_parallel(db, [model_id_1, model_id_2])

        # Fast path:
        # keep GPU busy with model-B inference while model-A result rendering happens on CPU threads.
        # DB persistence path remains on the existing synchronous flow for consistency.
        if save_result_image and not save_to_db:
            workers = max(1, min(4, len(ordered_images)))
            with ThreadPoolExecutor(max_workers=workers) as draw_pool:
                model_1_results = cls.predict_batch(
                    db=db,
                    model_id=model_id_1,
                    image_ids=image_ids,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    batch_size=batch_size,
                    save_result_image=False,
                    save_to_db=False,
                    ordered_images=ordered_images,
                    source_inputs=source_images,
                )
                model_1_draw_jobs = cls._schedule_result_image_draws(
                    pool=draw_pool,
                    model_id=model_id_1,
                    task_type=model_1_record.task_type,
                    ordered_images=ordered_images,
                    source_images=source_images,
                    rows=model_1_results,
                    conf_threshold=conf_threshold,
                )

                model_2_results = cls.predict_batch(
                    db=db,
                    model_id=model_id_2,
                    image_ids=image_ids,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    batch_size=batch_size,
                    save_result_image=False,
                    save_to_db=False,
                    ordered_images=ordered_images,
                    source_inputs=source_images,
                )
                model_2_draw_jobs = cls._schedule_result_image_draws(
                    pool=draw_pool,
                    model_id=model_id_2,
                    task_type=model_2_record.task_type,
                    ordered_images=ordered_images,
                    source_images=source_images,
                    rows=model_2_results,
                    conf_threshold=conf_threshold,
                )

                cls._resolve_result_image_draws(model_1_results, model_1_draw_jobs)
                cls._resolve_result_image_draws(model_2_results, model_2_draw_jobs)
        else:
            model_1_results = cls.predict_batch(
                db=db,
                model_id=model_id_1,
                image_ids=image_ids,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                batch_size=batch_size,
                save_result_image=save_result_image,
                save_to_db=save_to_db,
                ordered_images=ordered_images,
                source_inputs=source_images,
            )
            model_2_results = cls.predict_batch(
                db=db,
                model_id=model_id_2,
                image_ids=image_ids,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                batch_size=batch_size,
                save_result_image=save_result_image,
                save_to_db=save_to_db,
                ordered_images=ordered_images,
                source_inputs=source_images,
            )

        joined_results = []
        for m1, m2 in zip(model_1_results, model_2_results):
            joined_results.append({
                "image_id": m1["image_id"],
                "model_1": {
                    "model_id": model_id_1,
                    "inference_result_id": m1["inference_result_id"],
                    "detections": m1["detections"] if include_detections else [],
                    "result_image_path": m1["result_image_path"],
                    "inference_time_ms": m1["inference_time_ms"],
                    "detection_count": m1["detection_count"],
                },
                "model_2": {
                    "model_id": model_id_2,
                    "inference_result_id": m2["inference_result_id"],
                    "detections": m2["detections"] if include_detections else [],
                    "result_image_path": m2["result_image_path"],
                    "inference_time_ms": m2["inference_time_ms"],
                    "detection_count": m2["detection_count"],
                }
            })

        return {"results": joined_results}

    @classmethod
    def _schedule_result_image_draws(
        cls,
        pool: ThreadPoolExecutor,
        model_id: int,
        task_type: str,
        ordered_images: List[Image],
        source_images: List[np.ndarray],
        rows: List[Dict[str, Any]],
        conf_threshold: float
    ) -> List[Tuple[int, Any]]:
        jobs: List[Tuple[int, Any]] = []
        output_dir = Path(settings.RESULTS_DIR) / "inference" / str(model_id)

        for idx, (image_record, source_img, row) in enumerate(zip(ordered_images, source_images, rows)):
            filename = cls._build_result_filename(image_record, model_id)
            if task_type == "obb":
                future = pool.submit(
                    cls._draw_obb_detections,
                    image_path=image_record.file_path,
                    detections=row["detections"],
                    output_dir=output_dir,
                    filename=filename,
                    conf_threshold=conf_threshold,
                    overlap_iou=None,
                    image_array=source_img,
                )
            else:
                future = pool.submit(
                    cls._draw_bbox_detections,
                    image_path=image_record.file_path,
                    detections=row["detections"],
                    output_dir=output_dir,
                    filename=filename,
                    image_array=source_img,
                )
            jobs.append((idx, future))

        return jobs

    @staticmethod
    def _resolve_result_image_draws(rows: List[Dict[str, Any]], jobs: List[Tuple[int, Any]]) -> None:
        for row_idx, future in jobs:
            rows[row_idx]["result_image_path"] = future.result()

    @classmethod
    def _draw_bbox_detections(
        cls,
        image_path: str,
        detections: List[Dict],
        output_dir: Path,
        filename: str,
        image_array: Optional[np.ndarray] = None
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

        # Use provided image to avoid disk re-read when available.
        if image_array is not None:
            img = image_array.copy()
        else:
            img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Render on review canvas first to avoid text/box quality loss from post-resize.
        img, scale, pad_x, pad_y = cls._resize_to_square_review_with_transform(
            img, cls._result_review_size
        )
        h, w = img.shape[:2]

        # Draw each detection
        for det in detections:
            bbox = det["bbox"]  # [x1, y1, x2, y2]
            x1 = int(round(float(bbox[0]) * scale + pad_x))
            y1 = int(round(float(bbox[1]) * scale + pad_y))
            x2 = int(round(float(bbox[2]) * scale + pad_x))
            y2 = int(round(float(bbox[3]) * scale + pad_y))
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))
            conf = det["confidence"]
            class_id = det["class_id"]

            # Draw rectangle
            color = cls._get_class_color(class_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, cls._result_box_thickness)

            # Draw label
            label = f"Class {class_id}: {conf:.2f}"
            label_x = max(8, x1)
            label_y = y1 - int(30 * cls._result_text_scale_bbox)
            if label_y < 6:
                label_y = min(h - 6, y1 + 6)
            cls._draw_label_with_bg(
                img=img,
                label=label,
                x=label_x,
                y=label_y,
                accent_color=color,
                font_scale=cls._result_text_scale_bbox,
                text_thickness=cls._result_text_thickness,
            )

        cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, cls._result_jpeg_quality])

        return str(output_path)

    @classmethod
    def _draw_obb_detections(
        cls,
        image_path: str,
        detections: List[Dict],
        output_dir: Path,
        filename: str,
        conf_threshold: float = 0.25,
        overlap_iou: Optional[float] = None,
        image_array: Optional[np.ndarray] = None
    ) -> str:
        """
        Draw OBB detections on image.
        Uses optional custom polygon-NMS when overlap_iou is provided.

        Args:
            image_path: Path to input image
            detections: List of OBB detection dictionaries
            output_dir: Output directory
            filename: Output filename
            conf_threshold: Confidence threshold for drawing
            overlap_iou: IOU threshold for custom NMS (None disables extra suppression)

        Returns:
            Path to result image
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        # Use provided image to avoid disk re-read when available.
        if image_array is not None:
            img = image_array.copy()
        else:
            img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Render on review canvas first to avoid text/box quality loss from post-resize.
        img, scale, pad_x, pad_y = cls._resize_to_square_review_with_transform(
            img, cls._result_review_size
        )
        h, w = img.shape[:2]

        if not detections:
            cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, cls._result_jpeg_quality])
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
            cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, cls._result_jpeg_quality])
            return str(output_path)

        if overlap_iou is not None and overlap_iou > 0:
            keep_idxs = cls._suppress_obb(polys, confs, iou_th=overlap_iou)
        else:
            keep_idxs = list(range(len(polys)))

        # Draw
        for i in keep_idxs:
            pts = polys[i].copy()
            pts[:, 0] = pts[:, 0] * scale + pad_x
            pts[:, 1] = pts[:, 1] * scale + pad_y
            pts = np.round(pts).astype(int)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            cls_id = int(class_ids[i])
            conf = float(confs[i])

            color = cls._get_class_color(cls_id)
            label = f"Class {cls_id}: {conf:.2f}"

            # Draw polygon
            cv2.polylines(img, [pts], True, color, cls._result_box_thickness)

            # Draw label near top-most polygon point for readability.
            top_idx = int(np.argmin(pts[:, 1]))
            label_x = max(8, int(pts[top_idx, 0]) + 4)
            label_y = int(pts[top_idx, 1]) - int(30 * cls._result_text_scale_obb)
            if label_y < 6:
                label_y = min(h - 6, int(pts[top_idx, 1]) + 6)
            cls._draw_label_with_bg(
                img=img,
                label=label,
                x=label_x,
                y=label_y,
                accent_color=color,
                font_scale=cls._result_text_scale_obb,
                text_thickness=cls._result_text_thickness,
            )

        cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, cls._result_jpeg_quality])
        return str(output_path)

    @classmethod
    def _resize_to_square_review_with_transform(
        cls,
        img: np.ndarray,
        target: int = 1024
    ) -> Tuple[np.ndarray, float, int, int]:
        """
        Resize image to a square review frame while preserving aspect ratio.
        Returns resized canvas and transform values used for coordinate mapping.
        """
        if img is None or img.size == 0:
            return img, 1.0, 0, 0

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return img, 1.0, 0, 0

        scale = min(target / float(w), target / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

        canvas = np.zeros((target, target, 3), dtype=np.uint8)
        x = (target - new_w) // 2
        y = (target - new_h) // 2
        canvas[y:y + new_h, x:x + new_w] = resized
        return canvas, scale, x, y

    @classmethod
    def _resize_to_square_review(cls, img: np.ndarray, target: int = 1024) -> np.ndarray:
        """Backwards-compatible helper returning only review canvas image."""
        canvas, _, _, _ = cls._resize_to_square_review_with_transform(img, target)
        return canvas

    @classmethod
    def _draw_label_with_bg(
        cls,
        img: np.ndarray,
        label: str,
        x: int,
        y: int,
        accent_color: Tuple[int, int, int],
        font_scale: float,
        text_thickness: int,
    ) -> None:
        """Draw high-contrast label text with rounded-looking background and border."""
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        pad_x = 8
        pad_y = 6
        box_w = text_w + pad_x * 2
        box_h = text_h + baseline + pad_y * 2

        x1 = max(0, min(w - box_w - 1, x))
        y1 = max(0, min(h - box_h - 1, y))
        x2 = min(w - 1, x1 + box_w)
        y2 = min(h - 1, y1 + box_h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (10, 18, 33), thickness=-1)
        cv2.rectangle(img, (x1, y1), (x2, y2), accent_color, thickness=1)

        text_org = (x1 + pad_x, y1 + pad_y + text_h)
        cv2.putText(
            img,
            label,
            text_org,
            font,
            font_scale,
            (0, 0, 0),
            text_thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            label,
            text_org,
            font,
            font_scale,
            (245, 248, 255),
            text_thickness,
            cv2.LINE_AA,
        )

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
