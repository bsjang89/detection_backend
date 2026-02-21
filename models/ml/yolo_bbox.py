from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import torch
from ultralytics import YOLO

from .base import BaseDetectionModel


class YOLOBBoxModel(BaseDetectionModel):
    """
    YOLO model for regular bounding box detection.
    Integrates logic from detection_backend/train.py
    """

    def __init__(self, model_type: str = "yolov8n", device: int = 0):
        """
        Initialize YOLO BBox model.

        Args:
            model_type: Model variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            device: GPU device ID
        """
        super().__init__(model_type=model_type, task_type="bbox", device=device)

    def load_pretrained(self, weights_path: Optional[str] = None):
        """
        Load pretrained YOLO weights.

        Args:
            weights_path: Path to custom weights. If None, loads official pretrained.
        """
        if weights_path is None:
            # Load official pretrained model
            weights_path = f"{self.model_type}.pt"

        self.model = YOLO(weights_path)
        print(f"Loaded YOLO BBox model: {weights_path} on {self.get_device_name()}")

    def train(
        self,
        data_yaml: str,
        config: Dict[str, Any],
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train YOLO model for bbox detection.

        Args:
            data_yaml: Path to data.yaml
            config: Training config with keys:
                - epochs (default: 100)
                - batch (default: 16)
                - imgsz (default: 640)
                - lr0 (default: 0.01)
                - mosaic (default: 0.5)
                - mixup (default: 0.0)
                - close_mosaic (default: 0)
                - cache (default: True)
                - workers (default: 8)
                - patience (default: 50)
            callback: Function called after each epoch

        Returns:
            Training results dictionary
        """
        if self.model is None:
            self.load_pretrained()

        # Prepare training arguments
        train_args = {
            "data": data_yaml,
            "epochs": config.get("epochs", 100),
            "imgsz": config.get("imgsz", 640),
            "batch": config.get("batch", 16),
            "device": self.device,
            "lr0": config.get("lr0", 0.01),
            "mosaic": config.get("mosaic", 0.5),
            "mixup": config.get("mixup", 0.0),
            "close_mosaic": config.get("close_mosaic", 0),
            "cache": "ram" if config.get("cache", True) else False,
            "workers": config.get("workers", 8),
            "patience": config.get("patience", 50),
            "verbose": config.get("verbose", True),
            "optimizer": config.get("optimizer", "auto"),
            "pretrained": config.get("pretrained", True),
        }

        # Add callback if provided
        if callback:
            # Ultralytics callback wrapper
            def on_train_epoch_end(trainer):
                metrics = {
                    "epoch": trainer.epoch,
                    "train_box_loss": trainer.metrics.get("train/box_loss"),
                    "train_cls_loss": trainer.metrics.get("train/cls_loss"),
                    "train_dfl_loss": trainer.metrics.get("train/dfl_loss"),
                    "val_box_loss": trainer.metrics.get("val/box_loss"),
                    "val_cls_loss": trainer.metrics.get("val/cls_loss"),
                    "val_dfl_loss": trainer.metrics.get("val/dfl_loss"),
                    "map50": trainer.metrics.get("metrics/mAP50(B)"),
                    "map50_95": trainer.metrics.get("metrics/mAP50-95(B)"),
                    "precision": trainer.metrics.get("metrics/precision(B)"),
                    "recall": trainer.metrics.get("metrics/recall(B)"),
                    "learning_rate": trainer.optimizer.param_groups[0]['lr'],
                }
                callback(trainer.epoch, metrics)

            self.model.add_callback("on_train_epoch_end", on_train_epoch_end)

        # Train
        results = self.model.train(**train_args)

        # Extract final metrics
        final_results = {
            "best_weights": str(Path(results.save_dir) / "weights" / "best.pt"),
            "last_weights": str(Path(results.save_dir) / "weights" / "last.pt"),
            "save_dir": str(results.save_dir),
            "final_metrics": {
                "map50": results.results_dict.get("metrics/mAP50(B)"),
                "map50_95": results.results_dict.get("metrics/mAP50-95(B)"),
                "precision": results.results_dict.get("metrics/precision(B)"),
                "recall": results.results_dict.get("metrics/recall(B)"),
            }
        }

        return final_results

    def predict(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Run inference on an image.

        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold
            iou_threshold: NMS IOU threshold

        Returns:
            List of detections
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_pretrained() or load_model() first.")

        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )

        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                detection = {
                    "class_id": int(boxes.cls[i]),
                    "confidence": float(boxes.conf[i]),
                    "bbox": boxes.xyxy[i].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                }
                detections.append(detection)

        return detections

    def save_model(self, save_path: str):
        """Save model weights"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(save_path)

    def load_model(self, weights_path: str):
        """Load model from weights file"""
        self.model = YOLO(weights_path)
        print(f"Loaded YOLO BBox model from: {weights_path}")
