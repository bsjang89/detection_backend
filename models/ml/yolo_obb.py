from typing import Dict, Any, Optional, Callable, List, Union
from pathlib import Path
import torch
import numpy as np
from ultralytics import YOLO

from .base import BaseDetectionModel


class YOLOOBBModel(BaseDetectionModel):
    """
    YOLO model for Oriented Bounding Box (OBB) detection.
    Integrates logic from detection_backend/Train_OBB.py
    """

    def __init__(self, model_type: str = "yolov8n-obb", device: Union[int, str] = 0):
        """
        Initialize YOLO OBB model.

        Args:
            model_type: Model variant (yolov8n-obb, yolov8s-obb, yolov8m-obb, etc.)
            device: GPU device ID or "cpu"
        """
        # Ensure model_type has -obb suffix
        if not model_type.endswith("-obb"):
            model_type = f"{model_type}-obb"

        super().__init__(model_type=model_type, task_type="obb", device=device)

    def load_pretrained(self, weights_path: Optional[str] = None):
        """
        Load pretrained YOLO OBB weights.

        Args:
            weights_path: Path to custom weights. If None, loads official pretrained.
        """
        weights_path = self.resolve_pretrained_weights(
            default_filename=f"{self.model_type}.pt",
            weights_path=weights_path
        )

        self.model = YOLO(weights_path)
        print(f"Loaded YOLO OBB model: {weights_path} on {self.get_device_name()}")

    def train(
        self,
        data_yaml: str,
        config: Dict[str, Any],
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train YOLO model for OBB detection.

        Args:
            data_yaml: Path to data.yaml
            config: Training config with keys:
                - epochs (default: 50)
                - batch (default: 16)
                - imgsz (default: 640)
                - lr0 (default: 0.005)
                - mosaic (default: 0.5)
                - mixup (default: 0.0)
                - close_mosaic (default: 10)
                - cache (default: True)
                - workers (default: 8)
                - patience (default: 50)
            callback: Function called after each epoch

        Returns:
            Training results dictionary
        """
        if self.model is None:
            self.load_pretrained()

        # Prepare training arguments (OBB-specific defaults from Train_OBB.py)
        train_args = {
            "data": data_yaml,
            "epochs": config.get("epochs", 50),
            "imgsz": config.get("imgsz", 640),
            "batch": config.get("batch", 16),
            "device": self.device,
            "lr0": config.get("lr0", 0.005),  # Lower LR for OBB
            "mosaic": config.get("mosaic", 0.5),
            "mixup": config.get("mixup", 0.0),
            "close_mosaic": config.get("close_mosaic", 10),  # Close mosaic in last 10 epochs
            "cache": "ram" if config.get("cache", True) else False,
            "workers": config.get("workers", 4),
            "patience": config.get("patience", 50),
            "verbose": config.get("verbose", True),
            "optimizer": config.get("optimizer", "auto"),
            "pretrained": config.get("pretrained", True),
            "amp": config.get("amp", True),
            "deterministic": config.get("deterministic", False),
            "plots": config.get("plots", False),
        }

        # Add callback if provided
        if callback:
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
                should_stop = callback(trainer.epoch, metrics)
                if should_stop:
                    trainer.stop = True

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
        Run inference on an image for OBB detection.

        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold
            iou_threshold: NMS IOU threshold

        Returns:
            List of detections with OBB coordinates
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
            if hasattr(result, 'obb') and result.obb is not None:
                obbs = result.obb
                for i in range(len(obbs)):
                    # Get OBB coordinates (8 points: x1,y1,x2,y2,x3,y3,x4,y4)
                    xyxyxyxy = obbs.xyxyxyxy[i].cpu().numpy().flatten().tolist()

                    detection = {
                        "class_id": int(obbs.cls[i]),
                        "confidence": float(obbs.conf[i]),
                        "obb": xyxyxyxy,  # 8 coordinates
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
        print(f"Loaded YOLO OBB model from: {weights_path}")
