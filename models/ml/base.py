from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List, Union
from pathlib import Path
import shutil
import torch
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

from config.settings import settings


class BaseDetectionModel(ABC):
    """
    Abstract base class for object detection models.
    Provides a unified interface for training, inference, and model management.
    """

    def __init__(self, model_type: str, task_type: str, device: Union[int, str] = 0):
        """
        Initialize detection model.

        Args:
            model_type: Model architecture (e.g., 'yolov8n', 'yolov8s', 'yolov8m')
            task_type: Task type ('bbox' or 'obb')
            device: GPU device ID or "cpu"
        """
        self.model_type = model_type
        self.task_type = task_type
        self.device = device
        self.model = None

    @abstractmethod
    def load_pretrained(self, weights_path: Optional[str] = None):
        """
        Load pretrained weights or initialize model.

        Args:
            weights_path: Path to weights file. If None, loads official pretrained weights.
        """
        pass

    @abstractmethod
    def train(
        self,
        data_yaml: str,
        config: Dict[str, Any],
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            data_yaml: Path to YOLO data.yaml file
            config: Training configuration (epochs, batch, imgsz, lr0, etc.)
            callback: Optional callback function called after each epoch
                      Signature: callback(epoch: int, metrics: dict) -> bool
                      Return True to request early stop.

        Returns:
            Dictionary containing training results and metrics
        """
        pass

    @abstractmethod
    def predict(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Run inference on an image.

        Args:
            image_path: Path to image file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS

        Returns:
            List of detections. Each detection is a dict with:
            - class_id: int
            - confidence: float
            - bbox: [x1, y1, x2, y2] for regular bbox
            - obb: [x1, y1, x2, y2, x3, y3, x4, y4] for oriented bbox
        """
        pass

    @abstractmethod
    def save_model(self, save_path: str):
        """
        Save model weights.

        Args:
            save_path: Path to save the model
        """
        pass

    @abstractmethod
    def load_model(self, weights_path: str):
        """
        Load model weights from file.

        Args:
            weights_path: Path to weights file
        """
        pass

    def get_device_name(self) -> str:
        """Get the name of the device being used"""
        if isinstance(self.device, int) and torch.cuda.is_available() and self.device >= 0:
            return f"cuda:{self.device} ({torch.cuda.get_device_name(self.device)})"
        return "cpu"

    def check_gpu_memory(self) -> Dict[str, float]:
        """
        Check GPU memory usage.

        Returns:
            Dictionary with 'used_gb' and 'total_gb'
        """
        if isinstance(self.device, int) and torch.cuda.is_available() and self.device >= 0:
            torch.cuda.set_device(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            return {
                "used_gb": round(allocated, 2),
                "total_gb": round(total, 2),
                "free_gb": round(total - allocated, 2)
            }
        return {"used_gb": 0, "total_gb": 0, "free_gb": 0}

    def resolve_pretrained_weights(self, default_filename: str, weights_path: Optional[str] = None) -> str:
        """
        Resolve pretrained weights path.

        If `weights_path` is provided, use it directly.
        Otherwise, keep pretrained files under settings.NETWORK_DIR and download there if missing.
        """
        if weights_path:
            return str(weights_path)

        network_dir = Path(settings.NETWORK_DIR)
        network_dir.mkdir(parents=True, exist_ok=True)

        target = network_dir / default_filename
        if target.exists():
            return str(target)

        downloaded = attempt_download_asset(str(target))
        downloaded_path = Path(downloaded)
        if downloaded_path.exists():
            return str(downloaded_path)

        # Fallback: let Ultralytics resolve built-in model name, then copy into network folder.
        fallback_name = default_filename
        try:
            probe = YOLO(fallback_name)
            ckpt_path = Path(getattr(probe, "ckpt_path", fallback_name))
            if ckpt_path.exists():
                if ckpt_path.resolve() != target.resolve():
                    shutil.copy2(ckpt_path, target)
                return str(target if target.exists() else ckpt_path)
        except Exception:
            pass

        return str(target if target.exists() else fallback_name)
