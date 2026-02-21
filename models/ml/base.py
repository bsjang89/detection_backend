from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import torch


class BaseDetectionModel(ABC):
    """
    Abstract base class for object detection models.
    Provides a unified interface for training, inference, and model management.
    """

    def __init__(self, model_type: str, task_type: str, device: int = 0):
        """
        Initialize detection model.

        Args:
            model_type: Model architecture (e.g., 'yolov8n', 'yolov8s', 'yolov8m')
            task_type: Task type ('bbox' or 'obb')
            device: GPU device ID
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
                      Signature: callback(epoch: int, metrics: dict)

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
        if torch.cuda.is_available() and self.device >= 0:
            return f"cuda:{self.device} ({torch.cuda.get_device_name(self.device)})"
        return "cpu"

    def check_gpu_memory(self) -> Dict[str, float]:
        """
        Check GPU memory usage.

        Returns:
            Dictionary with 'used_gb' and 'total_gb'
        """
        if torch.cuda.is_available() and self.device >= 0:
            torch.cuda.set_device(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            return {
                "used_gb": round(allocated, 2),
                "total_gb": round(total, 2),
                "free_gb": round(total - allocated, 2)
            }
        return {"used_gb": 0, "total_gb": 0, "free_gb": 0}
