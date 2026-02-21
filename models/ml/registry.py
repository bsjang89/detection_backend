from typing import Dict, Type, Union
from .base import BaseDetectionModel
from .yolo_bbox import YOLOBBoxModel
from .yolo_obb import YOLOOBBModel


class ModelRegistry:
    """
    Registry for managing different detection model types.
    Allows easy extension to support more models in the future.
    """

    _models: Dict[str, Type[BaseDetectionModel]] = {
        "bbox": YOLOBBoxModel,
        "obb": YOLOOBBModel,
    }

    @classmethod
    def get_model_class(cls, task_type: str) -> Type[BaseDetectionModel]:
        """
        Get model class for a given task type.

        Args:
            task_type: 'bbox' or 'obb'

        Returns:
            Model class

        Raises:
            ValueError: If task_type not supported
        """
        if task_type not in cls._models:
            raise ValueError(
                f"Unsupported task type: {task_type}. "
                f"Supported types: {list(cls._models.keys())}"
            )

        return cls._models[task_type]

    @classmethod
    def create_model(
        cls,
        task_type: str,
        model_type: str,
        device: Union[int, str] = 0
    ) -> BaseDetectionModel:
        """
        Factory method to create a detection model.

        Args:
            task_type: 'bbox' or 'obb'
            model_type: Model variant (e.g., 'yolov8n', 'yolov8s', etc.)
            device: GPU device ID or "cpu"

        Returns:
            Instantiated model

        Example:
            >>> model = ModelRegistry.create_model('bbox', 'yolov8n', device=0)
            >>> model = ModelRegistry.create_model('obb', 'yolov8m-obb', device=1)
        """
        model_class = cls.get_model_class(task_type)
        return model_class(model_type=model_type, device=device)

    @classmethod
    def register_model(cls, task_type: str, model_class: Type[BaseDetectionModel]):
        """
        Register a new model type.

        Args:
            task_type: Unique identifier for the task
            model_class: Model class inheriting from BaseDetectionModel

        Example:
            >>> class CustomModel(BaseDetectionModel):
            ...     pass
            >>> ModelRegistry.register_model('custom', CustomModel)
        """
        cls._models[task_type] = model_class

    @classmethod
    def list_supported_types(cls) -> list:
        """Get list of supported task types"""
        return list(cls._models.keys())
