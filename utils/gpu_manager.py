import torch
from typing import Dict, Union
from config.settings import settings


class GPUManager:
    """
    Manages GPU allocation for training and inference.
    - GPU 0: Training only
    - GPU 1: Inference only
    """

    gpu_assignments = {
        settings.TRAINING_GPU_ID: 'training',
        settings.INFERENCE_GPU_ID: 'inference'
    }

    @classmethod
    def get_training_gpu(cls) -> int:
        """Get GPU ID for training"""
        return settings.TRAINING_GPU_ID

    @classmethod
    def get_training_device(cls) -> Union[int, str]:
        """Get training device, falling back to CPU when configured GPU is unavailable."""
        gpu_id = cls.get_training_gpu()
        return gpu_id if cls.check_gpu_available(gpu_id) else "cpu"

    @classmethod
    def get_inference_gpu(cls) -> int:
        """Get GPU ID for inference"""
        return settings.INFERENCE_GPU_ID

    @classmethod
    def check_gpu_available(cls, gpu_id: int) -> bool:
        """
        Check if GPU is available.

        Args:
            gpu_id: GPU device ID

        Returns:
            True if GPU is available
        """
        if not torch.cuda.is_available():
            return False

        return gpu_id < torch.cuda.device_count()

    @classmethod
    def get_gpu_memory_info(cls, gpu_id: int) -> Dict[str, float]:
        """
        Get GPU memory information.

        Args:
            gpu_id: GPU device ID

        Returns:
            Dictionary with memory info in GB
        """
        if not cls.check_gpu_available(gpu_id):
            return {
                "total_gb": 0,
                "allocated_gb": 0,
                "free_gb": 0
            }

        torch.cuda.set_device(gpu_id)
        total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)

        return {
            "total_gb": round(total, 2),
            "allocated_gb": round(allocated, 2),
            "free_gb": round(total - allocated, 2)
        }

    @classmethod
    def get_gpu_name(cls, gpu_id: int) -> str:
        """Get GPU device name"""
        if cls.check_gpu_available(gpu_id):
            return torch.cuda.get_device_name(gpu_id)
        return "No GPU"

    @classmethod
    def clear_cache(cls, gpu_id: int):
        """Clear GPU cache"""
        if cls.check_gpu_available(gpu_id):
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()

    @classmethod
    def validate_training_gpu(cls) -> bool:
        """Validate that training GPU is available"""
        gpu_id = cls.get_training_gpu()
        if not cls.check_gpu_available(gpu_id):
            raise RuntimeError(
                f"Training GPU {gpu_id} is not available. "
                f"Available GPUs: {torch.cuda.device_count()}"
            )
        return True

    @classmethod
    def validate_inference_gpu(cls) -> bool:
        """Validate that inference GPU is available"""
        gpu_id = cls.get_inference_gpu()
        if not cls.check_gpu_available(gpu_id):
            raise RuntimeError(
                f"Inference GPU {gpu_id} is not available. "
                f"Available GPUs: {torch.cuda.device_count()}"
            )
        return True
