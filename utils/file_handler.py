from pathlib import Path
from typing import List
import shutil

from config.settings import settings


def get_project_dir(project_id: int) -> Path:
    """Get project directory path"""
    return Path(settings.UPLOAD_DIR) / str(project_id)


def get_project_images_dir(project_id: int) -> Path:
    """Get project images directory"""
    return get_project_dir(project_id) / "images"


def get_project_dataset_dir(project_id: int) -> Path:
    """Get project dataset directory"""
    return get_project_dir(project_id) / "dataset"


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_image_files(directory: str) -> List[str]:
    """List all image files in a directory"""
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    dir_path = Path(directory)

    if not dir_path.exists():
        return []

    return sorted([
        str(f) for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in img_extensions
    ])


def cleanup_directory(directory: str):
    """Remove a directory and all its contents"""
    dir_path = Path(directory)
    if dir_path.exists():
        shutil.rmtree(dir_path)
