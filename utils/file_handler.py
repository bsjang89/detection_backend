from pathlib import Path
from typing import List
import shutil
from PIL import Image as PILImage

from config.settings import settings


def get_project_dir(project_id: int) -> Path:
    """Get project directory path"""
    return Path(settings.UPLOAD_DIR) / str(project_id)


def get_project_images_dir(project_id: int) -> Path:
    """Get project images directory"""
    return get_project_dir(project_id) / "images"


def get_project_thumbnails_dir(project_id: int) -> Path:
    """Get project thumbnail directory"""
    return get_project_dir(project_id) / "thumbnails"


def safe_filename(filename: str) -> str:
    """Normalize uploaded filename and drop any directory part"""
    return Path(filename).name


def get_thumbnail_filename(filename: str) -> str:
    """Build a deterministic thumbnail filename"""
    return f"{safe_filename(filename)}.thumb.jpg"


def get_thumbnail_path(project_id: int, filename: str) -> Path:
    """Get absolute thumbnail path for a project image"""
    return get_project_thumbnails_dir(project_id) / get_thumbnail_filename(filename)


def create_thumbnail(source_path: Path, thumbnail_path: Path, max_size: int = 256) -> None:
    """Create a JPEG thumbnail that fits in max_size x max_size"""
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    resample = PILImage.Resampling.LANCZOS if hasattr(PILImage, "Resampling") else PILImage.LANCZOS

    with PILImage.open(source_path) as img:
        thumb = img.copy()
        thumb.thumbnail((max_size, max_size), resample)
        if thumb.mode not in ("RGB", "L"):
            thumb = thumb.convert("RGB")
        thumb.save(thumbnail_path, format="JPEG", quality=85, optimize=True)


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
