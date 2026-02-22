import logging
from pathlib import Path
from typing import Dict, Set, List

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .settings import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
logger = logging.getLogger(__name__)


def get_db():
    """Database dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema_compatibility()


def _safe_exec(ddl_sql: str) -> None:
    """Execute best-effort DDL without stopping app startup."""
    try:
        with engine.begin() as conn:
            conn.execute(text(ddl_sql))
    except SQLAlchemyError as exc:
        logger.warning("DDL skipped: %s (%s)", ddl_sql, exc)


def _resolve_duplicate_filename(filename: str, used_names: Set[str], path_exists) -> str:
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    idx = 1
    candidate = filename
    while candidate in used_names or path_exists(candidate):
        candidate = f"{stem}({idx}){suffix}"
        idx += 1
    return candidate


def _artifact_paths(project_id: int, filename: str) -> Dict[str, Path]:
    base_dir = Path(settings.UPLOAD_DIR) / str(project_id)
    return {
        "thumbnail": base_dir / "thumbnails" / f"{filename}.thumb.jpg",
        "viewer": base_dir / "viewers" / f"{filename}.view.jpg",
    }


def _rename_if_exists(old_path: Path, new_path: Path) -> None:
    if not old_path.exists():
        return
    new_path.parent.mkdir(parents=True, exist_ok=True)
    if new_path.exists():
        return
    old_path.rename(new_path)


def _deduplicate_image_filenames() -> None:
    """
    Normalize duplicate (project_id, filename) rows so unique index can be created safely.
    Keeps the oldest row filename and renames later duplicates with (N) suffix.
    """
    try:
        with engine.begin() as conn:
            rows = conn.execute(
                text("SELECT id, project_id, filename, file_path FROM images ORDER BY project_id, id")
            ).mappings().all()

            used_by_project: Dict[int, Set[str]] = {}
            renamed: List[Dict[str, str]] = []

            for row in rows:
                image_id = int(row["id"])
                project_id = int(row["project_id"])
                filename = str(row["filename"])
                file_path = str(row["file_path"])

                used = used_by_project.setdefault(project_id, set())
                if filename not in used:
                    used.add(filename)
                    continue

                old_file_path = Path(file_path)
                images_dir = old_file_path.parent if old_file_path.parent.name == "images" else (Path(settings.UPLOAD_DIR) / str(project_id) / "images")

                def candidate_exists(candidate_name: str) -> bool:
                    return (images_dir / candidate_name).exists()

                new_filename = _resolve_duplicate_filename(filename, used, candidate_exists)
                new_file_path = images_dir / new_filename

                # Rename source image file if needed
                if old_file_path.exists():
                    new_file_path.parent.mkdir(parents=True, exist_ok=True)
                    if not new_file_path.exists():
                        old_file_path.rename(new_file_path)
                    else:
                        # Fallback: pick another candidate that does not exist on disk
                        new_filename = _resolve_duplicate_filename(
                            new_filename,
                            used,
                            candidate_exists,
                        )
                        new_file_path = images_dir / new_filename
                        old_file_path.rename(new_file_path)

                # Rename derived files (thumbnail/viewer) if present
                old_artifacts = _artifact_paths(project_id, filename)
                new_artifacts = _artifact_paths(project_id, new_filename)
                _rename_if_exists(old_artifacts["thumbnail"], new_artifacts["thumbnail"])
                _rename_if_exists(old_artifacts["viewer"], new_artifacts["viewer"])

                conn.execute(
                    text("UPDATE images SET filename = :filename, file_path = :file_path WHERE id = :id"),
                    {
                        "id": image_id,
                        "filename": new_filename,
                        "file_path": str(new_file_path),
                    },
                )
                used.add(new_filename)
                renamed.append({"id": str(image_id), "from": filename, "to": new_filename})

            if renamed:
                logger.warning(
                    "Deduplicated %d image filename rows before unique index creation",
                    len(renamed),
                )
    except SQLAlchemyError as exc:
        logger.warning("Image filename dedup skipped (%s)", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Image filename dedup encountered non-SQL error (%s)", exc)


def ensure_runtime_schema_compatibility() -> None:
    """
    Apply additive schema changes for already-initialized databases.
    This keeps local/dev instances working without a dedicated migration tool.
    """
    dialect = engine.dialect.name

    if dialect == "postgresql":
        _safe_exec(
            "ALTER TABLE images "
            "ADD COLUMN IF NOT EXISTS annotation_version INTEGER NOT NULL DEFAULT 0"
        )
        _deduplicate_image_filenames()
        _safe_exec(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_images_project_filename "
            "ON images (project_id, filename)"
        )
        _safe_exec(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_classes_project_class_id "
            "ON classes (project_id, class_id)"
        )
        _safe_exec(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_classes_project_class_name "
            "ON classes (project_id, class_name)"
        )
        return

    if dialect == "sqlite":
        _safe_exec(
            "ALTER TABLE images "
            "ADD COLUMN annotation_version INTEGER NOT NULL DEFAULT 0"
        )
        _deduplicate_image_filenames()
        _safe_exec(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_images_project_filename "
            "ON images (project_id, filename)"
        )
        _safe_exec(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_classes_project_class_id "
            "ON classes (project_id, class_id)"
        )
        _safe_exec(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_classes_project_class_name "
            "ON classes (project_id, class_name)"
        )
