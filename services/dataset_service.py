from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, List, Tuple
from pathlib import Path
import yaml
import shutil
import random

from models.database import Project, Image, Annotation, Class, SplitType
from config.settings import settings


class DatasetService:
    """
    Service for managing dataset preparation and splitting.
    """

    @staticmethod
    def split_dataset(
        db: Session,
        project_id: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Dict[str, int]:
        """
        Split labeled images into train/val/test sets.

        Args:
            db: Database session
            project_id: Project ID
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed for reproducibility

        Returns:
            Dictionary with counts for each split
        """
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("Train, val, and test ratios must sum to 1.0")

        # Get all labeled images (images with at least one annotation)
        labeled_image_ids = db.query(Image.id).join(Annotation).filter(
            Image.project_id == project_id
        ).distinct().all()

        labeled_image_ids = [img_id[0] for img_id in labeled_image_ids]

        if not labeled_image_ids:
            raise ValueError("No labeled images found for splitting")

        # Shuffle with seed
        random.seed(seed)
        random.shuffle(labeled_image_ids)

        # Calculate split indices
        total = len(labeled_image_ids)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_ids = labeled_image_ids[:train_end]
        val_ids = labeled_image_ids[train_end:val_end]
        test_ids = labeled_image_ids[val_end:]

        # Update split_type in database
        db.query(Image).filter(Image.id.in_(train_ids)).update(
            {Image.split_type: SplitType.TRAIN},
            synchronize_session=False
        )
        db.query(Image).filter(Image.id.in_(val_ids)).update(
            {Image.split_type: SplitType.VAL},
            synchronize_session=False
        )
        db.query(Image).filter(Image.id.in_(test_ids)).update(
            {Image.split_type: SplitType.TEST},
            synchronize_session=False
        )

        db.commit()

        return {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "total": total
        }

    @staticmethod
    def generate_yolo_dataset(
        db: Session,
        project_id: int,
        output_dir: str
    ) -> str:
        """
        Generate YOLO format dataset with data.yaml.

        Args:
            db: Database session
            project_id: Project ID
            output_dir: Output directory for dataset

        Returns:
            Path to generated data.yaml file
        """
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project {project_id} not found")

        dataset_path = Path(output_dir)
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Create directories
        for split in ["train", "val", "test"]:
            (dataset_path / split / "images").mkdir(parents=True, exist_ok=True)
            (dataset_path / split / "labels").mkdir(parents=True, exist_ok=True)

        # Get class mapping
        classes = db.query(Class).filter(Class.project_id == project_id).order_by(Class.class_id).all()
        class_names = [cls.class_name for cls in classes]

        # Process each split
        for split_type in [SplitType.TRAIN, SplitType.VAL, SplitType.TEST]:
            split_name = split_type.value

            images = db.query(Image).filter(
                Image.project_id == project_id,
                Image.split_type == split_type
            ).all()

            for image in images:
                # Copy image
                src_path = Path(image.file_path)
                dst_image_path = dataset_path / split_name / "images" / image.filename

                if src_path.exists():
                    shutil.copy2(src_path, dst_image_path)

                # Get annotations
                annotations = db.query(Annotation).filter(
                    Annotation.image_id == image.id
                ).all()

                # Write label file
                label_file = dataset_path / split_name / "labels" / f"{src_path.stem}.txt"

                with open(label_file, 'w') as f:
                    for ann in annotations:
                        # Get class_id from Class table
                        class_obj = db.query(Class).filter(Class.id == ann.class_id).first()
                        if not class_obj:
                            continue

                        # YOLO format
                        if project.task_type.value == "obb":
                            # OBB format: class_id cx cy w h rotation
                            f.write(f"{class_obj.class_id} {ann.cx} {ann.cy} {ann.width} {ann.height} {ann.rotation}\n")
                        else:
                            # BBox format: class_id cx cy w h
                            f.write(f"{class_obj.class_id} {ann.cx} {ann.cy} {ann.width} {ann.height}\n")

        # Generate data.yaml
        data_yaml_path = dataset_path / "data.yaml"

        data_yaml = {
            "path": str(dataset_path.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(class_names),
            "names": class_names
        }

        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        return str(data_yaml_path)

    @staticmethod
    def get_dataset_stats(db: Session, project_id: int) -> Dict[str, any]:
        """
        Get dataset statistics.

        Args:
            db: Database session
            project_id: Project ID

        Returns:
            Statistics dictionary
        """
        stats = {}

        for split_type in [SplitType.TRAIN, SplitType.VAL, SplitType.TEST, SplitType.UNLABELED]:
            count = db.query(func.count(Image.id)).filter(
                Image.project_id == project_id,
                Image.split_type == split_type
            ).scalar() or 0

            stats[split_type.value] = count

        # Get annotation count per class
        class_counts = {}
        classes = db.query(Class).filter(Class.project_id == project_id).all()

        for cls in classes:
            count = db.query(func.count(Annotation.id)).filter(
                Annotation.class_id == cls.id
            ).scalar() or 0

            class_counts[cls.class_name] = count

        stats["class_distribution"] = class_counts

        return stats
