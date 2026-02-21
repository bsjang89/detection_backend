from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
import threading

from models.database import (
    Project, TrainingSession, TrainingStatus,
    TrainingMetricsLog, Model
)
from models.ml import ModelRegistry
from utils.gpu_manager import GPUManager
from .dataset_service import DatasetService
from config.settings import settings


class TrainingService:
    """
    Service for managing model training.
    Uses GPU 0 exclusively for training.
    """

    # Track active training sessions
    _active_sessions: Dict[int, threading.Thread] = {}

    @classmethod
    def start_training(
        cls,
        db: Session,
        training_session_id: int,
        on_epoch_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Start a training session.

        Args:
            db: Database session
            training_session_id: TrainingSession ID
            on_epoch_callback: Callback for epoch updates (for WebSocket broadcast)

        Returns:
            Status dictionary
        """
        # Get training session
        session = db.query(TrainingSession).filter(
            TrainingSession.id == training_session_id
        ).first()

        if not session:
            raise ValueError(f"Training session {training_session_id} not found")

        if session.status == TrainingStatus.RUNNING:
            raise ValueError("Training session is already running")

        # Get project
        project = db.query(Project).filter(Project.id == session.project_id).first()
        if not project:
            raise ValueError(f"Project {session.project_id} not found")

        # Validate training GPU
        try:
            GPUManager.validate_training_gpu()
        except RuntimeError as e:
            session.status = TrainingStatus.FAILED
            session.error_message = str(e)
            db.commit()
            raise

        # Check if dataset is split
        dataset_dir = Path(settings.UPLOAD_DIR) / str(project.id) / "dataset"

        # Generate YOLO dataset if not exists
        data_yaml_path = dataset_dir / "data.yaml"
        if not data_yaml_path.exists():
            try:
                data_yaml_path = DatasetService.generate_yolo_dataset(
                    db=db,
                    project_id=project.id,
                    output_dir=str(dataset_dir)
                )
            except Exception as e:
                session.status = TrainingStatus.FAILED
                session.error_message = f"Failed to generate dataset: {str(e)}"
                db.commit()
                raise

        # Update session status
        session.status = TrainingStatus.RUNNING
        session.started_at = datetime.utcnow()
        session.error_message = None
        db.commit()

        # Start training in background thread
        training_thread = threading.Thread(
            target=cls._run_training,
            args=(training_session_id, str(data_yaml_path), on_epoch_callback),
            daemon=True
        )

        cls._active_sessions[training_session_id] = training_thread
        training_thread.start()

        return {
            "status": "started",
            "training_session_id": training_session_id,
            "gpu": GPUManager.get_training_gpu()
        }

    @classmethod
    def _run_training(
        cls,
        training_session_id: int,
        data_yaml_path: str,
        on_epoch_callback: Optional[Callable] = None
    ):
        """
        Internal method to run training (runs in background thread).

        Args:
            training_session_id: TrainingSession ID
            data_yaml_path: Path to data.yaml
            on_epoch_callback: Callback for epoch updates
        """
        # Create new DB session for this thread
        from config.database import SessionLocal
        db = SessionLocal()

        try:
            session = db.query(TrainingSession).filter(
                TrainingSession.id == training_session_id
            ).first()

            project = db.query(Project).filter(Project.id == session.project_id).first()

            # Create model using registry
            model = ModelRegistry.create_model(
                task_type=project.task_type.value,
                model_type=session.model_type,
                device=GPUManager.get_training_gpu()
            )

            # Load pretrained weights
            model.load_pretrained()

            # Training configuration
            config = session.config.copy()
            config['device'] = GPUManager.get_training_gpu()

            # Define epoch callback
            def epoch_callback(epoch: int, metrics: Dict[str, Any]):
                # Save metrics to database
                try:
                    metrics_log = TrainingMetricsLog(
                        training_session_id=training_session_id,
                        epoch=epoch,
                        train_box_loss=metrics.get('train_box_loss'),
                        train_cls_loss=metrics.get('train_cls_loss'),
                        train_dfl_loss=metrics.get('train_dfl_loss'),
                        val_box_loss=metrics.get('val_box_loss'),
                        val_cls_loss=metrics.get('val_cls_loss'),
                        val_dfl_loss=metrics.get('val_dfl_loss'),
                        map50=metrics.get('map50'),
                        map50_95=metrics.get('map50_95'),
                        precision=metrics.get('precision'),
                        recall=metrics.get('recall'),
                        learning_rate=metrics.get('learning_rate')
                    )

                    db.add(metrics_log)
                    db.commit()

                    # Update progress
                    total_epochs = config.get('epochs', 100)
                    progress = (epoch / total_epochs) * 100
                    session.progress = progress
                    db.commit()

                    # Call external callback (for WebSocket)
                    if on_epoch_callback:
                        on_epoch_callback(training_session_id, epoch, metrics)

                except Exception as e:
                    print(f"Error saving metrics: {e}")

            # Train
            results = model.train(
                data_yaml=data_yaml_path,
                config=config,
                callback=epoch_callback
            )

            # Save results
            session.status = TrainingStatus.COMPLETED
            session.completed_at = datetime.utcnow()
            session.progress = 100.0
            session.best_weights_path = results['best_weights']
            session.last_weights_path = results['last_weights']

            # Save final metrics
            final_metrics = results['final_metrics']
            session.final_map50 = final_metrics.get('map50')
            session.final_map50_95 = final_metrics.get('map50_95')
            session.final_precision = final_metrics.get('precision')
            session.final_recall = final_metrics.get('recall')

            # Create Model record
            model_record = Model(
                training_session_id=training_session_id,
                name=f"{session.name or 'model'}_{training_session_id}_best",
                weights_path=results['best_weights'],
                is_best=True,
                map50=final_metrics.get('map50'),
                map50_95=final_metrics.get('map50_95'),
                precision=final_metrics.get('precision'),
                recall=final_metrics.get('recall'),
                model_type=session.model_type,
                task_type=project.task_type.value,
                img_size=config.get('imgsz', 640),
                num_classes=project.classes.__len__() if hasattr(project, 'classes') else 0
            )

            db.add(model_record)
            db.commit()

        except Exception as e:
            session.status = TrainingStatus.FAILED
            session.error_message = str(e)
            session.completed_at = datetime.utcnow()
            db.commit()
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()

        finally:
            db.close()
            # Remove from active sessions
            if training_session_id in cls._active_sessions:
                del cls._active_sessions[training_session_id]

    @classmethod
    def stop_training(cls, db: Session, training_session_id: int) -> Dict[str, str]:
        """
        Stop a running training session.

        Args:
            db: Database session
            training_session_id: TrainingSession ID

        Returns:
            Status dictionary
        """
        session = db.query(TrainingSession).filter(
            TrainingSession.id == training_session_id
        ).first()

        if not session:
            raise ValueError(f"Training session {training_session_id} not found")

        if session.status != TrainingStatus.RUNNING:
            raise ValueError("Training session is not running")

        # Update status
        session.status = TrainingStatus.STOPPED
        session.completed_at = datetime.utcnow()
        db.commit()

        # Note: Ultralytics training doesn't support graceful stopping
        # The thread will continue but the status is marked as stopped

        return {"status": "stopped", "message": "Training marked as stopped"}

    @classmethod
    def get_training_status(cls, db: Session, training_session_id: int) -> Dict[str, Any]:
        """
        Get training session status.

        Args:
            db: Database session
            training_session_id: TrainingSession ID

        Returns:
            Status dictionary
        """
        session = db.query(TrainingSession).filter(
            TrainingSession.id == training_session_id
        ).first()

        if not session:
            raise ValueError(f"Training session {training_session_id} not found")

        return {
            "id": session.id,
            "status": session.status.value,
            "progress": session.progress,
            "started_at": session.started_at,
            "completed_at": session.completed_at,
            "error_message": session.error_message,
            "is_active": training_session_id in cls._active_sessions
        }
