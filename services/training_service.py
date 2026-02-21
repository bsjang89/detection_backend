from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, Callable, Union, Set
from pathlib import Path
from datetime import datetime
import threading
import platform
import shutil

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
    Uses configured training GPU when available, otherwise falls back to CPU.
    """

    # Track active training sessions
    _active_sessions: Dict[int, threading.Thread] = {}
    _stop_requests: Set[int] = set()

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

        if training_session_id in cls._stop_requests:
            cls._stop_requests.discard(training_session_id)

        # Get project
        project = db.query(Project).filter(Project.id == session.project_id).first()
        if not project:
            raise ValueError(f"Project {session.project_id} not found")

        # Resolve training device (GPU if available, otherwise CPU fallback)
        training_device: Union[int, str] = GPUManager.get_training_device()

        # Prepare dataset directory
        dataset_dir = Path(settings.UPLOAD_DIR) / str(project.id) / "dataset"

        # Always regenerate dataset to reflect latest labels and avoid stale cache artifacts
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
            args=(training_session_id, str(data_yaml_path), training_device, on_epoch_callback),
            daemon=True
        )

        cls._active_sessions[training_session_id] = training_thread
        training_thread.start()

        return {
            "status": "started",
            "training_session_id": training_session_id,
            "device": training_device
        }

    @classmethod
    def _run_training(
        cls,
        training_session_id: int,
        data_yaml_path: str,
        training_device: Union[int, str],
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
                device=training_device
            )

            # Load pretrained weights
            model.load_pretrained()

            # Training configuration
            config = session.config.copy()
            config['device'] = training_device
            # On Windows, multi-worker dataloaders can stall before first epoch in this setup.
            # Force single-process loading for stable progress reporting.
            if platform.system().lower().startswith("win") and config.get("workers", 0) != 0:
                config["workers"] = 0
            # Background-thread training should not use interactive tqdm console output.
            config["verbose"] = False

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
                    progress = ((epoch + 1) / total_epochs) * 100
                    session.progress = progress
                    db.commit()

                    # Call external callback (for WebSocket)
                    if on_epoch_callback:
                        on_epoch_callback(training_session_id, epoch, metrics)

                    # Cooperative stop request: stop at epoch boundary.
                    if training_session_id in cls._stop_requests:
                        session.status = TrainingStatus.STOPPED
                        session.completed_at = datetime.utcnow()
                        db.commit()
                        return True

                except Exception as e:
                    print(f"Error saving metrics: {e}")
                return False

            # Train
            results = model.train(
                data_yaml=data_yaml_path,
                config=config,
                callback=epoch_callback
            )

            stop_requested = training_session_id in cls._stop_requests

            # Save results
            session.completed_at = datetime.utcnow()
            session.best_weights_path = results.get('best_weights')
            session.last_weights_path = results.get('last_weights')

            # Save final metrics
            final_metrics = results.get('final_metrics', {})
            session.final_map50 = final_metrics.get('map50')
            session.final_map50_95 = final_metrics.get('map50_95')
            session.final_precision = final_metrics.get('precision')
            session.final_recall = final_metrics.get('recall')

            if stop_requested:
                session.status = TrainingStatus.STOPPED
            else:
                session.status = TrainingStatus.COMPLETED
                session.progress = 100.0

                # Create Model record for completed sessions
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
            cls._stop_requests.discard(training_session_id)

    @classmethod
    def is_session_active(cls, training_session_id: int) -> bool:
        """
        Check whether a training session thread is still alive.
        """
        thread = cls._active_sessions.get(training_session_id)
        if thread is None:
            return False
        if not thread.is_alive():
            del cls._active_sessions[training_session_id]
            return False
        return True

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

        if session.status != TrainingStatus.RUNNING and not cls.is_session_active(training_session_id):
            raise ValueError("Training session is not running")

        cls._stop_requests.add(training_session_id)

        # Update status
        session.status = TrainingStatus.STOPPED
        session.completed_at = datetime.utcnow()
        db.commit()

        return {"status": "stopping", "message": "Stop requested"}

    @classmethod
    def delete_training_session(cls, db: Session, training_session_id: int) -> Dict[str, Any]:
        """
        Delete a training session and related artifacts.
        """
        session = db.query(TrainingSession).filter(
            TrainingSession.id == training_session_id
        ).first()

        if not session:
            raise ValueError(f"Training session {training_session_id} not found")

        if cls.is_session_active(training_session_id):
            raise RuntimeError("Training session is still active. Stop training and wait until it fully stops.")

        cls._cleanup_training_artifacts(session)
        db.delete(session)
        db.commit()

        return {
            "status": "deleted",
            "training_session_id": training_session_id
        }

    @classmethod
    def _cleanup_training_artifacts(cls, session: TrainingSession):
        """
        Remove persisted weight files and run directories for a session.
        """
        base_dir = Path.cwd().resolve()
        file_paths = set()

        for p in [session.best_weights_path, session.last_weights_path]:
            if p:
                file_paths.add(p)

        for model in session.models:
            if model.weights_path:
                file_paths.add(model.weights_path)

        run_dirs = set()
        resolved_files = set()

        for raw_path in file_paths:
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            try:
                resolved = candidate.resolve()
            except Exception:
                continue

            # Only allow deletion under project working directory.
            if resolved != base_dir and base_dir not in resolved.parents:
                continue

            resolved_files.add(resolved)
            if resolved.parent.name == "weights":
                run_dirs.add(resolved.parent.parent)

        for file_path in resolved_files:
            try:
                if file_path.exists() and file_path.is_file():
                    file_path.unlink()
            except Exception:
                pass

        for run_dir in run_dirs:
            try:
                if run_dir.exists() and run_dir.is_dir():
                    shutil.rmtree(run_dir, ignore_errors=True)
            except Exception:
                pass

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
            "is_active": cls.is_session_active(training_session_id)
        }
