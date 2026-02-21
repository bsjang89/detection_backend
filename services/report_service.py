from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from jinja2 import Template

from models.database import (
    Model, TrainingSession, TrainingMetricsLog,
    InferenceResult, Image, Class, Project
)
from config.settings import settings


class ReportService:
    """
    Service for generating HTML reports for training and inference results.
    """

    REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detection Report - {{ model_name }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f0f; color: #e5e5e5; padding: 24px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { font-size: 28px; margin-bottom: 8px; }
        h2 { font-size: 20px; margin: 24px 0 12px; border-bottom: 1px solid #333; padding-bottom: 8px; }
        h3 { font-size: 16px; margin: 16px 0 8px; }
        .meta { color: #888; font-size: 13px; margin-bottom: 24px; }
        .card {
            background: #1e1e1e; border: 1px solid #333;
            border-radius: 10px; padding: 20px; margin-bottom: 16px;
        }
        .stats-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px; margin: 12px 0;
        }
        .stat-box {
            text-align: center; padding: 16px; background: #2a2a2a;
            border-radius: 8px;
        }
        .stat-value { font-size: 24px; font-weight: 700; color: #fff; }
        .stat-label { font-size: 11px; color: #888; margin-top: 4px; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #333; }
        th { color: #888; font-weight: 600; }
        .badge {
            display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 11px; font-weight: 700;
        }
        .badge-success { background: #16a34a; color: #fff; }
        .badge-info { background: #2563eb; color: #fff; }
        .result-grid {
            display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 12px; margin: 12px 0;
        }
        .result-card {
            background: #2a2a2a; border-radius: 8px; overflow: hidden;
        }
        .result-card img {
            width: 100%; display: block;
        }
        .result-card .info { padding: 8px 12px; font-size: 12px; }
        .config-grid {
            display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 8px;
        }
        .config-item { padding: 8px; background: #2a2a2a; border-radius: 6px; }
        .config-key { color: #888; font-size: 11px; }
        .config-val { font-weight: 600; }
        .footer { margin-top: 40px; text-align: center; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Detection Report</h1>
        <p class="meta">
            Model: <strong>{{ model_name }}</strong> |
            Generated: {{ generated_at }} |
            Project: {{ project_name }}
        </p>

        <!-- Model Summary -->
        <div class="card">
            <h2 style="margin-top:0; border:none;">Model Summary</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-value">{{ model_type }}</div>
                    <div class="stat-label">Model Type</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{{ task_type }}</div>
                    <div class="stat-label">Task Type</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{{ "%.1f"|format(map50 * 100) if map50 else "N/A" }}%</div>
                    <div class="stat-label">mAP50</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{{ "%.1f"|format(map50_95 * 100) if map50_95 else "N/A" }}%</div>
                    <div class="stat-label">mAP50-95</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{{ "%.1f"|format(precision * 100) if precision else "N/A" }}%</div>
                    <div class="stat-label">Precision</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{{ "%.1f"|format(recall * 100) if recall else "N/A" }}%</div>
                    <div class="stat-label">Recall</div>
                </div>
            </div>
        </div>

        <!-- Training Configuration -->
        {% if training_config %}
        <div class="card">
            <h2 style="margin-top:0; border:none;">Training Configuration</h2>
            <div class="config-grid">
                {% for key, value in training_config.items() %}
                <div class="config-item">
                    <div class="config-key">{{ key }}</div>
                    <div class="config-val">{{ value }}</div>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}

        <!-- Training Metrics -->
        {% if metrics_data %}
        <div class="card">
            <h2 style="margin-top:0; border:none;">Training Progress</h2>
            <table>
                <thead>
                    <tr>
                        <th>Epoch</th>
                        <th>Train Box Loss</th>
                        <th>Train Cls Loss</th>
                        <th>Val Box Loss</th>
                        <th>Val Cls Loss</th>
                        <th>mAP50</th>
                        <th>mAP50-95</th>
                        <th>Precision</th>
                        <th>Recall</th>
                    </tr>
                </thead>
                <tbody>
                    {% for m in metrics_data %}
                    <tr>
                        <td>{{ m.epoch }}</td>
                        <td>{{ "%.4f"|format(m.train_box_loss) if m.train_box_loss else "-" }}</td>
                        <td>{{ "%.4f"|format(m.train_cls_loss) if m.train_cls_loss else "-" }}</td>
                        <td>{{ "%.4f"|format(m.val_box_loss) if m.val_box_loss else "-" }}</td>
                        <td>{{ "%.4f"|format(m.val_cls_loss) if m.val_cls_loss else "-" }}</td>
                        <td>{{ "%.4f"|format(m.map50) if m.map50 else "-" }}</td>
                        <td>{{ "%.4f"|format(m.map50_95) if m.map50_95 else "-" }}</td>
                        <td>{{ "%.4f"|format(m.precision) if m.precision else "-" }}</td>
                        <td>{{ "%.4f"|format(m.recall) if m.recall else "-" }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        <!-- Inference Results -->
        {% if inference_results %}
        <div class="card">
            <h2 style="margin-top:0; border:none;">Sample Detection Results</h2>
            <div class="result-grid">
                {% for result in inference_results %}
                <div class="result-card">
                    {% if result.result_image_path %}
                    <img src="file:///{{ result.result_image_path }}" alt="{{ result.filename }}">
                    {% endif %}
                    <div class="info">
                        <strong>{{ result.filename }}</strong><br>
                        Detections: {{ result.detection_count }} |
                        Time: {{ "%.1f"|format(result.inference_time_ms) }}ms
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}

        <!-- Class Distribution -->
        {% if class_distribution %}
        <div class="card">
            <h2 style="margin-top:0; border:none;">Class Distribution</h2>
            <table>
                <thead>
                    <tr>
                        <th>Class ID</th>
                        <th>Class Name</th>
                        <th>Annotation Count</th>
                    </tr>
                </thead>
                <tbody>
                    {% for cls in class_distribution %}
                    <tr>
                        <td>{{ cls.class_id }}</td>
                        <td>{{ cls.class_name }}</td>
                        <td>{{ cls.count }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        <div class="footer">
            Generated by Object Detection Training System | {{ generated_at }}
        </div>
    </div>
</body>
</html>
"""

    @classmethod
    def generate_report(
        cls,
        db: Session,
        model_id: int,
        max_inference_results: int = 20
    ) -> str:
        """
        Generate an HTML report for a model.

        Args:
            db: Database session
            model_id: Model ID
            max_inference_results: Max number of inference results to show

        Returns:
            HTML report content as string
        """
        # Get model
        model = db.query(Model).filter(Model.id == model_id).first()
        if not model:
            raise ValueError(f"Model {model_id} not found")

        # Get training session
        training_session = db.query(TrainingSession).filter(
            TrainingSession.id == model.training_session_id
        ).first()

        # Get project
        project = None
        if training_session:
            project = db.query(Project).filter(Project.id == training_session.project_id).first()

        # Get metrics
        metrics_data = []
        if training_session:
            metrics = db.query(TrainingMetricsLog).filter(
                TrainingMetricsLog.training_session_id == training_session.id
            ).order_by(TrainingMetricsLog.epoch).all()

            # Sample every Nth epoch for large datasets
            step = max(1, len(metrics) // 50)
            for i in range(0, len(metrics), step):
                m = metrics[i]
                metrics_data.append({
                    "epoch": m.epoch,
                    "train_box_loss": m.train_box_loss,
                    "train_cls_loss": m.train_cls_loss,
                    "val_box_loss": m.val_box_loss,
                    "val_cls_loss": m.val_cls_loss,
                    "map50": m.map50,
                    "map50_95": m.map50_95,
                    "precision": m.precision,
                    "recall": m.recall,
                })

        # Get inference results
        inference_results = []
        results = db.query(InferenceResult).filter(
            InferenceResult.model_id == model_id
        ).limit(max_inference_results).all()

        for r in results:
            image = db.query(Image).filter(Image.id == r.image_id).first()
            inference_results.append({
                "filename": image.filename if image else "unknown",
                "detection_count": len(r.detections) if r.detections else 0,
                "inference_time_ms": r.inference_time_ms or 0,
                "result_image_path": r.result_image_path,
            })

        # Get class distribution
        class_distribution = []
        if project:
            from sqlalchemy import func
            from models.database import Annotation
            classes = db.query(Class).filter(Class.project_id == project.id).order_by(Class.class_id).all()
            for cls in classes:
                count = db.query(func.count(Annotation.id)).filter(
                    Annotation.class_id == cls.id
                ).scalar() or 0
                class_distribution.append({
                    "class_id": cls.class_id,
                    "class_name": cls.class_name,
                    "count": count,
                })

        # Training config
        training_config = training_session.config if training_session else None

        # Render template
        template = Template(cls.REPORT_TEMPLATE)
        html = template.render(
            model_name=model.name,
            model_type=model.model_type,
            task_type=model.task_type,
            map50=model.map50,
            map50_95=model.map50_95,
            precision=model.precision,
            recall=model.recall,
            project_name=project.name if project else "Unknown",
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            training_config=training_config,
            metrics_data=metrics_data,
            inference_results=inference_results,
            class_distribution=class_distribution,
        )

        return html

    @classmethod
    def save_report(cls, db: Session, model_id: int) -> str:
        """
        Generate and save HTML report to file.

        Args:
            db: Database session
            model_id: Model ID

        Returns:
            Path to saved report file
        """
        html = cls.generate_report(db, model_id)

        report_dir = Path(settings.REPORTS_DIR)
        report_dir.mkdir(parents=True, exist_ok=True)

        filename = f"report_model_{model_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = report_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        return str(filepath)
