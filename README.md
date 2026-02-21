# Object Detection Backend (FastAPI + YOLO)

Backend API server for project management, labeling data, training orchestration, and inference.
It supports YOLO BBox/OBB training and real-time training updates via WebSocket.

## Architecture

```mermaid
flowchart LR
    UI[React Frontend] -->|REST /api/v1| API[FastAPI]
    UI -->|WS /api/v1/ws/training/:id| API
    API --> DB[(PostgreSQL)]
    API --> FS[(Local Storage: uploads/runs/models)]
    API --> YOLO[Ultralytics YOLO]
    YOLO --> GPU[CUDA GPU or CPU Fallback]
```

## Core Features

- Project, class, image, and annotation CRUD
- Training session create/start/stop/delete
- Real-time epoch metrics broadcast (WebSocket)
- Training metrics and model artifacts persistence
- Session delete cleanup for weights and run folders

## Tech Stack

- Python 3.11
- FastAPI + Uvicorn
- SQLAlchemy + PostgreSQL
- Ultralytics + PyTorch

## Quick Start

### 1) Requirements

- Python 3.11+
- PostgreSQL 14+
- Optional: NVIDIA GPU and CUDA driver

### 2) Install

```powershell
cd detection_backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 3) Configure `.env`

```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/detection_db
TRAINING_GPU_ID=0
INFERENCE_GPU_ID=1
UPLOAD_DIR=uploads
MODELS_DIR=trained_models
NETWORK_DIR=network
RESULTS_DIR=results
REPORTS_DIR=reports
BACKEND_CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]
```

### 4) Run

```powershell
cd detection_backend
.\.venv\Scripts\activate
python main.py
```

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

## Training Lifecycle

```mermaid
sequenceDiagram
    participant UI as Frontend
    participant API as FastAPI
    participant T as Training Thread
    participant DB as PostgreSQL

    UI->>API: POST /training/sessions
    UI->>API: POST /training/sessions/{id}/start
    API->>T: spawn thread
    loop each epoch
        T->>DB: insert metrics
        T-->>UI: WS epoch_update
    end
    UI->>API: POST /training/sessions/{id}/stop (optional)
    UI->>API: DELETE /training/sessions/{id}
    API->>DB: delete session + related rows
```

## Important Endpoints

| Area | Endpoint |
|---|---|
| Projects | `GET/POST /api/v1/projects/` |
| Images | `/api/v1/images/...` |
| Annotations | `/api/v1/annotations/...` |
| Training sessions | `POST /api/v1/training/sessions` |
| Start/Stop training | `POST /api/v1/training/sessions/{id}/start`, `POST /api/v1/training/sessions/{id}/stop` |
| Delete session | `DELETE /api/v1/training/sessions/{id}` |
| Metrics | `GET /api/v1/training/sessions/{id}/metrics` |
| WebSocket | `WS /api/v1/ws/training/{id}` |

## Project Structure

```text
detection_backend/
  api/v1/            # REST + WS routes
  services/          # training and dataset logic
  models/database/   # SQLAlchemy models
  models/ml/         # YOLO wrappers (bbox/obb)
  schemas/           # Pydantic schemas
  config/            # settings and database
  network/           # pretrained .pt cache/download folder
  uploads/ runs/ trained_models/ results/
  main.py
```

## Troubleshooting

- GPU start errors
  - verify CUDA torch build
  - verify `TRAINING_GPU_ID` matches available GPU index
- Training appears slow before first epoch
  - initial cache/data loader warmup can take time
  - on Windows, `workers=0` is recommended
- Session delete returns `409`
  - training thread is still active; retry after it becomes inactive

## Git Notes

- Pretrained weights are downloaded into `network/`.
- Default `.gitignore` keeps `network/*.pt` untracked and preserves `network/.gitkeep`.
