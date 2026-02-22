from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from config.settings import settings
from config.database import init_db
from api.v1 import projects, images, annotations, training, websocket, inference

# Initialize database tables
init_db()

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_origin_regex=settings.BACKEND_CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.MODELS_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.NETWORK_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.REPORTS_DIR).mkdir(parents=True, exist_ok=True)

# Include routers (BEFORE static file mounts)
app.include_router(
    projects.router,
    prefix=f"{settings.API_V1_STR}/projects",
    tags=["projects"]
)
app.include_router(
    images.router,
    prefix=f"{settings.API_V1_STR}/images",
    tags=["images"]
)
app.include_router(
    annotations.router,
    prefix=f"{settings.API_V1_STR}/annotations",
    tags=["annotations"]
)
app.include_router(
    training.router,
    prefix=f"{settings.API_V1_STR}/training",
    tags=["training"]
)
app.include_router(
    websocket.router,
    prefix=f"{settings.API_V1_STR}/ws",
    tags=["websocket"]
)
app.include_router(
    inference.router,
    prefix=f"{settings.API_V1_STR}/inference",
    tags=["inference"]
)

# Mount static files (AFTER routers)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.mount("/results", StaticFiles(directory=settings.RESULTS_DIR), name="results")


@app.get("/")
def root():
    return {
        "message": "Object Detection Training System API",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
