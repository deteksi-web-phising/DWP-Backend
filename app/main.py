from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.features.detection.router import router as detection_router
from app.features.benchmark.router import router as benchmark_router
from app.features.eda.router import router as eda_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="REST API for Machine Learning-based Phishing Website Detection and Analysis.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS Middleware
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Register Feature-First Routers
app.include_router(detection_router, prefix=settings.API_V1_STR)
app.include_router(benchmark_router, prefix=settings.API_V1_STR)
app.include_router(eda_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Health Check"])
def root():
    """Root endpoint verifying API operational status."""
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
    }


@app.get("/health", tags=["Health Check"])
def health_check():
    """Health check endpoint for container probes."""
    return {"status": "healthy"}
