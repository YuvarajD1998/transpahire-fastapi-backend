import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.database import db_manager
from app.limiter import limiter
from app.middleware.errorhandler import ErrorHandlerMiddleware, setup_exception_handlers
from app.routers import health, resumes, embeddings, embeddings_local, jd, skills, match as match_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    db_manager.init_db()
    logging.info("FastAPI TranspaHire Profile Service started (Read-Only Mode)")
    yield
    # Shutdown
    await db_manager.close()
    logging.info("FastAPI TranspaHire Profile Service stopped")

def create_app() -> FastAPI:
    app = FastAPI(
        title="TranspaHire Profile Service",
        description="Profile Management and Resume Processing Microservice (Read-Only)",
        version="1.0.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # Add rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Add middleware
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Setup exception handlers
    setup_exception_handlers(app)

    # Include routers
    app.include_router(health.router, prefix=settings.API_V1_PREFIX)
    app.include_router(resumes.router, prefix=settings.API_V1_PREFIX)
    app.include_router(embeddings.router, prefix=settings.API_V1_PREFIX)
    app.include_router(embeddings_local.router, prefix=settings.API_V1_PREFIX)
    app.include_router(jd.router, prefix=settings.API_V1_PREFIX)
    app.include_router(skills.router, prefix=settings.API_V1_PREFIX)
    app.include_router(match_router.router, prefix=settings.API_V1_PREFIX)

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0", 
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )
