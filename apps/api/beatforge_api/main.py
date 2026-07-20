from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select
from starlette.exceptions import HTTPException

from .config import get_settings
from .database import SessionLocal, init_db
from .errors import BeatForgeError
from .media import cleanup_stale_decoded
from .models import AnalysisJobModel, VocalAlignmentJobModel
from .routes import router
from .serialization import dumps, json_safe

logger = logging.getLogger("beatforge_api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    cleanup_stale_decoded(get_settings().analyses_dir / "decoded")
    with SessionLocal() as session:
        interrupted = session.scalars(
            select(AnalysisJobModel).where(AnalysisJobModel.status.in_(["queued", "processing"]))
        ).all()
        for job in interrupted:
            job.status = "failed"
            job.stage = "failed"
            job.error_json = dumps(
                {
                    "code": "ANALYSIS_INTERRUPTED",
                    "message": (
                        "The local API stopped before this analysis completed; retry is available."
                    ),
                }
            )
            if job.track and job.track.project:
                job.track.project.status = "failed"
        interrupted_vocal_jobs = session.scalars(
            select(VocalAlignmentJobModel).where(
                VocalAlignmentJobModel.status.in_(["queued", "processing"])
            )
        ).all()
        for job in interrupted_vocal_jobs:
            job.status = "failed"
            job.error_json = dumps(
                {
                    "code": "VOCAL_ALIGNMENT_INTERRUPTED",
                    "message": (
                        "The local API stopped before lyric alignment completed; "
                        "retry is available."
                    ),
                }
            )
        session.commit()
    yield


app = FastAPI(
    title="BeatForge Studio API",
    summary="Sample-accurate local beat detection and editing service",
    version="0.7.1",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Disposition"],
)
app.include_router(router)


@app.exception_handler(BeatForgeError)
async def beatforge_error_handler(_request: Request, exc: BeatForgeError) -> JSONResponse:
    headers = None
    if exc.status_code == 416 and isinstance(exc.details, dict) and "size" in exc.details:
        headers = {"Content-Range": f"bytes */{exc.details['size']}"}
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": json_safe(exc.details),
            }
        },
        headers=headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "The request contains invalid fields",
                "details": json_safe(exc.errors()),
            }
        },
    )


@app.exception_handler(HTTPException)
async def http_error_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        content = exc.detail
    else:
        content = {
            "error": {
                "code": "HTTP_ERROR",
                "message": str(exc.detail),
                "details": None,
            }
        }
    return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error on %s", request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected local API error occurred",
                "details": None,
            }
        },
    )
