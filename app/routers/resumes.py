import logging
import time

from fastapi import APIRouter, Depends, Request, UploadFile, File, Form
from pydantic import BaseModel
from typing import Any, Optional

from app.models.schemas import ParseResponse, ParsedResumeData
from app.services.file_service import ResumeParserService
from app.services.resume_critique_service import ResumeCritiqueService
from app.limiter import limiter

router = APIRouter(prefix="/resumes", tags=["resumes"])

logger = logging.getLogger(__name__)


class CritiqueRequest(BaseModel):
    resume_id: int
    parsed_data: dict[str, Any]


class CritiqueResponse(BaseModel):
    success: bool
    resume_id: int
    overall_score: int
    sections: dict
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]
    error: str | None = None


@router.post("/parse-resume", response_model=ParseResponse)
async def parse_resume(
    file: UploadFile = File(...),
    resume_id: int = Form(...),
    enhance_images: bool = Form(True)
):
    """
    Parse resume from multipart/form-data upload.
    Accepts file + metadata fields.
    """
    try:
        file_content = await file.read()
        parser = ResumeParserService()
        parsed_data = await parser.parse(
            file_content=file_content,
            filename=file.filename,
            enhance_images=enhance_images
        )

        return ParseResponse(
            success=True,
            resume_id=resume_id,
            parsed_data=parsed_data,
            error=None,
            confidence_score=parsed_data.confidence_score if parsed_data else 0.0
        )

    except Exception as e:
        logger.error(f"Parse failed: {str(e)}")
        return ParseResponse(
            success=False,
            resume_id=resume_id,
            parsed_data=None,
            error=str(e),
            confidence_score=0.0
        )


@router.post("/critique", response_model=CritiqueResponse)
@limiter.limit("10/minute")
async def generate_resume_critique(
    req: CritiqueRequest,
    request: Request,
):
    """
    Generate an AI critique for a parsed resume.
    Called by NestJS after resume parsing completes.
    Does NOT write to DB — NestJS writes the result via BullMQ processor.
    Returns the critique data for NestJS to persist.
    """
    t0 = time.monotonic()
    service = ResumeCritiqueService()
    try:
        critique = await service.generate_critique(req.resume_id, req.parsed_data)
        latency_ms = round((time.monotonic() - t0) * 1000)
        logger.info("critique resume_id=%s score=%d latency_ms=%d", req.resume_id, critique["overall_score"], latency_ms)
        return CritiqueResponse(
            success=True,
            resume_id=req.resume_id,
            **critique,
        )
    except Exception as e:
        latency_ms = round((time.monotonic() - t0) * 1000)
        logger.error("critique resume_id=%s failed latency_ms=%d: %s", req.resume_id, latency_ms, e)
        return CritiqueResponse(
            success=False,
            resume_id=req.resume_id,
            overall_score=0,
            sections={},
            strengths=[],
            weaknesses=[],
            suggestions=[],
            error=str(e),
        )
