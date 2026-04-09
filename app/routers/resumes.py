from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
import logging

from app.models.schemas import ParseResponse, ParsedResumeData
from app.services.file_service import ResumeParserService

router = APIRouter(prefix="/resumes", tags=["resumes"])

logger = logging.getLogger(__name__)


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
