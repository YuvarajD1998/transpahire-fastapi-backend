from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
import logging

from app.models.schemas import JdParseResponse
from app.services.file_service import JdParserService
from app.services.jd_gemini_service import JdGeminiService

router = APIRouter(prefix="/jd", tags=["jd"])

logger = logging.getLogger(__name__)


@router.post("/parse-file", response_model=JdParseResponse)
async def parse_jd_file(
    file: UploadFile = File(...),
    jd_id: int = Form(...),
):
    try:
        file_content = await file.read()
        parser = JdParserService()
        parsed_data = await parser.parse_jd(file_content, file.filename)
        return JdParseResponse(
            success=True,
            jd_id=jd_id,
            parsed_data=parsed_data,
            error=None,
            confidence_score=parsed_data.confidence_score,
        )
    except Exception as e:
        logger.error(f"JD file parse failed: {str(e)}")
        return JdParseResponse(
            success=False,
            jd_id=jd_id,
            parsed_data=None,
            error=str(e),
            confidence_score=0.0,
        )


@router.post("/parse-text", response_model=JdParseResponse)
async def parse_jd_text(payload: dict):
    jd_id: Optional[int] = payload.get("jd_id")
    try:
        text = payload.get("text", "")
        if not text:
            return JdParseResponse(
                success=False,
                jd_id=jd_id,
                error="Missing 'text' field in request body",
                confidence_score=0.0,
            )
        service = JdGeminiService()
        parsed_data = await service.parse_jd_text(text)
        return JdParseResponse(
            success=True,
            jd_id=jd_id,
            parsed_data=parsed_data,
            error=None,
            confidence_score=parsed_data.confidence_score,
        )
    except Exception as e:
        logger.error(f"JD text parse failed: {str(e)}")
        return JdParseResponse(
            success=False,
            jd_id=jd_id,
            parsed_data=None,
            error=str(e),
            confidence_score=0.0,
        )
