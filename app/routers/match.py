import logging
import time

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.limiter import limiter
from app.services.match_service import MatchService
from app.services.match_insights_service import MatchInsightsService
from app.models.match_schemas import (
    SimilarCandidatesRequest, SimilarCandidatesResponse,
    SimilarJobsRequest, SimilarJobsResponse,
    MatchFeaturesRequest, MatchFeaturesResponse,
    MatchInsightsRequest, MatchInsightsResponse,
)

router = APIRouter(prefix="/match", tags=["matching"])
logger = logging.getLogger(__name__)


@router.post("/similar-candidates", response_model=SimilarCandidatesResponse)
async def find_similar_candidates(
    req: SimilarCandidatesRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    t0 = time.monotonic()
    service = MatchService()
    results = await service.find_similar_candidates(
        db=db,
        job_id=req.job_id,
        embedding_type=req.embedding_type,
        top_k=req.top_k,
        min_similarity=req.min_similarity,
    )
    latency_ms = round((time.monotonic() - t0) * 1000)
    logger.info(
        "similar-candidates job_id=%s embedding_type=%s results=%d latency_ms=%d",
        req.job_id, req.embedding_type, len(results), latency_ms,
    )
    return SimilarCandidatesResponse(
        job_id=req.job_id,
        results=results,
        total=len(results),
        embedding_type=req.embedding_type,
    )


@router.post("/similar-jobs", response_model=SimilarJobsResponse)
async def find_similar_jobs(
    req: SimilarJobsRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    t0 = time.monotonic()
    service = MatchService()
    results = await service.find_similar_jobs(
        db=db,
        candidate_id=req.candidate_id,
        embedding_type=req.embedding_type,
        top_k=req.top_k,
        min_similarity=req.min_similarity,
    )
    latency_ms = round((time.monotonic() - t0) * 1000)
    logger.info(
        "similar-jobs candidate_id=%s embedding_type=%s results=%d latency_ms=%d",
        req.candidate_id, req.embedding_type, len(results), latency_ms,
    )
    return SimilarJobsResponse(
        candidate_id=req.candidate_id,
        results=results,
        total=len(results),
        embedding_type=req.embedding_type,
    )


@router.post("/features", response_model=MatchFeaturesResponse)
async def compute_match_features(
    req: MatchFeaturesRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    t0 = time.monotonic()
    service = MatchService()
    result = await service.compute_match_features(
        db=db,
        candidate_id=req.candidate_id,
        job_id=req.job_id,
    )
    latency_ms = round((time.monotonic() - t0) * 1000)

    # Warn if all similarities are null — means embeddings are missing for this pair
    all_null = (
        result.semantic_similarity is None
        and result.skills_similarity is None
        and result.experience_similarity is None
    )
    if all_null:
        logger.warning(
            "features: no embeddings found for candidate_id=%s job_id=%s — all similarities null",
            req.candidate_id, req.job_id,
        )
    else:
        logger.info(
            "features candidate_id=%s job_id=%s semantic=%.4f skills=%.4f exp=%.4f latency_ms=%d",
            req.candidate_id, req.job_id,
            result.semantic_similarity or 0,
            result.skills_similarity or 0,
            result.experience_similarity or 0,
            latency_ms,
        )
    return result


@router.post("/insights", response_model=MatchInsightsResponse)
@limiter.limit("20/minute")
async def generate_match_insights(
    req: MatchInsightsRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    t0 = time.monotonic()
    service = MatchInsightsService()
    result = await service.generate_insights(
        candidate_id=req.candidate_id,
        job_id=req.job_id,
        features=req.features,
        candidate_summary=req.candidate_summary,
        job_summary=req.job_summary,
    )
    latency_ms = round((time.monotonic() - t0) * 1000)
    logger.info(
        "insights candidate_id=%s job_id=%s recommendation=%s latency_ms=%d",
        req.candidate_id, req.job_id, result.recommendation, latency_ms,
    )
    return result
