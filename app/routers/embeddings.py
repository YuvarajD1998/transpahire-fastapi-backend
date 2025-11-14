from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict

from app.database import get_db
from app.dependencies import get_current_user
from app.services.multi_vector_embedding_service import MultiVectorEmbeddingService

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post("/candidates/{candidate_id}/generate")
async def generate_candidate_embeddings(
    candidate_id: int,
    background_tasks: BackgroundTasks,
    regenerate: bool = False,
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """
    Generate multi-vector embeddings for a candidate.
    Creates 5 embeddings: summary, skills, experience, education, full
    """
    embedding_service = MultiVectorEmbeddingService()
    
    # Run in background
    background_tasks.add_task(
        embedding_service.generate_candidate_embeddings,
        db, candidate_id, regenerate
    )
    
    return {
        "message": "Embedding generation started",
        "candidate_id": candidate_id,
        "regenerate": regenerate
    }


@router.get("/candidates/{candidate_id}/status")
async def get_candidate_embedding_status(
    candidate_id: int,
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """Check embedding status for a candidate."""
    embedding_service = MultiVectorEmbeddingService()
    
    embeddings = await embedding_service._get_candidate_embeddings(db, candidate_id)
    
    return {
        "candidate_id": candidate_id,
        "total_embeddings": len(embeddings),
        "embedding_types": [e.type for e in embeddings],
        "has_all_embeddings": len(embeddings) >= 5,
        "model_name": embeddings[0].model_name if embeddings else None,
        "dimension": embeddings[0].dimension if embeddings else None
    }


@router.post("/jobs/{job_id}/generate")
async def generate_job_embeddings(
    job_id: int,
    job_data: Dict,
    background_tasks: BackgroundTasks,
    regenerate: bool = False,
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """
    Generate multi-vector embeddings for a job.
    Creates 3 embeddings: jd_summary, required_skills, responsibilities
    """
    embedding_service = MultiVectorEmbeddingService()
    
    background_tasks.add_task(
        embedding_service.generate_job_embeddings,
        db, job_id, job_data, regenerate
    )
    
    return {
        "message": "Job embedding generation started",
        "job_id": job_id,
        "regenerate": regenerate
    }


@router.get("/jobs/{job_id}/status")
async def get_job_embedding_status(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """Check embedding status for a job."""
    embedding_service = MultiVectorEmbeddingService()
    
    embeddings = await embedding_service._get_job_embeddings(db, job_id)
    
    return {
        "job_id": job_id,
        "total_embeddings": len(embeddings),
        "embedding_types": [e.type for e in embeddings],
        "has_all_embeddings": len(embeddings) >= 3,
        "model_name": embeddings[0].model_name if embeddings else None,
        "dimension": embeddings[0].dimension if embeddings else None
    }
