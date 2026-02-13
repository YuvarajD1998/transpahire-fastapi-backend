from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

from app.services.embedding_service_local import embedding_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


class EmbeddingRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to embed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Skill: Python\nContext: Used for backend development with FastAPI"
            }
        }


class BatchEmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to embed")


class EmbeddingResponse(BaseModel):
    success: bool
    embedding: List[float]
    dimension: int
    error: Optional[str] = None


class BatchEmbeddingResponse(BaseModel):
    success: bool
    embeddings: List[List[float]]
    dimension: int
    count: int
    error: Optional[str] = None


@router.post("/generate-embedding", response_model=EmbeddingResponse)
async def generate_embedding(request: EmbeddingRequest):
    """Generate embedding with appropriate prefix."""
    try:
        embedding = embedding_service.generate_embedding(
            text=request.text,
            prefix_type="query", 
            normalize=True
        )
        
        return EmbeddingResponse(
            success=True,
            embedding=embedding,
            dimension=len(embedding),
            error=None
        )
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/generate-batch-embeddings", response_model=BatchEmbeddingResponse)
async def generate_batch_embeddings(request: BatchEmbeddingRequest):
    """
    Generate embeddings for multiple texts in batch.
    More efficient for bulk operations.
    """
    try:
        embeddings = embedding_service.generate_batch_embeddings(request.texts)
        
        return BatchEmbeddingResponse(
            success=True,
            embeddings=embeddings,
            dimension=len(embeddings[0]) if embeddings else 0,
            count=len(embeddings),
            error=None
        )
        
    except Exception as e:
        logger.error(f"Batch embedding generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate batch embeddings: {str(e)}"
        )

