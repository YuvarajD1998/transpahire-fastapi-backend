from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class SimilarCandidatesRequest(BaseModel):
    job_id: int
    embedding_type: Literal['summary', 'skills', 'experience', 'full'] = 'full'
    top_k: int = Field(default=20, ge=1, le=100)
    min_similarity: float = Field(default=0.3, ge=0.0, le=1.0)


class CandidateSimilarityResult(BaseModel):
    candidate_id: int
    similarity: float


class SimilarCandidatesResponse(BaseModel):
    job_id: int
    results: List[CandidateSimilarityResult]
    total: int
    embedding_type: str


class SimilarJobsRequest(BaseModel):
    candidate_id: int
    embedding_type: Literal['summary', 'skills', 'experience', 'full'] = 'full'
    top_k: int = Field(default=20, ge=1, le=100)
    min_similarity: float = Field(default=0.3, ge=0.0, le=1.0)


class JobSimilarityResult(BaseModel):
    job_id: int
    similarity: float


class SimilarJobsResponse(BaseModel):
    candidate_id: int
    results: List[JobSimilarityResult]
    total: int
    embedding_type: str


class MatchFeaturesRequest(BaseModel):
    candidate_id: int
    job_id: int


class MatchFeaturesResponse(BaseModel):
    candidate_id: int
    job_id: int
    semantic_similarity: Optional[float] = None
    skills_similarity: Optional[float] = None
    experience_similarity: Optional[float] = None


class MatchInsightsRequest(BaseModel):
    candidate_id: int
    job_id: int
    features: dict
    candidate_summary: str
    job_summary: str


class MatchInsightsResponse(BaseModel):
    strengths: List[str]
    gaps: List[str]
    suggestion: str
    recommendation: Literal['STRONG_MATCH', 'GOOD_MATCH', 'POTENTIAL', 'WEAK']
