# TranspaHire Phase 2 — FastAPI Implementation Plan

> **How to use this file**: Feed this into a fresh Claude Code thread with the instruction: "Implement Phase 2 of the FastAPI AI microservice for TranspaHire. Follow this plan exactly, verify each existing file against the codebase before modifying, and do not skip steps."

---

## 0. Current State Summary (verified from codebase)

### What exists
- **`routers/embeddings.py`**: endpoints to generate candidate embeddings (5 types) and job embeddings (3 types) — trigger only, background task
- **`routers/embeddings_local.py`**: local embedding generation (`/generate-embedding`, `/generate-batch-embeddings`)
- **`routers/jd.py`**: JD parsing from file and text
- **`routers/resumes.py`**: resume parsing
- **`routers/skills.py`**: skill categorization + taxonomy mapping
- **`services/multi_vector_embedding_service.py`**: generates and stores candidate/job embeddings using Gemini → pgvector
- **`services/gemini_service_embedding.py`**: Gemini embedding API wrapper (text-embedding-004, 768 dims)
- **`services/representation_service.py`**: builds text representations for candidates and jobs
- **`models/embedding_models.py`**: SQLAlchemy `CandidateEmbedding`, `JobEmbedding` ORM models

### What's missing (Phase 2 builds these)
1. **`routers/match.py`** — vector similarity search + scoring + RAG insights
2. **`services/match_service.py`** — cosine similarity search via pgvector
3. **`services/match_features_service.py`** — compute structured MatchFeatures from DB
4. **`services/match_insights_service.py`** — RAG-based Gemini insights generation
5. **`routers/resumes.py` extension** — add `/critique` endpoint
6. **`services/resume_critique_service.py`** — LLM-based resume critique

### Architecture constraint (must respect)
- **NestJS calls FastAPI via HTTP** — FastAPI never calls NestJS
- **FastAPI never calls the Prisma-managed DB directly for writes** — it only reads profiles/skills/experiences to build embeddings and reads embeddings for similarity search. Critique results are written to `resume_critiques` table (exception: FastAPI writes embeddings and critiques)
- **Gemini first, local sentence-transformers fallback** — same pattern already in `gemini_service_embedding.py`
- **No OpenAI, Pinecone, or external vector DB** — pgvector only
- **Raw SQL via SQLAlchemy** for vector `<=>` cosine distance — SQLAlchemy ORM cannot express this

---

## 1. New Router: `app/routers/match.py`

This router exposes all matching and similarity search endpoints.

### Register in `app/main.py`

```python
from app.routers import match
app.include_router(match.router, prefix="/api/v1")
```

### Pydantic schemas (add to `app/models/schemas.py` or new `match_schemas.py`)

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class SimilarCandidatesRequest(BaseModel):
    job_id: int
    embedding_type: Literal['summary', 'skills', 'experience', 'full'] = 'full'
    top_k: int = Field(default=20, ge=1, le=100)
    min_similarity: float = Field(default=0.3, ge=0.0, le=1.0)

class CandidateSimilarityResult(BaseModel):
    candidate_id: int
    similarity: float  # cosine similarity 0-1

class SimilarCandidatesResponse(BaseModel):
    job_id: int
    results: List[CandidateSimilarityResult]
    total: int
    embedding_type: str

# ---

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

# ---

class MatchFeaturesRequest(BaseModel):
    candidate_id: int
    job_id: int

class MatchFeaturesResponse(BaseModel):
    candidate_id: int
    job_id: int
    semantic_similarity: Optional[float]  # cosine similarity between full embeddings
    skills_similarity: Optional[float]    # cosine similarity between skill embeddings
    experience_similarity: Optional[float]

# ---

class MatchInsightsRequest(BaseModel):
    candidate_id: int
    job_id: int
    features: dict  # pre-computed numeric features from NestJS
    candidate_summary: str
    job_summary: str

class MatchInsightsResponse(BaseModel):
    strengths: List[str]
    gaps: List[str]
    suggestion: str
    recommendation: Literal['STRONG_MATCH', 'GOOD_MATCH', 'POTENTIAL', 'WEAK']
```

### `app/routers/match.py` — full implementation

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.dependencies import get_current_user
from app.services.match_service import MatchService
from app.services.match_insights_service import MatchInsightsService
from app.models.match_schemas import (
    SimilarCandidatesRequest, SimilarCandidatesResponse,
    SimilarJobsRequest, SimilarJobsResponse,
    MatchFeaturesRequest, MatchFeaturesResponse,
    MatchInsightsRequest, MatchInsightsResponse,
)

router = APIRouter(prefix="/match", tags=["matching"])


@router.post("/similar-candidates", response_model=SimilarCandidatesResponse)
async def find_similar_candidates(
    request: SimilarCandidatesRequest,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Given a job_id, find candidates whose embeddings are most similar to the job's embedding.
    Uses pgvector cosine distance (<=>).
    """
    service = MatchService()
    results = await service.find_similar_candidates(
        db=db,
        job_id=request.job_id,
        embedding_type=request.embedding_type,
        top_k=request.top_k,
        min_similarity=request.min_similarity,
    )
    return SimilarCandidatesResponse(
        job_id=request.job_id,
        results=results,
        total=len(results),
        embedding_type=request.embedding_type,
    )


@router.post("/similar-jobs", response_model=SimilarJobsResponse)
async def find_similar_jobs(
    request: SimilarJobsRequest,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Given a candidate_id, find jobs whose embeddings are most similar to the candidate's embedding.
    """
    service = MatchService()
    results = await service.find_similar_jobs(
        db=db,
        candidate_id=request.candidate_id,
        embedding_type=request.embedding_type,
        top_k=request.top_k,
        min_similarity=request.min_similarity,
    )
    return SimilarJobsResponse(
        candidate_id=request.candidate_id,
        results=results,
        total=len(results),
        embedding_type=request.embedding_type,
    )


@router.post("/features", response_model=MatchFeaturesResponse)
async def compute_match_features(
    request: MatchFeaturesRequest,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Compute semantic similarity features for a candidate-job pair.
    Returns cosine similarities across embedding types.
    """
    service = MatchService()
    return await service.compute_match_features(
        db=db,
        candidate_id=request.candidate_id,
        job_id=request.job_id,
    )


@router.post("/insights", response_model=MatchInsightsResponse)
async def generate_match_insights(
    request: MatchInsightsRequest,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Generate RAG-based human-readable match insights using Gemini.
    Takes pre-computed features from NestJS + candidate/job summary text.
    """
    service = MatchInsightsService()
    return await service.generate_insights(
        candidate_id=request.candidate_id,
        job_id=request.job_id,
        features=request.features,
        candidate_summary=request.candidate_summary,
        job_summary=request.job_summary,
    )
```

---

## 2. New Service: `app/services/match_service.py`

Core vector similarity search. Uses raw SQL via SQLAlchemy for pgvector `<=>` operator.

```python
import logging
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.models.match_schemas import CandidateSimilarityResult, JobSimilarityResult, MatchFeaturesResponse

logger = logging.getLogger(__name__)


class MatchService:
    """
    Performs vector similarity search using pgvector cosine distance.
    Raw SQL is used because SQLAlchemy ORM cannot express the <=> operator.
    """

    async def find_similar_candidates(
        self,
        db: AsyncSession,
        job_id: int,
        embedding_type: str,
        top_k: int,
        min_similarity: float,
    ) -> List[CandidateSimilarityResult]:
        """
        Find candidates most similar to a job's embedding.
        
        Strategy:
        1. Fetch job embedding vector (jd_summary or required_skills type)
        2. Run cosine similarity search against candidate_embeddings
        3. Filter by minimum similarity threshold
        4. Return ranked list of candidate IDs with similarity scores
        """
        # Map job-side embedding to a candidate-side embedding type for comparison
        # Full candidate profile compared against jd_summary or required_skills
        candidate_type = embedding_type  # caller controls which candidate embedding type

        # Step 1: Get the job embedding vector
        # For candidate matching, prefer using job's 'required_skills' embedding against candidate's 'skills'
        # or job's 'jd_summary' against candidate's 'full'
        job_embedding_type = self._map_to_job_embedding_type(embedding_type)
        
        result = await db.execute(
            text("""
                SELECT vector FROM job_embeddings
                WHERE job_id = :job_id AND type = :embedding_type
                ORDER BY version DESC
                LIMIT 1
            """),
            {"job_id": job_id, "embedding_type": job_embedding_type},
        )
        job_row = result.fetchone()
        if not job_row or job_row.vector is None:
            logger.warning(f"No {job_embedding_type} embedding found for job {job_id}")
            return []

        job_vector_str = job_row.vector  # pgvector returns as string representation

        # Step 2: Cosine similarity search against candidate_embeddings
        # 1 - cosine_distance = cosine_similarity
        rows = await db.execute(
            text("""
                SELECT
                    ce.candidate_id,
                    1 - (ce.vector <=> :job_vector::vector) AS similarity
                FROM candidate_embeddings ce
                WHERE
                    ce.type = :candidate_type
                    AND ce.vector IS NOT NULL
                    AND 1 - (ce.vector <=> :job_vector::vector) >= :min_similarity
                ORDER BY ce.vector <=> :job_vector::vector ASC
                LIMIT :top_k
            """),
            {
                "job_vector": job_vector_str,
                "candidate_type": candidate_type,
                "min_similarity": min_similarity,
                "top_k": top_k,
            },
        )

        results = []
        for row in rows.fetchall():
            results.append(CandidateSimilarityResult(
                candidate_id=row.candidate_id,
                similarity=round(float(row.similarity), 4),
            ))

        logger.info(f"Found {len(results)} similar candidates for job {job_id} (type={candidate_type})")
        return results

    async def find_similar_jobs(
        self,
        db: AsyncSession,
        candidate_id: int,
        embedding_type: str,
        top_k: int,
        min_similarity: float,
    ) -> List[JobSimilarityResult]:
        """
        Find jobs most similar to a candidate's embedding.
        """
        # Get candidate embedding vector
        result = await db.execute(
            text("""
                SELECT vector FROM candidate_embeddings
                WHERE candidate_id = :candidate_id AND type = :embedding_type
                ORDER BY version DESC
                LIMIT 1
            """),
            {"candidate_id": candidate_id, "embedding_type": embedding_type},
        )
        candidate_row = result.fetchone()
        if not candidate_row or candidate_row.vector is None:
            logger.warning(f"No {embedding_type} embedding found for candidate {candidate_id}")
            return []

        candidate_vector_str = candidate_row.vector

        # Map candidate embedding type to job embedding type
        job_embedding_type = self._map_to_job_embedding_type(embedding_type)

        rows = await db.execute(
            text("""
                SELECT
                    je.job_id,
                    1 - (je.vector <=> :candidate_vector::vector) AS similarity
                FROM job_embeddings je
                WHERE
                    je.type = :job_type
                    AND je.vector IS NOT NULL
                    AND 1 - (je.vector <=> :candidate_vector::vector) >= :min_similarity
                ORDER BY je.vector <=> :candidate_vector::vector ASC
                LIMIT :top_k
            """),
            {
                "candidate_vector": candidate_vector_str,
                "job_type": job_embedding_type,
                "min_similarity": min_similarity,
                "top_k": top_k,
            },
        )

        results = []
        for row in rows.fetchall():
            results.append(JobSimilarityResult(
                job_id=row.job_id,
                similarity=round(float(row.similarity), 4),
            ))

        logger.info(f"Found {len(results)} similar jobs for candidate {candidate_id}")
        return results

    async def compute_match_features(
        self,
        db: AsyncSession,
        candidate_id: int,
        job_id: int,
    ) -> MatchFeaturesResponse:
        """
        Compute cosine similarities across multiple embedding type pairs.
        Returns semantic similarity features for a candidate-job pair.
        """
        pairs = [
            ('full', 'jd_summary'),         # overall match
            ('skills', 'required_skills'),   # skill-specific match
            ('experience', 'responsibilities'),  # role-specific match
        ]

        similarities = {}
        for candidate_type, job_type in pairs:
            sim = await self._compute_pair_similarity(db, candidate_id, job_id, candidate_type, job_type)
            similarities[f"{candidate_type}_vs_{job_type}"] = sim

        return MatchFeaturesResponse(
            candidate_id=candidate_id,
            job_id=job_id,
            semantic_similarity=similarities.get('full_vs_jd_summary'),
            skills_similarity=similarities.get('skills_vs_required_skills'),
            experience_similarity=similarities.get('experience_vs_responsibilities'),
        )

    async def _compute_pair_similarity(
        self,
        db: AsyncSession,
        candidate_id: int,
        job_id: int,
        candidate_type: str,
        job_type: str,
    ) -> Optional[float]:
        """Compute cosine similarity between a specific candidate-job embedding pair."""
        result = await db.execute(
            text("""
                SELECT
                    1 - (ce.vector <=> je.vector) AS similarity
                FROM candidate_embeddings ce
                CROSS JOIN job_embeddings je
                WHERE
                    ce.candidate_id = :candidate_id
                    AND ce.type = :candidate_type
                    AND je.job_id = :job_id
                    AND je.type = :job_type
                    AND ce.vector IS NOT NULL
                    AND je.vector IS NOT NULL
                ORDER BY ce.version DESC, je.version DESC
                LIMIT 1
            """),
            {
                "candidate_id": candidate_id,
                "candidate_type": candidate_type,
                "job_id": job_id,
                "job_type": job_type,
            },
        )
        row = result.fetchone()
        if not row:
            return None
        return round(float(row.similarity), 4)

    def _map_to_job_embedding_type(self, candidate_embedding_type: str) -> str:
        """Map candidate embedding type to corresponding job embedding type."""
        mapping = {
            'full': 'jd_summary',
            'summary': 'jd_summary',
            'skills': 'required_skills',
            'experience': 'responsibilities',
            'education': 'jd_summary',  # fallback
        }
        return mapping.get(candidate_embedding_type, 'jd_summary')
```

---

## 3. New Service: `app/services/match_insights_service.py`

RAG-based match explanation using Gemini. Takes pre-computed features from NestJS + narrative summaries.

```python
import logging
from typing import List
from app.services.gemini_service import GeminiService  # reuse existing cascade
from app.models.match_schemas import MatchInsightsResponse

logger = logging.getLogger(__name__)

INSIGHTS_PROMPT_TEMPLATE = """
You are an expert recruiter AI analyzing a candidate-job match.

JOB SUMMARY:
{job_summary}

CANDIDATE SUMMARY:
{candidate_summary}

PRE-COMPUTED MATCH FEATURES:
- Required skill coverage: {required_coverage:.0%}
- Critical skill coverage: {critical_coverage:.0%}
- Experience fit score: {exp_curve:.0%}
- Semantic similarity: {semantic_similarity:.0%}
- Salary compatibility: {salary_score:.0%}
- Location/work mode fit: {location_score:.0%}

Based on this analysis, provide a concise match assessment:

1. STRENGTHS (2-3 bullet points, what aligns well)
2. GAPS (1-2 bullet points, what's missing or misaligned)
3. IMPROVEMENT SUGGESTION (one actionable thing that would most improve the match, e.g., "If the candidate gains X certification, their score would improve significantly")
4. RECOMMENDATION: Choose exactly one of: STRONG_MATCH, GOOD_MATCH, POTENTIAL, WEAK

Guidelines for recommendation:
- STRONG_MATCH: 75%+ skill coverage, good experience fit, minimal gaps
- GOOD_MATCH: 55-74% skill coverage, acceptable experience, 1-2 gaps
- POTENTIAL: 35-54% skill coverage, transferable skills exist, worth reviewing
- WEAK: <35% coverage, significant gaps

Respond in valid JSON format:
{{
  "strengths": ["point1", "point2"],
  "gaps": ["gap1"],
  "suggestion": "specific actionable suggestion",
  "recommendation": "STRONG_MATCH"
}}
"""


class MatchInsightsService:
    def __init__(self):
        self.gemini = GeminiService()

    async def generate_insights(
        self,
        candidate_id: int,
        job_id: int,
        features: dict,
        candidate_summary: str,
        job_summary: str,
    ) -> MatchInsightsResponse:
        prompt = INSIGHTS_PROMPT_TEMPLATE.format(
            job_summary=job_summary[:1000],
            candidate_summary=candidate_summary[:1000],
            required_coverage=features.get('required_coverage', 0),
            critical_coverage=features.get('critical_coverage', 0),
            exp_curve=features.get('exp_curve', 0),
            semantic_similarity=features.get('semantic_similarity', 0),
            salary_score=features.get('salary_score', 0.6),
            location_score=features.get('location_score', 0.5),
        )

        try:
            # Reuse the existing Gemini model cascade from GeminiService
            # Parse resume method does text → JSON — reuse the same pattern
            result_text = await self.gemini.generate_text(prompt)
            parsed = self._parse_insights_response(result_text)
            return MatchInsightsResponse(**parsed)
        except Exception as e:
            logger.error(f"Match insights generation failed for candidate={candidate_id} job={job_id}: {e}")
            # Return safe fallback
            return self._fallback_insights(features)

    def _parse_insights_response(self, text: str) -> dict:
        import json, re
        # Strip markdown code blocks if present
        text = re.sub(r'```(?:json)?', '', text).strip()
        try:
            data = json.loads(text)
            # Validate required fields
            return {
                "strengths": data.get("strengths", [])[:3],
                "gaps": data.get("gaps", [])[:3],
                "suggestion": data.get("suggestion", ""),
                "recommendation": data.get("recommendation", "POTENTIAL"),
            }
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse insights JSON: {text[:200]}")

    def _fallback_insights(self, features: dict) -> MatchInsightsResponse:
        required_cov = features.get('required_coverage', 0)
        if required_cov >= 0.75:
            rec = 'STRONG_MATCH'
        elif required_cov >= 0.55:
            rec = 'GOOD_MATCH'
        elif required_cov >= 0.35:
            rec = 'POTENTIAL'
        else:
            rec = 'WEAK'

        return MatchInsightsResponse(
            strengths=["Profile shows relevant experience for this role."],
            gaps=["Some required skills may be missing."],
            suggestion="Review the candidate's profile for transferable skills.",
            recommendation=rec,
        )
```

**Note on `GeminiService.generate_text`**: The existing `GeminiService` in `gemini_service.py` has `parse_resume_text()` which uses the model cascade internally. You need to extract a generic `generate_text(prompt: str) -> str` method from it, or add it as a new public method. The model cascade pattern is already there — just expose it for non-resume prompts. Add this to `gemini_service.py`:

```python
async def generate_text(self, prompt: str, max_tokens: int = 2000) -> str:
    """Generic text generation using model cascade. Returns raw text response."""
    # Use the same cascade pattern as parse_resume_text but without resume-specific schema
    for model_name in [self.model_1, self.model_2, self.model_3]:
        try:
            response = await self._call_model(model_name, prompt, max_tokens)
            if response:
                return response
        except Exception as e:
            logger.warning(f"Model {model_name} failed: {e}, trying next...")
    raise RuntimeError("All Gemini models failed for text generation")
```

---

## 4. Resume Critique Service: `app/services/resume_critique_service.py`

```python
import logging
import json
import re
from app.services.gemini_service import GeminiService

logger = logging.getLogger(__name__)

CRITIQUE_PROMPT = """
You are an expert resume reviewer with 15 years of recruiting experience across tech, finance, and consulting.

Analyze the following parsed resume data and provide actionable critique:

RESUME DATA:
{resume_text}

Provide a detailed critique in the following JSON format:
{{
  "overall_score": <integer 0-100>,
  "sections": {{
    "summary": {{"score": <0-100>, "feedback": "<specific feedback>"}},
    "experience": {{"score": <0-100>, "feedback": "<specific feedback>"}},
    "skills": {{"score": <0-100>, "feedback": "<specific feedback>"}},
    "education": {{"score": <0-100>, "feedback": "<specific feedback>"}},
    "formatting": {{"score": <0-100>, "feedback": "<specific feedback>"}}
  }},
  "strengths": ["strength1", "strength2", "strength3"],
  "weaknesses": ["weakness1", "weakness2"],
  "suggestions": [
    "Specific actionable suggestion 1",
    "Specific actionable suggestion 2",
    "Specific actionable suggestion 3"
  ]
}}

Scoring rubric:
- 85-100: Exceptional, ready to submit
- 70-84: Strong, minor improvements
- 55-69: Average, needs work
- 40-54: Below average, significant gaps
- 0-39: Poor, major restructuring needed

Be specific, actionable, and constructive. Reference actual content from the resume.
"""


class ResumeCritiqueService:
    def __init__(self):
        self.gemini = GeminiService()

    async def generate_critique(self, resume_id: int, parsed_data: dict) -> dict:
        """
        Generate a critique for a parsed resume.
        Returns a dict matching the ResumeCritique schema.
        """
        resume_text = self._format_resume_for_critique(parsed_data)

        prompt = CRITIQUE_PROMPT.format(resume_text=resume_text[:4000])

        try:
            result_text = await self.gemini.generate_text(prompt, max_tokens=3000)
            critique = self._parse_critique_response(result_text)
            logger.info(f"Generated critique for resume {resume_id}, score={critique['overall_score']}")
            return critique
        except Exception as e:
            logger.error(f"Critique generation failed for resume {resume_id}: {e}")
            return self._fallback_critique(parsed_data)

    def _format_resume_for_critique(self, parsed_data: dict) -> str:
        """Convert parsed resume data to readable text for critique."""
        parts = []

        personal = parsed_data.get('personal_info', {})
        if personal.get('name'):
            parts.append(f"Name: {personal['name']}")
        if personal.get('location'):
            parts.append(f"Location: {personal['location']}")

        summary = parsed_data.get('summary', {})
        if isinstance(summary, dict) and summary.get('raw'):
            parts.append(f"\nSUMMARY:\n{summary['raw'][:500]}")
        elif isinstance(summary, str):
            parts.append(f"\nSUMMARY:\n{summary[:500]}")

        experience = parsed_data.get('experience', [])
        if experience:
            parts.append("\nEXPERIENCE:")
            for exp in experience[:5]:
                parts.append(f"- {exp.get('position', '')} at {exp.get('company', '')}")
                if exp.get('description'):
                    parts.append(f"  {str(exp['description'])[:200]}")

        skills = parsed_data.get('skills', {})
        tech_skills = skills.get('technical_skills', []) if isinstance(skills, dict) else []
        if tech_skills:
            skill_names = [s.get('name', '') if isinstance(s, dict) else str(s) for s in tech_skills[:20]]
            parts.append(f"\nTECHNICAL SKILLS: {', '.join(filter(None, skill_names))}")

        education = parsed_data.get('education', [])
        if education:
            parts.append("\nEDUCATION:")
            for edu in education[:3]:
                parts.append(f"- {edu.get('degree', '')} from {edu.get('institution', '')}")

        certs = parsed_data.get('certifications', [])
        if certs:
            cert_names = [c.get('name', '') if isinstance(c, dict) else str(c) for c in certs[:5]]
            parts.append(f"\nCERTIFICATIONS: {', '.join(filter(None, cert_names))}")

        return '\n'.join(parts)

    def _parse_critique_response(self, text: str) -> dict:
        text = re.sub(r'```(?:json)?', '', text).strip()
        try:
            data = json.loads(text)
            return {
                "overall_score": max(0, min(100, int(data.get("overall_score", 50)))),
                "sections": data.get("sections", {}),
                "strengths": data.get("strengths", [])[:5],
                "weaknesses": data.get("weaknesses", [])[:5],
                "suggestions": data.get("suggestions", [])[:5],
            }
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse critique JSON: {e}")

    def _fallback_critique(self, parsed_data: dict) -> dict:
        has_experience = bool(parsed_data.get('experience'))
        has_skills = bool(parsed_data.get('skills'))
        has_education = bool(parsed_data.get('education'))
        score = 40 + (20 if has_experience else 0) + (20 if has_skills else 0) + (10 if has_education else 0)

        return {
            "overall_score": score,
            "sections": {
                "experience": {"score": 70 if has_experience else 30, "feedback": "Experience section detected." if has_experience else "No experience data found."},
                "skills": {"score": 70 if has_skills else 30, "feedback": "Skills listed." if has_skills else "Skills section missing or sparse."},
                "education": {"score": 70 if has_education else 30, "feedback": "Education present." if has_education else "Education section missing."},
            },
            "strengths": ["Resume has been successfully parsed with key sections."],
            "weaknesses": ["Detailed analysis unavailable at this time."],
            "suggestions": ["Ensure all sections are complete and specific achievements are listed."],
        }
```

---

## 5. Resume Critique Endpoint (extend `app/routers/resumes.py`)

Add the following endpoint to the existing resumes router:

```python
from app.services.resume_critique_service import ResumeCritiqueService
from pydantic import BaseModel
from typing import Any

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


@router.post("/critique", response_model=CritiqueResponse)
async def generate_resume_critique(
    request: CritiqueRequest,
    user=Depends(get_current_user),
):
    """
    Generate an AI critique for a parsed resume.
    Called by NestJS after resume parsing completes.
    Does NOT write to DB — NestJS writes the result via BullMQ processor.
    Returns the critique data for NestJS to persist.
    """
    service = ResumeCritiqueService()
    try:
        critique = await service.generate_critique(request.resume_id, request.parsed_data)
        return CritiqueResponse(
            success=True,
            resume_id=request.resume_id,
            **critique,
        )
    except Exception as e:
        return CritiqueResponse(
            success=False,
            resume_id=request.resume_id,
            overall_score=0,
            sections={},
            strengths=[],
            weaknesses=[],
            suggestions=[],
            error=str(e),
        )
```

---

## 6. SQLAlchemy Model: `match_feature_cache` (read-only)

FastAPI doesn't write to `match_feature_cache` — that's NestJS's responsibility. No SQLAlchemy model needed for FastAPI. Only NestJS writes and reads from this table.

---

## 7. pgvector Setup Verification

Before deploying, verify the `pgvector` extension is installed and HNSW indexes exist:

```sql
-- Verify extension
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Verify indexes
SELECT indexname, indexdef FROM pg_indexes
WHERE tablename IN ('candidate_embeddings', 'job_embeddings')
AND indexdef LIKE '%hnsw%';
```

If HNSW indexes are not present, they must be created by NestJS migration (see NestJS plan). The FastAPI service benefits from these indexes but does not create them.

---

## 8. Integration of `generate_text` into `GeminiService`

The `gemini_service.py` currently has `parse_resume_text()` which wraps the cascade. For insights and critique, we need a generic `generate_text()`. Inspect the existing `_call_model()` or `parse_resume_text()` private method and extract a reusable pattern.

**Key rule**: Do NOT duplicate the model cascade. Extract `generate_text(prompt, max_tokens)` as a shared method, then have `parse_resume_text` call it internally.

The cascade order to preserve: `GEMINI_PARSE_MODEL_1` → `GEMINI_PARSE_MODEL_2` → `GEMINI_PARSE_MODEL_3` (read from `config.py`).

---

## 9. `app/models/match_schemas.py` (new file)

Create this as a dedicated file to keep `schemas.py` clean:

```python
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
```

---

## 10. `app/main.py` Changes

Register the new `match` router. Find the section where other routers are included and add:

```python
from app.routers import match as match_router
app.include_router(match_router.router, prefix="/api/v1")
```

---

## 11. Implementation Order

### Phase A — Similarity Search (unblocked, start here)

1. Create `app/models/match_schemas.py`
2. Add `generate_text()` to existing `GeminiService` in `gemini_service.py`
3. Create `app/services/match_service.py` (vector similarity search only, no RAG)
4. Create `app/routers/match.py` with `/similar-candidates`, `/similar-jobs`, `/features` endpoints (skip `/insights` for now)
5. Register match router in `main.py`
6. Test with: verify candidate and job embeddings exist, then call `/match/similar-candidates`

### Phase B — Insights + Critique

7. Create `app/services/match_insights_service.py`
8. Add `/insights` endpoint to match router
9. Create `app/services/resume_critique_service.py`
10. Add `/critique` endpoint to resumes router

### Phase C — Polish

11. Add rate limiting to match endpoints (expensive Gemini calls)
12. Add request logging (latency, candidate/job IDs, similarity scores returned)
13. Add input validation (verify embeddings exist before attempting search, return 404 if not)

---

## 12. Testing Guide

### Test similarity search

```bash
# First, ensure embeddings exist
curl -X GET http://localhost:8000/api/v1/embeddings/candidates/1/status \
  -H "Authorization: Bearer <token>"

curl -X GET http://localhost:8000/api/v1/embeddings/jobs/1/status \
  -H "Authorization: Bearer <token>"

# Then test similarity search
curl -X POST http://localhost:8000/api/v1/match/similar-candidates \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"job_id": 1, "embedding_type": "full", "top_k": 10, "min_similarity": 0.2}'
```

Expected: list of candidate IDs with similarity scores between 0 and 1.

### Test match features

```bash
curl -X POST http://localhost:8000/api/v1/match/features \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"candidate_id": 1, "job_id": 1}'
```

Expected: `{ candidate_id, job_id, semantic_similarity, skills_similarity, experience_similarity }`.

### Test insights

```bash
curl -X POST http://localhost:8000/api/v1/match/insights \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_id": 1,
    "job_id": 1,
    "features": {"required_coverage": 0.7, "critical_coverage": 0.9, "exp_curve": 0.8, "semantic_similarity": 0.75, "salary_score": 0.6, "location_score": 1.0},
    "candidate_summary": "Senior React developer with 5 years experience...",
    "job_summary": "Looking for a senior frontend engineer with React expertise..."
  }'
```

Expected: `{ strengths, gaps, suggestion, recommendation }`.

### Test critique

```bash
curl -X POST http://localhost:8000/api/v1/resumes/critique \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"resume_id": 1, "parsed_data": { ... }}'
```

---

## 13. Error Handling Conventions

- If embeddings don't exist for a requested entity: return `{"results": [], "total": 0}` — do NOT raise 404
- If Gemini fails for insights/critique: return a fallback response (not an error) — the NestJS caller treats a fallback as acceptable
- Log all failures with `candidate_id`/`job_id` for debugging
- Similarity searches should never crash — missing vectors should return empty, not 500

---

## 14. Performance Notes

- Cosine similarity search with HNSW index on 100k candidate embeddings: < 50ms
- Without HNSW index: full table scan, potentially 5-30s — **indexes are mandatory before Phase B launch**
- Gemini insights: 2-4s per call — only call on explicit `explain=true` requests, never in hot path
- Batch embedding generation: use the existing background task pattern (`BackgroundTasks`) — don't block HTTP response
