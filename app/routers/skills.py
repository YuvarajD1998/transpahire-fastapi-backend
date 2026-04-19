from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.services.skill_mapping_service import SkillMappingService

router = APIRouter(prefix="/skills", tags=["skills"])
_skill_mapping_service = SkillMappingService()


# ── Request models ──────────────────────────────────────────────────────────

class NonTaxonomySkillInput(BaseModel):
    id: int
    skillName: str
    normalizedName: str
    skillType: Optional[str] = "TECHNICAL"
    frequency: Optional[int] = 1
    contextSnippets: Optional[list[str]] = []


class TaxonomySkillEntry(BaseModel):
    id: int
    name: str
    normalizedName: str
    category: Optional[str] = None
    subcategory: Optional[str] = None


class TaxonomyCategory(BaseModel):
    name: str
    subcategories: list[str] = []
    sample_skills: list[str] = []


class TaxonomyContext(BaseModel):
    categories: list[TaxonomyCategory]
    taxonomy_skills: list[TaxonomySkillEntry]


class SkillMappingRequest(BaseModel):
    skills: list[NonTaxonomySkillInput]
    taxonomy_context: TaxonomyContext


# ── Response models ─────────────────────────────────────────────────────────

class SkillMappingResult(BaseModel):
    id: int
    action: str  # SYNONYM | NEW_SKILL | NO_MATCH
    # SYNONYM fields
    taxonomy_id: Optional[int] = None
    taxonomy_name: Optional[str] = None
    # NEW_SKILL fields
    canonical_name: Optional[str] = None
    normalized_name: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    skill_type: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""


# ── Endpoints ────────────────────────────────────────────────────────────────

class SkillCategorizeInput(BaseModel):
    id: int
    name: str
    skillType: Optional[str] = "TECHNICAL"
    currentCategory: Optional[str] = None


class CategoryEntry(BaseModel):
    name: str
    subcategories: list[str] = []


class SkillCategorizeRequest(BaseModel):
    skills: list[SkillCategorizeInput]
    categories: list[CategoryEntry]


class SkillCategorizeResult(BaseModel):
    id: int
    category: Optional[str] = None
    subcategory: Optional[str] = None


@router.post("/categorize", response_model=list[SkillCategorizeResult])
async def categorize_skills(request: SkillCategorizeRequest):
    """
    Assign or fix category + subcategory for existing taxonomy skills.
    All returned category/subcategory values are validated against the provided list.
    """
    if not _skill_mapping_service.is_available():
        raise HTTPException(status_code=503, detail="Service unavailable")

    if not request.skills:
        return []

    if len(request.skills) > 100:
        raise HTTPException(status_code=422, detail="Batch size exceeds limit of 100 skills per request")

    try:
        skills_dicts = [s.model_dump() for s in request.skills]
        cats_dicts = [c.model_dump() for c in request.categories]
        return await _skill_mapping_service.categorize_skills(skills_dicts, cats_dicts)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/map-non-taxonomy", response_model=list[SkillMappingResult])
async def map_non_taxonomy_skills(request: SkillMappingRequest):
    """
    Use LLM to map a batch of unrecognised skills to existing taxonomy entries.

    - skills: list of NonTaxonomySkill records (max 30 per call recommended)
    - taxonomy_context: categories + representative taxonomy skills for LLM context
    Returns one result per input skill with action=SYNONYM|NO_MATCH.
    """
    if not _skill_mapping_service.is_available():
        raise HTTPException(
            status_code=503,
            detail="Skill mapping service unavailable (Gemini API not configured)",
        )

    if not request.skills:
        return []

    if len(request.skills) > 50:
        raise HTTPException(
            status_code=422,
            detail="Batch size exceeds limit of 50 skills per request",
        )

    try:
        skills_dicts = [s.model_dump() for s in request.skills]
        context_dict = request.taxonomy_context.model_dump()
        results = await _skill_mapping_service.map_skills_to_taxonomy(skills_dicts, context_dict)
        return results
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
