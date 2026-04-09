# app/models/schemas.py

import re
from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from app.models.enums import ParseStatus, ProficiencyLevel, SkillSource, PrivacyMode

# =====================================================================
#  BASIC UPLOAD MODEL
# =====================================================================

class ResumeUpload(BaseModel):
    filename: str
    is_primary: bool = False


# =====================================================================
#  ENRICHED SKILL OBJECTS (CORE FOR PARSED DATA)
# =====================================================================

class EnrichedSkill(BaseModel):
    name: str
    raw: Optional[str] = None
    type: Optional[str] = None  # TECHNICAL / TOOL / DOMAIN / LANGUAGE / CERTIFICATION / SOFT
    group: Optional[str] = None # Programming / Frontend / Backend / Database / Cloud / Data / DevOps / Testing / Design / Other
    source_sections: List[str] = Field(default_factory=list)
    context_snippets: List[str] = Field(default_factory=list)
    explicit_level: Optional[str] = None
    explicit_years_experience: Optional[float] = None

    @field_validator("explicit_years_experience", mode="before")
    @classmethod
    def normalize_years(cls, v):
        if v is None:
            return None

        # If already float/int → safe
        if isinstance(v, (int, float)):
            return float(v)

        # If string like "3+", "4 years", "2.5 yrs"
        if isinstance(v, str):
            match = re.search(r"\d+(\.\d+)?", v)
            if match:
                return float(match.group())

        return None


class ParsedTechnicalSkill(EnrichedSkill):
    """
    Enriched technical skill (top-level skills.technical_skills).
    Extends EnrichedSkill with normalized fields used in your profile schema.
    """
    proficiency_level: Optional[ProficiencyLevel] = None
    years_experience: Optional[float] = None
    context: Optional[str] = None          # single short phrase (max 10 words)


class ParsedSoftSkill(BaseModel):
    """
    Enriched soft skill.
    Kept separate from ParsedTechnicalSkill because soft skills rarely have
    meaningful years_experience, but we still want raw + context.
    """
    name: str
    raw: Optional[str] = None
    source_sections: List[str] = Field(default_factory=list)
    context_snippets: List[str] = Field(default_factory=list)
    context: Optional[str] = None          # single short phrase (max 10 words)
    proficiency_level: Optional[ProficiencyLevel] = ProficiencyLevel.INTERMEDIATE


class ParsedSkill(BaseModel):
    """
    Aggregated skills object for the parsed resume.
    """
    technical_skills: List[ParsedTechnicalSkill] = Field(default_factory=list)
    soft_skills: List[ParsedSoftSkill] = Field(default_factory=list)


# =====================================================================
#  EXPERIENCE / EDUCATION / RESUME DATA
# =====================================================================

class ParsedExperience(BaseModel):
    company: Optional[str] = None
    position: Optional[str] = None
    location: Optional[str] = None
    industry: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: bool = False
    description: Optional[str] = None
    achievements: List[str] = Field(default_factory=list)
    skills_used: List[str] = []   # string names only — feeds NestJS taxonomy join



class ParsedEducation(BaseModel):
    institution: str
    degree: str
    field: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    grade: Optional[str] = None
    # NEW: align with prompt (education.description)
    description: Optional[str] = None
    context_snippets: List[str] = Field(default_factory=list)

# =====================================================================
#  CERTIFICATIONS / PROJECTS / LANGUAGES
# =====================================================================

class ParsedCertification(BaseModel):
    name: str
    organization: Optional[str] = None
    issue_date: Optional[str] = None  # YYYY-MM-DD
    expiry_date: Optional[str] = None  # YYYY-MM-DD


class ParsedProject(BaseModel):
    name: str
    description: Optional[str] = None
    technologies: List[str] = Field(default_factory=list)
    role: Optional[str] = None
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None


class ParsedLanguage(BaseModel):
    name: str
    proficiency: Optional[str] = None  # Basic / Intermediate / Fluent / Native


class SummaryObject(BaseModel):
    raw: Optional[str] = None
    years_experience: Optional[float] = None   # only if explicitly stated in resume
    domains: List[str] = []                    # e.g. ["E-commerce", "Healthcare IT"]


class CareerGap(BaseModel):
    start_date: str          # YYYY-MM-DD
    end_date: str            # YYYY-MM-DD
    duration_months: int     # calculated


class ParsedResumeData(BaseModel):
    personal_info: Dict[str, Any] = Field(default_factory=dict)
    skills: ParsedSkill = Field(default_factory=ParsedSkill)
    experience: List[ParsedExperience] = Field(default_factory=list)
    education: List[ParsedEducation] = Field(default_factory=list)

    # Newly added fields (previously missing)
    certifications: List[ParsedCertification] = Field(default_factory=list)
    projects: List[ParsedProject] = Field(default_factory=list)
    languages: List[ParsedLanguage] = Field(default_factory=list)

    summary: Optional[SummaryObject] = None
    confidence_score: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)

    resume_language: str = "en"
    career_gaps: List[CareerGap] = Field(default_factory=list)
    total_experience_months: Optional[int] = None


class ParseRequest(BaseModel):
    resume_id: int
    file_content: str  # Base64 encoded file content
    filename: str
    enhance_images: bool = True

class ParseResponse(BaseModel):
    success: bool
    resume_id: int
    parsed_data: Optional[ParsedResumeData] = None
    error: Optional[str] = None
    confidence_score: float = 0.0


# =====================================================================
#  RESUME RESPONSE (DB ↔ API)
# =====================================================================

class ResumeResponse(BaseModel):
    id: int
    filename: str
    original_name: str
    file_path: str
    file_size: int
    mimetype: str
    parse_status: ParseStatus
    confidence_score: Optional[float] = None
    is_primary: bool
    created_at: datetime
    parsed_data: Optional[ParsedResumeData] = None
    
    class Config:
        from_attributes = True


# =====================================================================
#  CRITIQUE / FEEDBACK MODELS
# =====================================================================

class CritiqueSection(BaseModel):
    score: int = Field(..., ge=0, le=100, description="Score between 0-100")
    feedback: str = Field(..., min_length=1, description="Detailed feedback for this section")


class CritiqueSections(BaseModel):
    experience: CritiqueSection
    skills: CritiqueSection
    education: CritiqueSection
    summary: CritiqueSection
    personal_info: CritiqueSection


class CritiqueData(BaseModel):
    overall_score: int = Field(..., ge=0, le=100, description="Overall resume score between 0-100")
    sections: CritiqueSections
    suggestions: List[str] = Field(default_factory=list, description="List of actionable suggestions")
    strengths: List[str] = Field(default_factory=list, description="List of identified strengths")
    weaknesses: List[str] = Field(default_factory=list, description="List of identified weaknesses")
    ai_model: str = Field(default="gemini-1.5-flash", description="AI model used for critique")

    class Config:
        json_schema_extra = {
            "example": {
                "overall_score": 78,
                "sections": {
                    "experience": {
                        "score": 80,
                        "feedback": "Good work experience but needs more quantifiable achievements"
                    },
                    "skills": {
                        "score": 85,
                        "feedback": "Comprehensive skill set with modern technologies"
                    },
                    "education": {
                        "score": 75,
                        "feedback": "Solid educational background"
                    },
                    "summary": {
                        "score": 70,
                        "feedback": "Professional summary needs to be more compelling"
                    },
                    "personal_info": {
                        "score": 90,
                        "feedback": "Complete and professional contact information"
                    }
                },
                "suggestions": [
                    "Add quantifiable achievements with metrics",
                    "Include relevant industry keywords"
                ],
                "strengths": [
                    "Clear career progression",
                    "Relevant technical skills"
                ],
                "weaknesses": [
                    "Missing quantifiable achievements",
                    "Could use more industry keywords"
                ],
                "ai_model": "gemini-1.5-flash",
                "created_at": "2025-09-26T14:50:00Z"
            }
        }


class ResumeCritiqueResponse(BaseModel):
    id: int
    resume_id: int
    critique_data: CritiqueData
    created_at: datetime

    class Config:
        from_attributes = True


# =====================================================================
#  PROFILE-RELATED MODELS
# =====================================================================

class ProfileSkillResponse(BaseModel):
    id: int
    skill_name: str
    category: Optional[str] = None
    proficiency_level: Optional[ProficiencyLevel] = None
    years_experience: Optional[int] = None
    source: SkillSource
    verified: bool = False
    
    class Config:
        from_attributes = True


class ProfileUpdateRequest(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    headline: Optional[str] = None
    bio: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    website_url: Optional[str] = None
    privacy_mode: Optional[PrivacyMode] = None


class ProfileResponse(BaseModel):
    id: int
    user_id: int
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    headline: Optional[str] = None
    bio: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    website_url: Optional[str] = None
    profile_completeness: int = 0
    privacy_mode: PrivacyMode
    created_at: datetime
    updated_at: datetime
    embeddings_generated: bool = False
    embeddings_version: Optional[int] = None
    
    class Config:
        from_attributes = True


class CreateProfileSkillRequest(BaseModel):
    skill_name: str
    category: Optional[str] = None
    proficiency_level: Optional[ProficiencyLevel] = None
    years_experience: Optional[int] = None
    source: SkillSource = SkillSource.MANUAL


class UpdateProfileSkillRequest(BaseModel):
    category: Optional[str] = None
    proficiency_level: Optional[ProficiencyLevel] = None
    years_experience: Optional[int] = None
    verified: Optional[bool] = None


# =====================================================================
#  EMBEDDING-RELATED SCHEMAS
# =====================================================================

class EmbeddingStatusResponse(BaseModel):
    """Response schema for embedding status"""
    profile_id: int
    has_embeddings: bool
    embedding_model: Optional[str] = None
    embedding_dimensions: Optional[int] = None
    embedding_version: Optional[int] = None
    created_at: Optional[str] = None  # ISO format string
    updated_at: Optional[str] = None  # ISO format string


class SimilarProfileResponse(BaseModel):
    """Response schema for similar profile data"""
    profile_id: int
    name: str
    location: Optional[str] = None
    headline: Optional[str] = None
    bio: Optional[str] = None
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score between 0 and 1")
    profile_completeness: int = Field(..., ge=0, le=100)
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None


class SimilarProfilesResponse(BaseModel):
    """Response schema for similar profiles search"""
    target_profile_id: int
    embedding_model: str
    embedding_dimensions: int
    total_found: int
    similar_profiles: List[SimilarProfileResponse]


class EmbeddingGenerationRequest(BaseModel):
    """Request schema for embedding generation"""
    profile_id: Optional[int] = None
    batch_size: Optional[int] = Field(default=50, ge=1, le=100)


class EmbeddingGenerationResponse(BaseModel):
    """Response schema for embedding generation"""
    message: str
    profile_id: Optional[int] = None
    total_pending: Optional[int] = None
    batch_size: Optional[int] = None
    status: str  # "processing", "completed", "exists", "failed"


class BatchEmbeddingResult(BaseModel):
    """Response schema for batch embedding results"""
    processed: int
    success: int
    errors: int


class EmbeddingStatsResponse(BaseModel):
    """Response schema for embedding statistics"""
    total_profiles: int
    profiles_with_embeddings: int
    profiles_without_embeddings: int
    total_embeddings: int
    embedding_coverage: float = Field(..., ge=0.0, le=100.0, description="Percentage of profiles with embeddings")
    embedding_model: str


class EmbeddingFilterRequest(BaseModel):
    """Request schema for embedding search filters"""
    location: Optional[str] = None
    skills: Optional[List[str]] = None
    min_experience: Optional[int] = None
    max_distance: Optional[float] = Field(default=1.0, ge=0.0, le=2.0, description="Maximum vector distance")


# Update your existing ProfileResponse to include embedding status
class EnhancedProfileResponse(ProfileResponse):
    """Enhanced profile response with embedding status"""
    embeddings_generated: bool = False
    embeddings_version: Optional[int] = None
    
    class Config:
        from_attributes = True


# =====================================================================
#  JOB MATCHING / EMBEDDING SEARCH
# =====================================================================

class JobEmbeddingResponse(BaseModel):
    """Response schema for job embedding data"""
    job_id: int
    title: str
    company: str
    location: Optional[str] = None
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    

class JobMatchResponse(BaseModel):
    """Response schema for job matching results"""
    profile_id: int
    matched_jobs: List[JobEmbeddingResponse]
    total_matches: int
    search_criteria: Dict[str, Any]


class EmbeddingErrorResponse(BaseModel):
    """Error response schema for embedding operations"""
    error: str
    details: Optional[Dict[str, Any]] = None
    profile_id: Optional[int] = None
    

# =====================================================================
#  CONFIG / CONTROL SCHEMAS
# =====================================================================

class EmbeddingConfig(BaseModel):
    """Configuration schema for embedding settings"""
    model_name: str = "text-embedding-004"
    dimensions: int = 768
    task_type: str = "RETRIEVAL_DOCUMENT"
    batch_size: int = 50
    rate_limit_rpm: int = 1500  # Gemini free tier limit


class SetPrimaryResumeRequest(BaseModel):
    profile_id: int
    resume_id: int


class SetPrimaryResumeResponse(BaseModel):
    message: str
    profile_id: int
    resume_id: int
    embeddings_regenerated: bool
    data_updated: bool
