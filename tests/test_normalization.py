import pytest
from app.services.gemini_service import GeminiService
from app.models.schemas import (
    ParsedResumeData,
    ParsedSkill,
    ParsedTechnicalSkill,
    ParsedExperience,
    SummaryObject,
    CareerGap,
)


@pytest.fixture
def service():
    # GeminiService with no API key — only _normalize_parsed_data is tested
    svc = GeminiService.__new__(GeminiService)
    svc.api_key = None
    svc.model = "gemini-1.5-flash"
    svc.client = None
    return svc


def make_base_data(**kwargs):
    defaults = dict(
        personal_info={},
        skills=ParsedSkill(technical_skills=[], soft_skills=[]),
        experience=[],
        education=[],
        certifications=[],
        projects=[],
        languages=[],
        summary=None,
        confidence_score=0.95,
    )
    defaults.update(kwargs)
    return ParsedResumeData(**defaults)


class TestSkillsUsedNormalization:
    def test_dict_skills_converted_to_strings(self, service):
        exp = ParsedExperience(
            company="Acme",
            skills_used=[{"name": "Python", "type": "TECHNICAL"}, {"name": "FastAPI"}],
        )
        data = make_base_data(experience=[exp])
        result = service._normalize_parsed_data(data)
        assert result.experience[0].skills_used == ["Python", "FastAPI"]

    def test_string_skills_unchanged(self, service):
        exp = ParsedExperience(company="Acme", skills_used=["Python", "FastAPI"])
        data = make_base_data(experience=[exp])
        result = service._normalize_parsed_data(data)
        assert result.experience[0].skills_used == ["Python", "FastAPI"]

    def test_empty_skills_unchanged(self, service):
        exp = ParsedExperience(company="Acme", skills_used=[])
        data = make_base_data(experience=[exp])
        result = service._normalize_parsed_data(data)
        assert result.experience[0].skills_used == []


class TestSummaryNormalization:
    def test_string_summary_wrapped_in_summary_object(self, service):
        data = make_base_data(summary="Experienced backend engineer with 5 years.")
        result = service._normalize_parsed_data(data)
        assert isinstance(result.summary, SummaryObject)
        assert result.summary.raw == "Experienced backend engineer with 5 years."

    def test_summary_object_unchanged(self, service):
        s = SummaryObject(raw="Engineer", domains=["FinTech"])
        data = make_base_data(summary=s)
        result = service._normalize_parsed_data(data)
        assert result.summary.raw == "Engineer"
        assert result.summary.domains == ["FinTech"]

    def test_none_summary_unchanged(self, service):
        data = make_base_data(summary=None)
        result = service._normalize_parsed_data(data)
        assert result.summary is None


class TestContextSnippetsBackfill:
    def test_context_backfills_context_snippets(self, service):
        skill = ParsedTechnicalSkill(
            name="React",
            context="built UI with React",
            context_snippets=[],
        )
        data = make_base_data(
            skills=ParsedSkill(technical_skills=[skill], soft_skills=[])
        )
        result = service._normalize_parsed_data(data)
        assert result.skills.technical_skills[0].context_snippets == ["built UI with React"]

    def test_existing_context_snippets_not_overwritten(self, service):
        skill = ParsedTechnicalSkill(
            name="React",
            context="built UI with React",
            context_snippets=["already here"],
        )
        data = make_base_data(
            skills=ParsedSkill(technical_skills=[skill], soft_skills=[])
        )
        result = service._normalize_parsed_data(data)
        assert result.skills.technical_skills[0].context_snippets == ["already here"]

    def test_no_context_leaves_snippets_empty(self, service):
        skill = ParsedTechnicalSkill(name="React", context=None, context_snippets=[])
        data = make_base_data(
            skills=ParsedSkill(technical_skills=[skill], soft_skills=[])
        )
        result = service._normalize_parsed_data(data)
        assert result.skills.technical_skills[0].context_snippets == []


class TestSkillCap:
    def test_skills_capped_at_40(self, service):
        skills = [ParsedTechnicalSkill(name=f"Skill{i}") for i in range(50)]
        data = make_base_data(
            skills=ParsedSkill(technical_skills=skills, soft_skills=[])
        )
        result = service._normalize_parsed_data(data)
        assert len(result.skills.technical_skills) == 40


class TestDefaultLanguage:
    def test_missing_language_defaults_to_en(self, service):
        data = make_base_data(resume_language="")
        result = service._normalize_parsed_data(data)
        assert result.resume_language == "en"

    def test_set_language_preserved(self, service):
        data = make_base_data(resume_language="fr")
        result = service._normalize_parsed_data(data)
        assert result.resume_language == "fr"


class TestCareerGapsDefault:
    def test_none_career_gaps_becomes_empty_list(self, service):
        data = make_base_data()
        data.career_gaps = None  # force None to test guard
        result = service._normalize_parsed_data(data)
        assert result.career_gaps == []
