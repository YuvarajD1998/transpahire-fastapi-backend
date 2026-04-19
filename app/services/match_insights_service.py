import json
import logging
import re
from app.services.gemini_service import GeminiService
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
            result_text = await self.gemini.generate_text(prompt)
            parsed = self._parse_insights_response(result_text)
            return MatchInsightsResponse(**parsed)
        except Exception as e:
            logger.error(f"Match insights generation failed for candidate={candidate_id} job={job_id}: {e}")
            return self._fallback_insights(features)

    def _parse_insights_response(self, text: str) -> dict:
        text = re.sub(r'```(?:json)?', '', text).strip()
        try:
            data = json.loads(text)
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
