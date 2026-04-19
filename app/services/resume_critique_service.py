import json
import logging
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
Keep each feedback string under 60 words so the full JSON fits within the output limit.
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
            result_text = await self.gemini.generate_text(prompt, max_tokens=6000)
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
        cleaned = re.sub(r'```(?:json)?', '', text).strip()
        if not cleaned:
            logger.warning(f"Critique response empty after cleaning; raw length={len(text)}, raw={text[:200]!r}")
            raise ValueError("Gemini returned empty response for critique")
        try:
            data = json.loads(cleaned)
            return {
                "overall_score": max(0, min(100, int(data.get("overall_score", 50)))),
                "sections": data.get("sections", {}),
                "strengths": data.get("strengths", [])[:5],
                "weaknesses": data.get("weaknesses", [])[:5],
                "suggestions": data.get("suggestions", [])[:5],
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse critique JSON; cleaned={cleaned[:300]!r}")
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
