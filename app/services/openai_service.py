import json
import re
from openai import AsyncOpenAI
from app.config import settings
from app.models.schemas import (
    ParsedResumeData, ParsedSkill, ParsedTechnicalSkill,
    ParsedExperience, ParsedEducation, SummaryObject,
)


class OpenAIService:
    def __init__(self) -> None:
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL or "gpt-3.5-turbo"
        self.timeout = 60.0
        self._client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None

    async def parse_resume_text(self, text: str) -> ParsedResumeData:
        if not self._client:
            raise RuntimeError("OpenAI API key not configured")

        prompt = self._create_analysis_prompt(text)

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4096,
                timeout=self.timeout,
            )
            content = response.choices[0].message.content
            return self._parse_openai_response(content)

        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")

    def _create_analysis_prompt(self, resume_text: str) -> str:
        return f"""Analyze this resume and extract structured information. Return ONLY valid JSON.

Resume:
{resume_text[:6000]}

Return JSON with this structure:
{{
    "personal_info": {{"name": null, "email": null, "phone": null, "location": null, "linkedin": null, "github": null}},
    "skills": [{{"name": "string", "group": "string or null", "years_experience": null}}],
    "experience": [{{"company": "string", "position": "string", "start_date": null, "end_date": null, "is_current": false, "description": null, "achievements": []}}],
    "education": [{{"institution": "string", "degree": "string", "field": null, "start_date": null, "end_date": null}}],
    "summary": "string or null"
}}

JSON:"""

    def _parse_openai_response(self, content: str) -> ParsedResumeData:
        try:
            stripped = content.strip()
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', stripped, re.DOTALL)
                if not json_match:
                    raise ValueError("No JSON found in OpenAI response")
                obj = json.loads(json_match.group())

            technical_skills = [
                ParsedTechnicalSkill(
                    name=s.get("name", ""),
                    group=s.get("group"),
                    explicit_years_experience=s.get("years_experience"),
                )
                for s in obj.get("skills", [])
                if isinstance(s, dict) and s.get("name")
            ]

            experience = [
                ParsedExperience(**{k: v for k, v in e.items() if k in ParsedExperience.model_fields})
                for e in obj.get("experience", [])
                if isinstance(e, dict) and e.get("company")
            ]

            education = [
                ParsedEducation(**{k: v for k, v in e.items() if k in ParsedEducation.model_fields})
                for e in obj.get("education", [])
                if isinstance(e, dict) and e.get("institution")
            ]

            raw_summary = obj.get("summary")
            summary = SummaryObject(raw=raw_summary) if raw_summary else None

            return ParsedResumeData(
                personal_info=obj.get("personal_info", {}),
                skills=ParsedSkill(technical_skills=technical_skills),
                experience=experience,
                education=education,
                summary=summary,
                confidence_score=0.9,
            )

        except Exception as e:
            raise RuntimeError(f"OpenAI response parsing failed: {e}")
