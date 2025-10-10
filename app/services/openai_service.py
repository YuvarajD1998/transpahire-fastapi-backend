import openai
import json
import re
from typing import Dict, Any
from app.config import settings
from app.models.schemas import ParsedResumeData, ParsedSkill, ParsedExperience, ParsedEducation


class OpenAIService:
    def __init__(self) -> None:
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL or "gpt-3.5-turbo"
        self.timeout = 60.0
        
        if self.api_key:
            openai.api_key = self.api_key

    async def parse_resume_text(self, text: str) -> ParsedResumeData:
        if not self.api_key:
            raise RuntimeError("OpenAI API key not configured")

        prompt = self._create_analysis_prompt(text)
        
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
                timeout=self.timeout
            )
            
            content = response.choices[0].message.content
            return self._parse_openai_response(content)
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")

    def _create_analysis_prompt(self, resume_text: str) -> str:
        return f"""Analyze this resume and extract structured information. Return ONLY valid JSON:

Resume:
{resume_text[:6000]}

Extract:
1. personal_info (name, email, phone, location, links)
2. skills (name, category, proficiency)
3. experience (company, position, dates, description)
4. education (institution, degree, dates)
5. professional_summary

Return JSON with this structure:
{{
    "personal_info": {{...}},
    "skills": [...],
    "experience": [...], 
    "education": [...],
    "summary": "..."
}}

JSON:"""

    def _parse_openai_response(self, content: str) -> ParsedResumeData:
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in OpenAI response")
            
            obj = json.loads(json_match.group())
            
            skills = [ParsedSkill(**s) for s in obj.get("skills", [])]
            experience = [ParsedExperience(**e) for e in obj.get("experience", [])]
            education = [ParsedEducation(**e) for e in obj.get("education", [])]
            
            return ParsedResumeData(
                personal_info=obj.get("personal_info", {}),
                skills=skills,
                experience=experience,
                education=education,
                summary=obj.get("summary"),
                confidence_score=0.9  # Higher confidence for OpenAI
            )
            
        except Exception as e:
            print(f"OpenAI response parsing failed: {e}")
            return ParsedResumeData(
                personal_info={}, skills=[], experience=[], education=[], summary=None, confidence_score=0.1
            )