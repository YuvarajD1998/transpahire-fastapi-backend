import json
import re
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
import google.genai as genai
from google.genai import types
from app.config import settings
from app.models.schemas import ParsedResumeData, ParsedSkill, ParsedExperience, ParsedEducation


class GeminiService:
    def __init__(self) -> None:
        self.api_key = settings.GEMINI_API_KEY
        self.model = settings.GEMINI_MODEL or "gemini-1.5-flash"
        self.timeout = 60.0

        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    async def parse_resume_text(self, text: str) -> ParsedResumeData:
        """Parse resume text using Google Gemini with structured output."""
        if not self.client:
            raise RuntimeError("Gemini API key not configured")

        prompt = self._create_structured_prompt(text)

        try:
            response = await self._generate_content_async(prompt)
            return self._parse_gemini_response(response.text)
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")

    async def _generate_content_async(self, prompt: str):
        """Generate content using Gemini with async support."""
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=12000,
                    ),
                ),
            )
            return response
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {str(e)}")

    def _create_structured_prompt(self, resume_text: str) -> str:
        """Create a structured prompt for Gemini to extract resume data."""
        return f"""You are an expert resume parser. Extract structured information from this resume and return ONLY valid JSON.

Resume Text:
{resume_text[:16000]}

Return a JSON object with this exact structure:
{{
    "personal_info": {{
        "name": "Full name or null",
        "email": "Email address or null",
        "phone": "Phone number or null",
        "location": "Location/address or null",
        "linkedin": "LinkedIn URL or null",
        "github": "GitHub URL or null"
    }},
    "skills": [
        {{
            "name": "Skill name",
            "category": "Programming/Web/Database/Cloud/Data Science/Tools/Soft Skills",
            "level": "Beginner/Intermediate/Expert",
            "years_experience": null
        }}
    ],
    "experience": [
        {{
            "company": "Company name",
            "position": "Job title",
            "location": "Work location or null",
            "start_date": "YYYY-MM-DD or null",
            "end_date": "YYYY-MM-DD or null",
            "is_current": false,
            "description": "Job description",
            "achievements": ["Achievement 1", "Achievement 2"],
            "skills": ["Skill 1", "Skill 2"]
        }}
    ],
    "education": [
        {{
            "institution": "School/University name",
            "degree": "Degree type or null",
            "field": "Field of study or null",
            "start_date": "YYYY-MM-DD or null",
            "end_date": "YYYY-MM-DD or null",
            "grade": "GPA/Grade or null",
            "description": "Additional details or null"
        }}
    ],
    "summary": "Professional summary or null"
}}

Instructions:
1. Extract information accurately from the resume text
2. Use null for missing information
3. Categorize skills appropriately
4. Format dates as YYYY-MM-DD when possible
5. Return ONLY the JSON object, no additional text

JSON Output:"""

    def _normalize_date(self, date_str: Optional[str]) -> Optional[str]:
        """Convert a date string (YYYY-MM-DD) to ISO format string for JSON serialization."""
        if not date_str:
            return None
        try:
            # Parse the date and return as ISO string instead of datetime object
            dt = datetime.fromisoformat(date_str)
            return dt.isoformat()
        except ValueError:
            return None

    def _parse_gemini_response(self, response) -> ParsedResumeData:
        """Parse Gemini's JSON response into structured data."""
        try:
            # If response is a string, parse directly
            if isinstance(response, str):
                content_text = response
            # If response has candidates (older SDK style)
            elif hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                content_text = ""
                if hasattr(candidate, "content") and candidate.content:
                    if hasattr(candidate.content, "parts") and candidate.content.parts:
                        content_text = "".join(
                            part.text for part in candidate.content.parts if hasattr(part, "text")
                        )
            else:
                raise ValueError("Unexpected Gemini response format")

            # Console log the raw response before validation
            print("=== GEMINI RAW RESPONSE ===")
            print(f"Response type: {type(response)}")
            print(f"Content text: {content_text}")
            print("=== END RAW RESPONSE ===")

            # Extract JSON from the text
            json_match = re.search(r'\{.*\}', content_text, re.DOTALL)
            if not json_match:
                print(f"ERROR: No JSON found in response: {content_text[:500]}...")
                raise ValueError("No JSON found in Gemini response")

            json_text = json_match.group()
            print(f"=== EXTRACTED JSON ===")
            print(json_text)
            print("=== END EXTRACTED JSON ===")

            obj = json.loads(json_text)
            
            print(f"=== PARSED OBJECT ===")
            print(f"Parsed JSON keys: {list(obj.keys()) if isinstance(obj, dict) else 'Not a dict'}")
            print("=== END PARSED OBJECT ===")

            # Map skills
            skills = [
                ParsedSkill(
                    name=s.get("name", ""),
                    category=s.get("category", "General"),
                    proficiency_level=s.get("level", "Intermediate"),
                    years_experience=s.get("years_experience"),
                )
                for s in obj.get("skills", [])
                if isinstance(s, dict) and s.get("name")
            ]

            # Map experience with string dates
            experience = [
                ParsedExperience(
                    company=e.get("company", "Unknown") or "Unknown",
                    position=e.get("position") or "Not specified",
                    location=e.get("location"),
                    start_date=self._normalize_date(e.get("start_date")),
                    end_date=self._normalize_date(e.get("end_date")),
                    is_current=e.get("is_current", False),
                    description=e.get("description") or "",
                    achievements=e.get("achievements", []),
                    skills=e.get("skills", []),
                )
                for e in obj.get("experience", [])
                if isinstance(e, dict) and e.get("company")
            ]

            # Map education with string dates
            education = [
                ParsedEducation(
                    institution=ed.get("institution"),
                    degree=ed.get("degree", "Not specified"),
                    field=ed.get("field"),
                    start_date=self._normalize_date(ed.get("start_date")),
                    end_date=self._normalize_date(ed.get("end_date")),
                    grade=ed.get("grade"),
                    description=ed.get("description"),
                )
                for ed in obj.get("education", [])
                if isinstance(ed, dict) and ed.get("institution")
            ]

            return ParsedResumeData(
                personal_info=obj.get("personal_info", {}),
                skills=skills,
                experience=experience,
                education=education,
                summary=obj.get("summary"),
                confidence_score=0.95,
            )

        except Exception as e:
            print(f"=== PARSING ERROR ===")
            print(f"Gemini response parsing failed: {e}")
            print(f"Response content: {content_text[:1000] if 'content_text' in locals() else 'N/A'}...")
            print("=== END PARSING ERROR ===")
            return ParsedResumeData(
                personal_info={},
                skills=[],
                experience=[],
                education=[],
                summary=None,
                confidence_score=0.1,
            )

    def is_available(self) -> bool:
        return self.client is not None and self.api_key is not None
