import httpx
import json
import re
from typing import Dict, Any, Optional
from app.config import settings
from app.models.schemas import ParsedResumeData, ParsedSkill, ParsedExperience, ParsedEducation


class HuggingFaceService:
    def __init__(self) -> None:
        self.api_key = settings.HUGGINGFACE_API_KEY
        self.model_id = settings.HUGGINGFACE_MODEL_ID or "mistralai/Mistral-7B-Instruct-v0.1"
        self.base_url = "https://api-inference.huggingface.co"
        self.timeout = 120.0  # Increased timeout
        print(f"HuggingFaceService initialized with model {self.model_id}")

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _create_structured_prompt(self, resume_text: str) -> str:
        return f"""Extract structured information from this resume. Return ONLY valid JSON with this exact structure:
{{
    "personal_info": {{
        "name": "string or null",
        "email": "string or null", 
        "phone": "string or null",
        "location": "string or null",
        "linkedin": "string or null",
        "github": "string or null"
    }},
    "skills": [
        {{
            "name": "string",
            "category": "string", 
            "level": "Beginner/Intermediate/Expert",
            "years_experience": "number or null"
        }}
    ],
    "experience": [
        {{
            "company": "string",
            "position": "string",
            "location": "string or null",
            "start_date": "YYYY-MM-DD or null",
            "end_date": "YYYY-MM-DD or null", 
            "is_current": "boolean",
            "description": "string",
            "achievements": ["string"],
            "skills": ["string"]
        }}
    ],
    "education": [
        {{
            "institution": "string",
            "degree": "string or null",
            "field": "string or null", 
            "start_date": "YYYY-MM-DD or null",
            "end_date": "YYYY-MM-DD or null",
            "grade": "string or null",
            "description": "string or null"
        }}
    ],
    "summary": "string or null"
}}

Resume Text:
{resume_text[:8000]}  # Limit text length

JSON Output:"""

    async def parse_resume_text(self, text: str) -> ParsedResumeData:
        if not self.api_key:
            raise RuntimeError("Hugging Face API key not configured")
        
        if not text.strip():
            raise ValueError("Empty resume text provided")

        payload = {
            "inputs": self._create_structured_prompt(text),
            "parameters": {
                "max_new_tokens": 2500,
                "temperature": 0.1,
                "do_sample": False,
                "return_full_text": False
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/models/{self.model_id}",
                    headers=self._headers(),
                    json=payload,
                )
                
                if resp.status_code != 200:
                    raise RuntimeError(f"HF API error {resp.status_code}: {resp.text}")
                
                data = resp.json()
                generated_text = self._extract_generated_text(data)
                return self._parse_json_response(generated_text)
                
            except httpx.TimeoutException:
                raise RuntimeError("HuggingFace API timeout")
            except Exception as e:
                raise RuntimeError(f"HuggingFace API error: {str(e)}")

    def _extract_generated_text(self, data: Any) -> str:
        """Extract generated text from HF API response."""
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "")
        elif isinstance(data, dict):
            return data.get("generated_text", "")
        return ""

    def _parse_json_response(self, generated_text: str) -> ParsedResumeData:
        """Parse JSON response with enhanced error handling."""
        try:
            # Extract JSON from text response
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            json_str = json_match.group()
            obj = json.loads(json_str)
            
            # Validate and convert skills
            skills = []
            for skill_data in obj.get("skills", []):
                if isinstance(skill_data, dict) and "name" in skill_data:
                    skills.append(ParsedSkill(**{
                        "name": skill_data.get("name", ""),
                        "category": skill_data.get("category"),
                        "level": skill_data.get("level", "Intermediate"),
                        "years_experience": skill_data.get("years_experience")
                    }))
            
            # Validate and convert experience
            experience = []
            for exp_data in obj.get("experience", []):
                if isinstance(exp_data, dict) and "company" in exp_data:
                    experience.append(ParsedExperience(**exp_data))
            
            # Validate and convert education
            education = []
            for edu_data in obj.get("education", []):
                if isinstance(edu_data, dict) and "institution" in edu_data:
                    education.append(ParsedEducation(**edu_data))
            
            return ParsedResumeData(
                personal_info=obj.get("personal_info", {}),
                skills=skills,
                experience=experience,
                education=education,
                summary=obj.get("summary"),
                confidence_score=0.8  # HF-specific confidence
            )
            
        except Exception as e:
            print(f"JSON parsing failed: {e}")
            # Return safe fallback
            return ParsedResumeData(
                personal_info={}, skills=[], experience=[], education=[], summary=None, confidence_score=0.1
            )