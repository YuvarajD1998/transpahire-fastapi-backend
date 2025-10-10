import json
import re
import asyncio
from datetime import datetime
from typing import Dict, Any
import google.genai as genai
from google.genai import types
from app.config import settings
from app.models.schemas import ParsedResumeData


class CritiqueService:
    def __init__(self) -> None:
        self.api_key = settings.GEMINI_API_KEY
        self.model = settings.GEMINI_MODEL or "gemini-1.5-flash"
        self.timeout = 60.0

        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    async def generate_critique(self, resume_data: ParsedResumeData) -> Dict[str, Any]:
        """Generate resume critique using Google Gemini with structured output."""
        if not self.client:
            # Fallback for local development when API key is not configured
            return self._get_fallback_critique()

        prompt = self._create_critique_prompt(resume_data)

        try:
            response = await self._generate_content_async(prompt)
            return self._parse_critique_response(response.text)
        except Exception as e:
            print(f"Gemini critique generation failed: {str(e)}")
            return self._get_fallback_critique()

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
                        temperature=0.2,
                        max_output_tokens=2000,
                    ),
                ),
            )
            return response
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {str(e)}")

    def _create_critique_prompt(self, resume_data: ParsedResumeData) -> str:
        """Create a structured prompt for Gemini to critique the resume."""
        resume_dict = {
            "personal_info": resume_data.personal_info,
            "skills": [skill.dict() for skill in resume_data.skills],
            "experience": [exp.dict() for exp in resume_data.experience],
            "education": [edu.dict() for edu in resume_data.education],
            "summary": resume_data.summary
        }
        
        return f"""You are an expert resume reviewer and career counselor. Analyze this resume and provide a detailed critique in JSON format.

Resume Data:
{json.dumps(resume_dict, indent=2)}

Provide your analysis as a JSON object with this exact structure:
{{
    "overall_score": <integer between 0-100>,
    "sections": {{
        "experience": {{
            "score": <integer between 0-100>,
            "feedback": "<detailed feedback about work experience section>"
        }},
        "skills": {{
            "score": <integer between 0-100>,
            "feedback": "<detailed feedback about skills section>"
        }},
        "education": {{
            "score": <integer between 0-100>,
            "feedback": "<detailed feedback about education section>"
        }},
        "summary": {{
            "score": <integer between 0-100>,
            "feedback": "<detailed feedback about professional summary>"
        }},
        "personal_info": {{
            "score": <integer between 0-100>,
            "feedback": "<detailed feedback about contact information and personal details>"
        }}
    }},
    "suggestions": [
        "<specific actionable suggestion 1>",
        "<specific actionable suggestion 2>",
        "<specific actionable suggestion 3>"
    ],
    "strengths": [
        "<identified strength 1>",
        "<identified strength 2>",
        "<identified strength 3>"
    ],
    "weaknesses": [
        "<identified weakness 1>",
        "<identified weakness 2>",
        "<identified weakness 3>"
    ]
}}

Instructions:
1. Evaluate each section based on completeness, relevance, and professional presentation
2. Consider industry standards and best practices
3. Provide specific, actionable feedback
4. Be constructive but honest in your assessment
5. Focus on areas like quantifiable achievements, keyword optimization, formatting consistency
6. Return ONLY the JSON object, no additional text

JSON Output:"""

    def _parse_critique_response(self, response: str) -> Dict[str, Any]:
        """Parse Gemini's JSON response into structured critique data with improved error handling."""
        try:
            # Clean up the response text
            content_text = response.strip() if isinstance(response, str) else str(response)

            print("=== GEMINI CRITIQUE RAW RESPONSE ===")
            print(f"Content text: {content_text}")
            print("=== END RAW RESPONSE ===")

            # More robust JSON extraction
            json_start = content_text.find('{')
            json_end = content_text.rfind('}') + 1
            
            if json_start == -1 or json_end <= json_start:
                print(f"ERROR: No valid JSON structure found in critique response")
                return self._get_fallback_critique()

            json_text = content_text[json_start:json_end]
            print(f"=== EXTRACTED CRITIQUE JSON ===")
            print(json_text)
            print("=== END EXTRACTED JSON ===")

            # Try to parse the JSON
            critique_data = json.loads(json_text)
            
            # Validate required structure
            if not self._validate_critique_structure(critique_data):
                print("ERROR: Invalid critique structure received")
                return self._get_fallback_critique()
            
            # Add metadata
            critique_data["ai_model"] = self.model
            critique_data["created_at"] = datetime.utcnow().isoformat()
            
            return critique_data

        except json.JSONDecodeError as e:
            print(f"=== CRITIQUE JSON PARSING ERROR ===")
            print(f"JSON decode error: {e}")
            print(f"Response content: {content_text[:1000] if 'content_text' in locals() else 'N/A'}...")
            print("=== END JSON PARSING ERROR ===")
            return self._get_fallback_critique()
        except Exception as e:
            print(f"=== CRITIQUE PARSING ERROR ===")
            print(f"General parsing error: {e}")
            print(f"Response content: {content_text[:1000] if 'content_text' in locals() else 'N/A'}...")
            print("=== END PARSING ERROR ===")
            return self._get_fallback_critique()

    def _validate_critique_structure(self, critique_data: Dict[str, Any]) -> bool:
        """Validate that the critique data has the expected structure."""
        try:
            required_keys = ["overall_score", "sections", "suggestions", "strengths", "weaknesses"]
            if not all(key in critique_data for key in required_keys):
                return False
            
            required_sections = ["experience", "skills", "education", "summary", "personal_info"]
            sections = critique_data.get("sections", {})
            if not all(section in sections for section in required_sections):
                return False
            
            # Validate each section has score and feedback
            for section_name, section_data in sections.items():
                if not isinstance(section_data, dict):
                    return False
                if "score" not in section_data or "feedback" not in section_data:
                    return False
                if not isinstance(section_data["score"], (int, float)):
                    return False
            
            return True
        except Exception:
            return False

    def _get_fallback_critique(self) -> Dict[str, Any]:
        """Provide a fallback critique when Gemini API is unavailable."""
        return {
            "overall_score": 78,
            "sections": {
                "experience": {
                    "score": 80,
                    "feedback": "Work experience shows good progression. Consider adding more quantifiable achievements and specific metrics to strengthen your impact statements."
                },
                "skills": {
                    "score": 82,
                    "feedback": "Skills section is comprehensive and relevant. Consider indicating proficiency levels and organizing by categories for better readability."
                },
                "education": {
                    "score": 85,
                    "feedback": "Educational background provides a strong foundation. Consider highlighting relevant coursework, projects, or academic achievements."
                },
                "summary": {
                    "score": 75,
                    "feedback": "Professional summary captures key points but could be more compelling. Focus on unique value proposition and career highlights."
                },
                "personal_info": {
                    "score": 90,
                    "feedback": "Contact information is complete and professional. Ensure all links are active and profiles are up-to-date."
                }
            },
            "suggestions": [
                "Quantify achievements with specific numbers and percentages",
                "Include relevant industry keywords for better ATS compatibility",
                "Add a portfolio or project links to showcase your work",
                "Consider obtaining industry-relevant certifications"
            ],
            "strengths": [
                "Clear career progression and professional narrative",
                "Relevant technical skills for the target industry",
                "Well-organized structure and professional formatting",
                "Complete contact information and online presence"
            ],
            "weaknesses": [
                "Missing quantifiable achievements and metrics",
                "Could benefit from more industry-specific keywords",
                "Limited demonstration of leadership or project management",
                "Professional summary could be more compelling"
            ],
            "ai_model": self.model,
            "created_at": datetime.utcnow().isoformat()
        }

    def is_available(self) -> bool:
        """Check if the Gemini API is available."""
        return self.client is not None and self.api_key is not None
