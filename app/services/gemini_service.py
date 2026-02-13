import json
import re
import asyncio
from typing import Optional
import google.genai as genai
from google.genai import types

from app.config import settings
from app.models.schemas import (
    ParsedResumeData, 
    ParsedSkill, 
    ParsedTechnicalSkill, 
    ParsedSoftSkill, 
    ParsedExperience, 
    ParsedEducation, 
    ParsedCertification, 
    ParsedProject, 
    ParsedLanguage,
    EnrichedSkill
)


class GeminiService:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model = settings.GEMINI_MODEL
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    def is_available(self) -> bool:
        """Check if Gemini service is available."""
        return self.client is not None

    async def parse_resume_text(self, text: str) -> ParsedResumeData:
        """Parse resume text using Google Gemini."""
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
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=65536,
                ),
            ),
        )
        return response

    def _create_structured_prompt(self, resume_text: str) -> str:
        """Create a structured prompt for Gemini to extract resume data."""
        return f"""You are an expert enterprise-grade resume parser and HR data architect.

Extract structured, detailed information from this resume text and return ONLY valid JSON, with no explanations, no commentary, and no markdown.

Resume Text:
{resume_text[:50000]}

Return a JSON object with this exact structure:

{{
  "personal_info": {{
    "name": "Full name or null",
    "email": "Email address or null",
    "phone": "Phone number or null",
    "location": "City, State, Country or null",
    "linkedin": "LinkedIn profile URL or null",
    "github": "GitHub or portfolio URL or null",
    "website": "Personal/portfolio website or null"
  }},
  "summary": "Professional summary or null",

  "skills": {{
    "technical_skills": [
      {{
        "name": "Canonical skill name (e.g., 'React', '3D Modeling')",
        "raw": "Original phrase from the resume (e.g., 'React.js', '3D modeling using SolidWorks')",
        "type": "TECHNICAL/TOOL/DOMAIN/LANGUAGE/CERTIFICATION",
        "group": "Programming/Frontend/Backend/Database/Cloud/DevOps/Data/Testing/Design/Other",
        "source_sections": ["skills_section", "experience", "projects", "summary"],
        "context_snippets": [
          "Short sentence or bullet that proves usage of this skill from the resume."
        ],
        "explicit_level": null,
        "explicit_years_experience": null
      }}
    ],
    "soft_skills": [
      {{
        "name": "Soft skill name (e.g., 'Communication', 'Teamwork', 'Leadership')",
        "raw": "Original phrase from the resume if available",
        "source_sections": ["summary", "experience", "projects"],
        "context_snippets": [
          "Optional short sentence that supports this soft skill, if present in the resume."
        ]
      }}
    ]
  }},

  "experience": [
    {{
      "company": "Company name",
      "position": "Job title",
      "location": "City, Country or null",
      "industry": "Industry or domain (IT, Healthcare, Finance, Education, Manufacturing, etc.)",
      "start_date": "YYYY-MM-DD or null",
      "end_date": "YYYY-MM-DD or null",
      "is_current": false,
      "description": "Role summary and responsibilities",
      "achievements": ["Achievement 1", "Achievement 2"],
      "skills_used": [
        {{
          "name": "Skill name",
          "raw": "Original phrase from resume or null",
          "type": "TECHNICAL/TOOL/DOMAIN/SOFT",
          "group": "Programming/Frontend/Backend/Database/Cloud/DevOps/Data/Testing/Design/Other",
          "source_sections": ["experience"],
          "context_snippets": ["Context from this job description"],
          "explicit_level": null,
          "explicit_years_experience": null
        }}
      ]
    }}
  ],

  "education": [
    {{
      "institution": "University or school name",
      "degree": "Degree type or null (e.g., B.Tech, MBA, Diploma)",
      "field": "Field of study or specialization or null",
      "start_date": "YYYY-MM-DD or null",
      "end_date": "YYYY-MM-DD or null",
      "grade": "GPA or percentage or null",
      "description": "Additional academic details or projects",
      "context_snippets": [
        "Short bullet or sentence from the resume that shows coursework, projects, or hands-on technical activities relevant to this education entry."
      ]
    }}
  ],

  "certifications": [
    {{
      "name": "Certification name",
      "organization": "Issuing body or null",
      "issue_date": "YYYY-MM-DD or null",
      "expiry_date": "YYYY-MM-DD or null"
    }}
  ],

  "projects": [
    {{
      "name": "Project name",
      "description": "Project summary and objective",
      "technologies": ["Tech 1", "Tech 2"],
      "role": "Role or contribution",
      "start_date": "YYYY-MM-DD or null",
      "end_date": "YYYY-MM-DD or null"
    }}
  ],

  "languages": [
    {{
      "name": "Language name",
      "proficiency": "Basic/Intermediate/Fluent/Native"
    }}
  ]
}}

CRITICAL INSTRUCTIONS:
1. Extract only information that is explicitly present in the resume text. Do NOT hallucinate skills, companies, dates, or technologies.
2. For each TECHNICAL skill, always provide:
   - name
   - raw
   - type
   - group
   - source_sections (at least one)
   - at least one meaningful context_snippet, if possible.
3. **IMPORTANT: skills_used in experience MUST be an array of objects with the same structure as technical_skills, NOT just an array of strings.**
4. For explicit_level and explicit_years_experience:
   - ONLY fill these if the resume clearly states the level or years (e.g., "5+ years of Java", "Advanced in Excel").
   - Otherwise, set them to null. Do NOT guess.
5. Soft skills must come from evidence in the text. If no evidence is present, omit them.
6. Use null for any missing field values.
7. ALWAYS return clean, valid JSON that can be parsed directly. Do not include any extra text before or after the JSON.
8. All dates must be in YYYY-MM-DD format when possible; otherwise null.

JSON Output:
"""

    def _parse_gemini_response(self, response_text: str) -> ParsedResumeData:
        """Parse Gemini's JSON response into structured data."""
        try:
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in Gemini response")

            json_text = json_match.group()
            obj = json.loads(json_text)

            # Parse skills
            skills_section = obj.get("skills", {}) or {}
            technical_skills_raw = skills_section.get("technical_skills", []) or []
            soft_skills_raw = skills_section.get("soft_skills", []) or []

            technical_skills = [
                ParsedTechnicalSkill(**s) for s in technical_skills_raw if isinstance(s, dict) and s.get("name")
            ]
            soft_skills = [
                ParsedSoftSkill(**s) for s in soft_skills_raw if isinstance(s, dict) and s.get("name")
            ]

            # Parse experience with skills_used fallback
            experience_raw = obj.get("experience", []) or []
            experience = []
            
            for e in experience_raw:
                if not isinstance(e, dict) or not e.get("company"):
                    continue
                
                # Fix skills_used if it's a list of strings
                skills_used = e.get("skills_used", [])
                if skills_used:
                    normalized_skills = []
                    for skill in skills_used:
                        if isinstance(skill, str):
                            # Convert string to EnrichedSkill object
                            normalized_skills.append(
                                EnrichedSkill(
                                    name=skill,
                                    raw=skill,
                                    type="TECHNICAL",
                                    group="Other",
                                    source_sections=["experience"],
                                    context_snippets=[]
                                )
                            )
                        elif isinstance(skill, dict):
                            # Already an object, use as-is
                            normalized_skills.append(EnrichedSkill(**skill))
                    
                    e["skills_used"] = normalized_skills
                
                experience.append(ParsedExperience(**e))

            # Parse education
            education_raw = obj.get("education", []) or []
            education = [
                ParsedEducation(**ed) for ed in education_raw if isinstance(ed, dict) and ed.get("institution")
            ]

            # Parse certifications
            certifications_raw = obj.get("certifications", []) or []
            certifications = [
                ParsedCertification(**cert) for cert in certifications_raw if isinstance(cert, dict) and cert.get("name")
            ]

            # Parse projects
            projects_raw = obj.get("projects", []) or []
            projects = [
                ParsedProject(**proj) for proj in projects_raw if isinstance(proj, dict) and proj.get("name")
            ]

            # Parse languages
            languages_raw = obj.get("languages", []) or []
            languages = [
                ParsedLanguage(**lang) for lang in languages_raw if isinstance(lang, dict) and lang.get("name")
            ]

            return ParsedResumeData(
                personal_info=obj.get("personal_info", {}) or {},
                skills=ParsedSkill(
                    technical_skills=technical_skills,
                    soft_skills=soft_skills,
                ),
                experience=experience,
                education=education,
                certifications=certifications,
                projects=projects,
                languages=languages,
                summary=obj.get("summary"),
                confidence_score=0.95,
            )

        except Exception as e:
            print(f"Parsing error: {e}")
            # Return minimal valid structure on error
            return ParsedResumeData(
                personal_info={},
                skills=ParsedSkill(technical_skills=[], soft_skills=[]),
                experience=[],
                education=[],
                certifications=[],
                projects=[],
                languages=[],
                summary=None,
                confidence_score=0.1,
            )
