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
    SummaryObject,
    CareerGap,
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

        prompt = self._build_extraction_prompt(text)

        try:
            response = await self._generate_content_async(prompt)
            print(f"[Gemini raw output]\n{response.text}\n[/Gemini raw output]")
            parsed = self._parse_gemini_response(response.text)
            return self._normalize_parsed_data(parsed)
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
                    temperature=settings.GEMINI_TEMPERATURE,
                    max_output_tokens=settings.GEMINI_MAX_TOKENS,
                ),
            ),
        )
        return response

    
    def _build_extraction_prompt(self, resume_text: str) -> str:
      """Build the extraction prompt sent to Gemini for resume parsing."""
      capped = resume_text[:15000]
      return f"""You are a resume data extractor for an enterprise ATS system. Return ONLY valid JSON. No markdown, no explanation, no text before or after the JSON object. If extraction is partial, still return valid JSON with nulls and empty arrays.

          Extract information from this resume and return this exact JSON structure:

          {{
            "personal_info": {{
              "name": "Full name or null",
              "email": "Email or null",
              "phone": "Phone number or null",
              "location": "City, State, Country or null",
              "linkedin": "LinkedIn URL or null",
              "github": "GitHub URL or null",
              "website": "Personal website URL or null"
            }},
            "summary": {{
              "raw": "Verbatim professional summary from resume, or null",
              "years_experience": null,
              "domains": []
            }},
            "skills": {{
              "technical_skills": [
                {{
                  "name": "Canonical skill name (e.g. React, Node.js, AWS, Selenium, CNC Operation)",
                  "raw": "Exact phrase from resume",
                  "context": "Max 10-word phrase from resume proving usage, or null",
                  "type": "TECHNICAL|TOOL|DOMAIN|LANGUAGE|CERTIFICATION",
                  "group": "Programming|Frontend|Backend|Database|Cloud|DevOps|Data|Testing|Design|Other",
                  "source_sections": ["skills", "experience", "projects", "summary"],
                  "explicit_level": null,
                  "explicit_years_experience": null
                }}
              ],
              "soft_skills": [
                {{
                  "name": "Soft skill name",
                  "raw": "Exact phrase from resume - required",
                  "context": "Max 10-word phrase from resume, or null"
                }}
              ]
            }},
            "experience": [
              {{
                "company": "Company name",
                "position": "Most recent or primary job title at this company",
                "location": "City, Country or null",
                "industry": "e.g. IT, Healthcare, Finance, Manufacturing",
                "start_date": "YYYY-MM-DD or null",
                "end_date": "YYYY-MM-DD or null",
                "is_current": false,
                "description": "Max 25-word role summary",
                "achievements": ["Max 15 words each - max 4 items"],
                "skills_used": ["skill name only - must match a name in skills.technical_skills"]
              }}
            ],
            "education": [
              {{
                "institution": "University or school name",
                "degree": "Degree type or null - do not infer",
                "field": "Field of study or null - do not infer",
                "start_date": "YYYY-MM-DD or null",
                "end_date": "YYYY-MM-DD or null",
                "grade": "GPA or percentage or null"
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
                "description": "Max 20-word project summary",
                "technologies": ["string"],
                "role": "Role or contribution or null",
                "start_date": "YYYY-MM-DD or null",
                "end_date": "YYYY-MM-DD or null"
              }}
            ],
            "languages": [
              {{
                "name": "Language name",
                "proficiency": "Basic|Intermediate|Fluent|Native"
              }}
            ],
            "career_gaps": [],
            "resume_language": "ISO 639-1 code, e.g. en, hi, fr",
            "total_experience_months": null
          }}

          --- RULES: SKILLS ---

          1. Max 40 skills total in technical_skills.
          2. INCLUDE: programming languages, frameworks, libraries, cloud platforms, databases, DevOps tools, design tools, domain skills, trade skills (e.g. CNC Operation, Welding, Circuit Assembly).
          3. EXCLUDE: generic verbs (managing, handling, responsible for, leading, coordinating), vague traits (hardworking, team player, fast learner, detail oriented, self motivated), overly broad single words (Testing, Management, Development, Analysis) unless qualified (e.g. Regression Testing is fine).
          4. CANONICAL naming - use these exact forms: React (not React.js or ReactJS), Node.js (not NodeJS), AWS (not Amazon Web Services), PostgreSQL (not Postgres), JavaScript (not JS), TypeScript (not TS). Always use the widely accepted industry short form.
          5. DEDUPLICATION: if two skills map to the same canonical name, keep only one. Prefer the entry with a non-null context.
          6. context: copy a short phrase directly from the resume proving the skill was used. Max 10 words. null if no evidence phrase available.
          7. explicit_level and explicit_years_experience: only populate if the resume explicitly states this (e.g. "5+ years of Java", "Advanced Excel"). Otherwise null. Never guess.
          8. soft_skills: only include if an exact phrase is present in the resume text. Do not infer from job titles or responsibilities. Return empty array if none found.

          --- RULES: EXPERIENCE ---

          9. Sort all experience entries in reverse chronological order (most recent first).
          10. SAME-COMPANY MERGING: if a candidate held multiple roles at the same company, merge into ONE entry. Use the highest or most recent title as position. Use earliest start_date and latest end_date. Mention role progression in description.
          11. description: max 25 words. Factual only. No storytelling.
          12. achievements: max 4 per role. Each achievement max 15 words. Quantified where possible. Do not repeat information already in description.
          13. skills_used: string array of skill names only. Every name here MUST exactly match a name in skills.technical_skills[].name. Do not include skills not present in the skills section.
          14. No duplicate entries for the same company.

          --- RULES: SUMMARY ---

          15. summary.raw: copy verbatim from resume if a summary, profile, or objective section exists. null if not present.
          16. summary.years_experience: only populate if the resume explicitly states a number of years (e.g. "8+ years of experience"). Otherwise null. Do not calculate.
          17. summary.domains: max 3 short domain labels. Only include if explicit industry or domain signals exist in the resume (e.g. company industry, job titles, certifications). Examples: E-commerce, Healthcare IT, Embedded Systems, FinTech. Empty array if signals are absent or ambiguous.

          --- RULES: EDUCATION AND CERTIFICATIONS ---

          18. degree and field: do not infer or guess. If not explicitly stated in the resume, set null.
          19. Certifications: only extract if an explicit certification name or issuing body is present in the resume.

          --- RULES: DATES ---

          20. Use YYYY-MM-DD when full date is available.
          21. Use YYYY-MM-01 when only month and year are known.
          22. Use YYYY-01-01 when only year is known.
          23. Use null when date is unknown or not mentioned.

          --- RULES: OUTPUT ---

          24. Never hallucinate companies, skills, dates, technologies, or any detail not present in the resume text.
          25. Do not repeat identical information across multiple fields.
          26. Always return valid parseable JSON even if most fields are null or empty arrays.
          27. No trailing commas. No comments inside the JSON.
          28. career_gaps: return empty array always - this is calculated by the backend.
          29. total_experience_months: return null always - this is calculated by the backend.

          Resume:
          {capped}

          JSON:"""

    def _parse_gemini_response(self, response_text: str) -> ParsedResumeData:
        """Parse Gemini's JSON response into structured data."""
        try:
            if not response_text:
                raise ValueError("Gemini returned empty response")

            # Try direct parse first (response should be pure JSON per prompt)
            obj = None
            stripped = response_text.strip()
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                # Fall back to regex extraction (handles markdown code fences etc.)
                json_match = re.search(r"\{.*\}", stripped, re.DOTALL)
                if not json_match:
                    print(f"[Gemini parse error] Response length: {len(stripped)}, "
                          f"first 200 chars: {stripped[:200]!r}, "
                          f"last 200 chars: {stripped[-200:]!r}")
                    raise ValueError("No JSON found in Gemini response")
                obj = json.loads(json_match.group())

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

            # Parse experience — skills_used must be strings
            experience_raw = obj.get("experience", []) or []
            experience = []
            for e in experience_raw:
                if not isinstance(e, dict) or not e.get("company"):
                    continue
                skills_used = e.get("skills_used", [])
                normalized = []
                for skill in skills_used:
                    if isinstance(skill, str):
                        normalized.append(skill)
                    elif isinstance(skill, dict):
                        normalized.append(skill.get("name", ""))
                e["skills_used"] = [s for s in normalized if s]
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

            # Parse summary (now an object)
            summary_raw = obj.get("summary")
            if isinstance(summary_raw, dict):
                summary = SummaryObject(**{k: v for k, v in summary_raw.items() if k in SummaryObject.model_fields})
            elif isinstance(summary_raw, str):
                summary = SummaryObject(raw=summary_raw)
            else:
                summary = None

            # Parse career_gaps
            career_gaps_raw = obj.get("career_gaps", []) or []
            career_gaps = [
                CareerGap(**cg) for cg in career_gaps_raw
                if isinstance(cg, dict) and cg.get("start_date") and cg.get("end_date")
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
                summary=summary,
                career_gaps=career_gaps,
                resume_language=obj.get("resume_language") or "en",
                total_experience_months=obj.get("total_experience_months"),
                confidence_score=0.95,
            )

        except Exception as e:
            print(f"Parsing error: {e}")
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

    def _normalize_parsed_data(self, parsed: ParsedResumeData) -> ParsedResumeData:
        """Post-process LLM output to enforce contract rules."""

        # 1. Ensure experience skills_used are strings, not objects
        if parsed.experience:
            for exp in parsed.experience:
                if exp.skills_used and isinstance(exp.skills_used[0], dict):
                    exp.skills_used = [
                        s.get("name", "") for s in exp.skills_used if isinstance(s, dict)
                    ]

        # 2. Ensure summary is SummaryObject, not string (handles old prompt output)
        if parsed.summary and isinstance(parsed.summary, str):
            parsed.summary = SummaryObject(raw=parsed.summary)

        # 3. Cap skills at 40
        if parsed.skills and parsed.skills.technical_skills:
            parsed.skills.technical_skills = parsed.skills.technical_skills[:40]

        # 4. Default resume_language
        if not parsed.resume_language:
            parsed.resume_language = "en"

        # 5. Ensure career_gaps is a list
        if parsed.career_gaps is None:
            parsed.career_gaps = []

        # 6. Backfill context_snippets from context for NestJS compat
        if parsed.skills and parsed.skills.technical_skills:
            for skill in parsed.skills.technical_skills:
                if skill.context and not skill.context_snippets:
                    skill.context_snippets = [skill.context]

        return parsed
