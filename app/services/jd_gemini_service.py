import re
import asyncio
from typing import Optional
import google.genai as genai
from google.genai import types

from app.config import settings
from app.models.schemas import (
    ParsedJdData, ParsedJdRole, ParsedJdSkill,
    JDSalaryRange, JDLocation, JDEducation, JDEmploymentType,
)
from app.utils.json_utils import safe_parse_json

VALID_IMPORTANCE = {"CRITICAL", "REQUIRED", "PREFERRED", "BONUS"}


def _to_int(val) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


class JdGeminiService:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model = settings.GEMINI_MODEL
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        self.parse_models = [
            settings.GEMINI_PARSE_MODEL_1,
            settings.GEMINI_PARSE_MODEL_2,
            settings.GEMINI_PARSE_MODEL_3,
        ]

    def is_available(self) -> bool:
        return self.client is not None

    async def parse_jd_text(self, text: str) -> ParsedJdData:
        if not self.client:
            raise RuntimeError("Gemini API key not configured")

        prompt, input_truncated = self._build_jd_extraction_prompt(text)
        last_error: Exception = RuntimeError("No models attempted")

        for attempt, model in enumerate(self.parse_models, start=1):
            try:
                print(f"[JdGemini] Attempt {attempt}/{len(self.parse_models)} using model: {model}")
                response = await self._generate_content_async(prompt, model=model)
                print(f"[JdGemini raw output]\n{response.text}\n[/JdGemini raw output]")
                parsed = self._parse_jd_response(response.text)
                parsed.input_truncated = input_truncated
                parsed.schema_version = "2.1"
                return self._normalize_jd_data(parsed)
            except Exception as e:
                print(f"[JdGemini] Attempt {attempt} failed ({model}): {e}")
                last_error = e

        raise RuntimeError(f"All Gemini models failed for JD parsing. Last error: {last_error}")

    async def _generate_content_async(self, prompt: str, model: Optional[str] = None):
        resolved_model = model or self.model
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=resolved_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=settings.GEMINI_TEMPERATURE,
                    max_output_tokens=settings.GEMINI_MAX_TOKENS,
                    response_mime_type="application/json",
                ),
            ),
        )
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason
            if finish_reason and str(finish_reason) not in ("FinishReason.STOP", "STOP", "1"):
                raise RuntimeError(
                    f"Gemini response truncated (finish_reason={finish_reason}). "
                    "Increase GEMINI_MAX_TOKENS or reduce input size."
                )
        return response

    def _build_jd_extraction_prompt(self, text: str) -> tuple[str, bool]:
        """Returns (prompt_string, input_truncated_flag)."""
        input_truncated = len(text) > 10000
        capped = text[:10000]

        prompt = f"""You are a job description data extractor for an enterprise ATS system.
Return ONLY valid JSON. No markdown, no explanation, no text before or after the JSON object.
No trailing commas. No comments inside JSON.

Extract information from this job description and return this exact JSON structure:

{{
  "role": {{
    "raw_title": "exact job title from JD",
    "normalized": "lowercase_with_underscores",
    "confidence": 0.0
  }},
  "employment_type": {{
    "value": null,
    "inferred": false
  }},
  "seniority_level": null,
  "skills": [
    {{
      "name": "Python",
      "skill_type": "LANGUAGE",
      "importance": "REQUIRED",
      "years_required": 5,
      "context": "5+ years of Python development"
    }},
    {{
      "name": "AWS",
      "skill_type": "TOOL",
      "importance": "REQUIRED",
      "years_required": null,
      "context": "experience with AWS services required"
    }},
    {{
      "name": "fintech domain knowledge",
      "skill_type": "DOMAIN",
      "importance": "PREFERRED",
      "years_required": null,
      "context": "fintech or payments experience preferred"
    }},
    {{
      "name": "AWS Certified Solutions Architect",
      "skill_type": "CERTIFICATION",
      "importance": "BONUS",
      "years_required": null,
      "context": "AWS certification is a bonus"
    }}
  ],
  "soft_skills": [
    {{
      "name": "ownership",
      "skill_type": "SOFT",
      "importance": "REQUIRED",
      "context": "strong sense of ownership expected"
    }}
  ],
  "location": {{
    "city": null,
    "country": null,
    "remote": false,
    "hybrid": false
  }},
  "experience_range": {{"min": null, "max": null}},
  "salary_range": {{
    "min": null,
    "max": null,
    "currency": null,
    "period": null,
    "equity_mentioned": false
  }},
  "education": {{
    "degree_required": null,
    "field": null,
    "required": null,
    "explicitly_not_required": false
  }},
  "department": null,
  "key_responsibilities": [],
  "key_requirements": [],
  "language": "en",
  "confidence_score": 0.0
}}

--- EXTRACTION RULES ---

ROLE
1. raw_title: extract the primary job title exactly as written in the JD.
2. normalized: lowercase, replace spaces/hyphens with underscores, strip special characters.
3. confidence: 0.0-1.0 reflecting certainty about the extracted title.

EMPLOYMENT TYPE
4. employment_type.value: one of FULL_TIME | PART_TIME | CONTRACT | INTERNSHIP | FREELANCE.
   employment_type.inferred: false if explicitly stated in JD.
   If not mentioned anywhere, set value: "FULL_TIME", inferred: true.
   Do NOT set employment_type to null -- always populate with a value + inferred flag.
   Signals: "full-time", "permanent role" -> FULL_TIME. "6-month contract", "fixed term" -> CONTRACT.
   "part-time", "20hrs/week" -> PART_TIME. "internship", "intern" -> INTERNSHIP.

SENIORITY LEVEL
5. seniority_level: one of INTERN | JUNIOR | MID | SENIOR | STAFF | PRINCIPAL | DIRECTOR | VP | C_LEVEL.
   Infer from title AND description. "Lead Engineer" -> SENIOR. "VP of Engineering" -> VP.
   null if genuinely ambiguous.

SKILLS — CLASSIFICATION RULES
6. Every skill must have a `skill_type`. Use EXACTLY one of these six values:

   LANGUAGE     — programming, query, or markup languages.
                  Examples: Python, JavaScript, TypeScript, SQL, Go, Rust, GraphQL, HTML, Bash.
                  Do NOT use for spoken/natural languages (English, Hindi) — those go in the
                  `language` field. Rule of thumb: if you write it in a code editor, it is LANGUAGE.

   TOOL         — named tools, platforms, services, or products that a person uses as-is.
                  Examples: React, Node.js, Docker, Kubernetes, AWS, GCP, Jira, Figma,
                  PostgreSQL, Redis, Kafka, GitHub Actions, Datadog, Postman, Terraform.
                  Rule of thumb: if it has a brand name / product page, it is TOOL.

   TECHNICAL    — engineering concepts, patterns, methodologies, and paradigms that are not
                  a named tool or language.
                  Examples: microservices architecture, REST API design, system design,
                  CI/CD practices, test-driven development, distributed systems, OOP.
                  Rule of thumb: if it describes HOW to build, not WHAT you use, it is TECHNICAL.

   DOMAIN       — industry, vertical, or business domain knowledge.
                  Examples: fintech, payments, healthcare, e-commerce, B2B SaaS, logistics,
                  lending, capital markets, EdTech.
                  Rule of thumb: it describes WHERE the work happens, not how or with what.

   CERTIFICATION — a named credential, licence, or certification.
                  Examples: AWS Certified Solutions Architect, PMP, CFA, Google Cloud
                  Professional, Kubernetes CKAD, CISSP.
                  Rule of thumb: it has a specific issuing body and a formal name.

   SOFT         — interpersonal, behavioural, and cognitive traits.
                  Examples: ownership, mentoring, proactive, stakeholder communication,
                  collaborative, attention to detail, cross-functional.
                  Soft skills go in the `soft_skills` array, not the `skills` array.
                  Rule of thumb: it describes HOW a person works with people, not technology.

BOUNDARY CASES (apply these when ambiguous):
   - "React" → TOOL (named library). "Component-based architecture" → TECHNICAL.
   - "PostgreSQL" → TOOL. "Database design" → TECHNICAL.
   - "Python" → LANGUAGE. "Scripting experience" → TECHNICAL.
   - "AWS" → TOOL. "Cloud architecture" → TECHNICAL.
   - "Agile" → TECHNICAL (methodology). "Jira" → TOOL (the software).
   - "Fintech experience" → DOMAIN. "Payment gateway integration" → TECHNICAL.
   - "Communication" → SOFT. "Technical writing" → TECHNICAL.

`skills` array: max 25 entries. No soft skills in this array.

7. DEDUPLICATION -- Before finalising the skills array, check for grouping consistency:
   - If multiple skills appear in the same context sentence (e.g. "React, Redux, TypeScript"),
     they must all share the same importance level -- do not assign one CRITICAL and siblings REQUIRED
     from the same sentence unless the sentence explicitly calls out one skill specifically.
   - Do not list the same skill twice under different names (e.g. "ReactJS" and "React.js" are
     the same skill -- use canonical name "React").
   - If a set of skills are alternatives (e.g. "Tailwind CSS or Styled Components"), still list
     each separately but give them the same importance.

8. importance rules -- use EXACTLY these criteria, no overlap:
   - CRITICAL: phrases "must have", "required", "essential", "mandatory", "must be proficient"
   - REQUIRED: listed in requirements section without a qualifier
   - PREFERRED: "nice to have", "preferred", "plus", "would be great", "desirable"
   - BONUS: "bonus", "bonus points", "big plus", "great to have"
   "desirable" -> PREFERRED only. "bonus" -> BONUS only. Never mix.

9. years_required: integer if explicitly stated per-skill (e.g. "7+ years of React" -> 7).
   Read each skill's surrounding sentence, not just the role-level experience range.
   null if no per-skill years are stated.

10. context: copy a short phrase directly from the JD for this specific skill. Max 15 words.
    null if no specific evidence phrase.

SKILLS -- SOFT
11. `soft_skills` array: max 10 entries. Only interpersonal, behavioural, and cognitive traits.
    signal phrases to watch for: "collaborative", "ownership", "mentoring", "proactive",
    "stakeholder communication", "cross-functional", "self-starter", "empathy", "customer-focused",
    "attention to detail", "problem-solving mindset".
    skill_type must always be "SOFT". No years_required field for soft skills.
    importance: REQUIRED if in requirements section, PREFERRED otherwise. No CRITICAL or BONUS.
    Empty array if no soft skill signals found -- do not invent soft skills.

LOCATION
10. Populate city, country, remote, hybrid independently.
    "Bangalore or Remote" -> city: "Bangalore", remote: true.
    "Hybrid -- New York" -> city: "New York", hybrid: true.
    "Remote-first" -> remote: true, city: null.
    All fields null/false if no location info present.

EXPERIENCE
12. experience_range: parse carefully:
    - "7+ years" -> min: 7, max: null   (open-ended -- do NOT set max to 7)
    - "5-8 years" -> min: 5, max: 8
    - "at least 5 years" -> min: 5, max: null
    - "up to 3 years" -> min: null, max: 3
    - "3 years" (exact, no range) -> min: 3, max: 3
    Both null if no experience range mentioned.

SALARY
12. salary_range: extract min/max as integers (no currency symbols, no commas).
    currency: ISO 4217 code ("INR", "USD", "GBP"). Infer from country/context if not explicit.
    period: "ANNUAL" | "MONTHLY" | "HOURLY". Infer from context.
    equity_mentioned: true if stock, ESOP, equity, options, RSU is mentioned anywhere.
    All null if salary not mentioned.

EDUCATION
13. degree_required: "BACHELOR" | "MASTER" | "PHD" | "ANY". null if not stated.
    field: field of study. null if not stated.
    required: true = mandatory, false = preferred/nice-to-have. null if unclear.
    explicitly_not_required: true ONLY when JD contains explicit phrases like
    "no degree required", "degree not required", "equivalent experience accepted",
    "we don't require a formal degree". false in all other cases (including when not mentioned).

DEPARTMENT
14. Infer from context: Engineering, Product, Sales, Marketing, Finance, Operations,
    Design, Data, Legal, HR, Customer Success, Security. null if cannot determine.

KEY RESPONSIBILITIES vs REQUIREMENTS
15. key_responsibilities: max 5 strings, each <= 20 words.
    These are things the person WILL DO in the role.
    Examples: "Lead the frontend architecture", "Collaborate with product managers",
    "Build reusable component libraries", "Conduct code reviews".
    Copy directly from JD where possible. Empty array if section absent.

16. key_requirements: max 5 strings, each <= 20 words.
    These are things the candidate MUST HAVE or KNOW -- qualifications, not activities.
    Examples: "5+ years of React experience", "Strong knowledge of REST APIs",
    "Experience with CI/CD pipelines", "Proven track record in fintech".
    Do NOT copy the same strings into both arrays. If ambiguous, prefer key_requirements.
    Empty array if section absent.

LANGUAGE
16. language: ISO 639-1 code of the JD's written language. Default "en" for English.
    "hi" for Hindi, "de" for German, etc. Always populate this field.

GENERAL
17. confidence_score: 0.0-1.0 overall confidence in the full extraction quality.
18. Never hallucinate. Use null for unknown fields. Return valid JSON only.

Job Description:
{capped}

JSON:"""

        return prompt, input_truncated

    def _merge_soft_skills(self, result: dict) -> dict:
        """Merge LLM's separate soft_skills array into skills. NestJS uses skill_type to distinguish."""
        soft_skills = result.pop("soft_skills", []) or []
        for s in soft_skills:
            if s.get("skill_type") not in ("SOFT",):
                s["skill_type"] = "SOFT"   # enforce — soft_skills array must always be SOFT
            s.setdefault("years_required", None)
        result.setdefault("skills", [])
        result["skills"].extend(soft_skills)
        return result

    def _normalise_skill_types(self, skills: list[dict]) -> list[dict]:
        """
        Coerce any skill_type value that doesn't match the taxonomy enum to a valid value.
        Handles LLM hallucinating values like "framework", "library", "database", "cloud".
        """
        valid_types = {"TECHNICAL", "SOFT", "DOMAIN", "TOOL", "CERTIFICATION", "LANGUAGE"}

        drift_map = {
            "framework":     "TOOL",
            "library":       "TOOL",
            "database":      "TOOL",
            "cloud":         "TOOL",
            "platform":      "TOOL",
            "service":       "TOOL",
            "language":      "LANGUAGE",
            "programming":   "LANGUAGE",
            "methodology":   "TECHNICAL",
            "practice":      "TECHNICAL",
            "concept":       "TECHNICAL",
            "skill":         "TECHNICAL",
            "certificate":   "CERTIFICATION",
            "cert":          "CERTIFICATION",
            "interpersonal": "SOFT",
            "behavioral":    "SOFT",
            "behavioural":   "SOFT",
            "industry":      "DOMAIN",
            "domain":        "DOMAIN",
            "vertical":      "DOMAIN",
        }

        for skill in skills:
            raw = (skill.get("skill_type") or "TECHNICAL").strip()
            if raw in valid_types:
                continue
            normalised = drift_map.get(raw.lower())
            skill["skill_type"] = normalised if normalised else "TECHNICAL"

        return skills

    def _deduplicate_skills(self, skills: list[dict]) -> list[dict]:
        """Remove duplicate skill entries by normalised name, keeping highest importance rank."""
        importance_rank = {"CRITICAL": 4, "REQUIRED": 3, "PREFERRED": 2, "BONUS": 1}
        seen: dict[str, dict] = {}
        for skill in skills:
            key = skill.get("name", "").lower().strip()
            if not key:
                continue
            if key not in seen:
                seen[key] = skill
            else:
                existing_rank = importance_rank.get(seen[key].get("importance", ""), 0)
                new_rank = importance_rank.get(skill.get("importance", ""), 0)
                if new_rank > existing_rank:
                    seen[key] = skill
        return list(seen.values())

    def _normalise_employment_type(self, result: dict) -> dict:
        """Coerce employment_type null or plain string into {value, inferred} shape."""
        et = result.get("employment_type")
        if et is None:
            result["employment_type"] = {"value": "FULL_TIME", "inferred": True}
        elif isinstance(et, str):
            result["employment_type"] = {"value": et, "inferred": False}
        elif isinstance(et, dict):
            if et.get("value") is None:
                et["value"] = "FULL_TIME"
                et["inferred"] = True
        return result

    def _fix_experience_range(self, result: dict) -> dict:
        """Structural guard for open-ended ranges; main fix is in the prompt."""
        er = result.get("experience_range")
        if er and er.get("min") is not None and er.get("max") is not None:
            if er["min"] == er["max"]:
                pass  # ambiguous — leave as-is; prompt handles new extractions
        return result

    def _normalise_skill_importance(self, skills: list[dict]) -> list[dict]:
        """Enforce strict PREFERRED vs BONUS separation on raw skill dicts from LLM output."""
        bonus_phrases = {"bonus", "bonus points", "big plus", "great to have"}
        for skill in skills:
            if skill.get("importance") == "BONUS":
                ctx = (skill.get("context") or "").lower()
                if not any(p in ctx for p in bonus_phrases):
                    skill["importance"] = "PREFERRED"
        return skills

    def _parse_jd_response(self, response_text: str) -> ParsedJdData:
        try:
            if not response_text:
                raise ValueError("Gemini returned empty response")

            obj = safe_parse_json(response_text)

            obj = self._merge_soft_skills(obj)
            obj["skills"] = self._normalise_skill_types(obj.get("skills", []))
            obj["skills"] = self._deduplicate_skills(obj.get("skills", []))
            obj["skills"] = self._normalise_skill_importance(obj.get("skills", []))
            obj = self._normalise_employment_type(obj)
            obj = self._fix_experience_range(obj)

            role_raw = obj.get("role")
            role = None
            if isinstance(role_raw, dict) and role_raw.get("raw_title"):
                role = ParsedJdRole(
                    raw_title=role_raw.get("raw_title", ""),
                    normalized=role_raw.get("normalized", ""),
                    confidence=float(role_raw.get("confidence", 0.9)),
                )

            skills = []
            for s in (obj.get("skills", []) or []):
                if not isinstance(s, dict) or not s.get("name"):
                    continue
                years = s.get("years_required")
                if years is not None:
                    try:
                        years = int(years)
                    except (TypeError, ValueError):
                        years = None
                skills.append(ParsedJdSkill(
                    name=s["name"],
                    importance=s.get("importance", "REQUIRED"),
                    skill_type=s.get("skill_type", "TECHNICAL"),
                    years_required=years,
                    context=s.get("context"),
                ))

            exp_range = obj.get("experience_range")
            if isinstance(exp_range, dict):
                min_val = exp_range.get("min")
                max_val = exp_range.get("max")
                exp_range = {"min": min_val, "max": max_val} if (min_val is not None or max_val is not None) else None
            else:
                exp_range = None

            salary_raw = obj.get("salary_range")
            salary_range = None
            if isinstance(salary_raw, dict):
                min_s = _to_int(salary_raw.get("min"))
                max_s = _to_int(salary_raw.get("max"))
                if min_s is not None or max_s is not None or salary_raw.get("currency"):
                    salary_range = JDSalaryRange(
                        min=min_s,
                        max=max_s,
                        currency=salary_raw.get("currency"),
                        period=salary_raw.get("period"),
                        equity_mentioned=bool(salary_raw.get("equity_mentioned", False)),
                    )

            loc_raw = obj.get("location")
            location_str = None
            location_structured = None
            if isinstance(loc_raw, dict):
                location_structured = JDLocation(
                    city=loc_raw.get("city"),
                    country=loc_raw.get("country"),
                    remote=bool(loc_raw.get("remote", False)),
                    hybrid=bool(loc_raw.get("hybrid", False)),
                )
            elif isinstance(loc_raw, str):
                location_str = loc_raw

            edu_raw = obj.get("education")
            education = None
            if isinstance(edu_raw, dict):
                education = JDEducation(
                    degree_required=edu_raw.get("degree_required"),
                    field=edu_raw.get("field"),
                    required=edu_raw.get("required"),
                    explicitly_not_required=bool(edu_raw.get("explicitly_not_required", False)),
                )

            et_raw = obj.get("employment_type")
            employment_type = None
            if isinstance(et_raw, dict) and et_raw.get("value"):
                employment_type = JDEmploymentType(
                    value=et_raw["value"],
                    inferred=bool(et_raw.get("inferred", False)),
                )

            key_resp = obj.get("key_responsibilities") or []
            key_resp = [r for r in key_resp if isinstance(r, str)][:5]

            key_reqs = obj.get("key_requirements") or []
            key_reqs = [r for r in key_reqs if isinstance(r, str)][:5]

            return ParsedJdData(
                role=role,
                skills=skills,
                location=location_str,
                location_structured=location_structured,
                experience_range=exp_range,
                salary_range=salary_range,
                education=education,
                employment_type=employment_type,
                seniority_level=obj.get("seniority_level"),
                key_responsibilities=key_resp,
                key_requirements=key_reqs,
                language=obj.get("language", "en"),
                department=obj.get("department"),
                confidence_score=float(obj.get("confidence_score", 0.9)),
            )

        except Exception as e:
            raise ValueError(f"Failed to parse Gemini JD response: {e}")

    def _normalize_jd_data(self, parsed: ParsedJdData) -> ParsedJdData:
        # Cap skills at 25
        parsed.skills = parsed.skills[:25]

        # Validate importance values
        for skill in parsed.skills:
            if skill.importance not in VALID_IMPORTANCE:
                skill.importance = "REQUIRED"

        # Normalize role.normalized
        if parsed.role and parsed.role.normalized:
            normalized = parsed.role.normalized.lower()
            normalized = re.sub(r"[^a-z0-9_]", "_", normalized)
            normalized = re.sub(r"_+", "_", normalized).strip("_")
            parsed.role.normalized = normalized

        # Default confidence
        if not parsed.confidence_score:
            parsed.confidence_score = 0.9

        return parsed
