import json
import re
import asyncio
from typing import Optional
import google.genai as genai
from google.genai import types

from app.config import settings

VALID_SKILL_TYPES = {"TECHNICAL", "SOFT", "DOMAIN", "TOOL", "CERTIFICATION", "LANGUAGE"}


class SkillMappingService:
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

    async def map_skills_to_taxonomy(
        self,
        non_taxonomy_skills: list[dict],
        taxonomy_context: dict,
    ) -> list[dict]:
        """
        Map a batch of non-taxonomy skills to existing taxonomy entries using LLM.
        Each skill gets one of three actions:
          SYNONYM   — variant/abbreviation of an existing taxonomy skill
          NEW_SKILL — genuinely new skill worth adding to taxonomy
          NO_MATCH  — too vague / junk / cannot be confidently classified
        """
        if not self.client:
            raise RuntimeError("Gemini API key not configured")

        prompt = self._build_mapping_prompt(non_taxonomy_skills, taxonomy_context)
        last_error: Exception = RuntimeError("No models attempted")

        for attempt, model in enumerate(self.parse_models, start=1):
            try:
                print(f"[SkillMapping] Attempt {attempt}/{len(self.parse_models)} using model: {model}")
                response = await self._generate_content_async(prompt, model=model)
                print(f"[SkillMapping raw output]\n{response.text[:600]}\n[/SkillMapping raw output]")
                return self._parse_mapping_response(response.text, non_taxonomy_skills)
            except Exception as e:
                print(f"[SkillMapping] Attempt {attempt} failed ({model}): {e}")
                last_error = e

        raise RuntimeError(f"All Gemini models failed for skill mapping. Last error: {last_error}")

    async def _generate_content_async(self, prompt: str, model: Optional[str] = None):
        resolved_model = model or self.model
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=resolved_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                ),
            ),
        )
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason
            if finish_reason and str(finish_reason) not in ("FinishReason.STOP", "STOP", "1"):
                raise RuntimeError(
                    f"Gemini response truncated (finish_reason={finish_reason}). "
                    "Reduce batch size or increase max_output_tokens."
                )
        return response

    def _build_mapping_prompt(self, skills: list[dict], context: dict) -> str:
        # Compact category overview
        category_lines = []
        for cat in context.get("categories", []):
            cat_name = cat.get("name", "")
            subcats = ", ".join(cat.get("subcategories", [])[:10])
            samples = ", ".join(cat.get("sample_skills", [])[:8])
            line = f"  {cat_name}"
            if subcats:
                line += f" | subcats: {subcats}"
            if samples:
                line += f" | examples: {samples}"
            category_lines.append(line)

        taxonomy_overview = "\n".join(category_lines)

        # Compact taxonomy skill list: id, name, category, subcategory only
        taxonomy_skills = context.get("taxonomy_skills", [])
        skills_compact = [
            {"id": s["id"], "n": s["name"], "c": s.get("category"), "sc": s.get("subcategory")}
            for s in taxonomy_skills
        ]
        taxonomy_skills_json = json.dumps(skills_compact)

        # Input skills
        input_skills = [
            {
                "id": s["id"],
                "name": s["skillName"],
                "norm": s["normalizedName"],
                "type": s.get("skillType", "TECHNICAL"),
                "freq": s.get("frequency", 1),
                "ctx": (s.get("contextSnippets") or [])[:2],
            }
            for s in skills
        ]
        input_json = json.dumps(input_skills, indent=2)

        return f"""You are a skill taxonomy expert for an ATS system. Classify each unrecognised skill into one of three actions.

TAXONOMY CATEGORIES (name | subcategories | example skills):
{taxonomy_overview}

TAXONOMY SKILLS (id, n=name, c=category, sc=subcategory):
{taxonomy_skills_json}

INPUT SKILLS TO CLASSIFY:
{input_json}

TASK: For each input skill decide the correct action and return a JSON array (one object per skill, same order as input):

[
  {{
    "id": <input skill id>,
    "action": "SYNONYM" | "NEW_SKILL" | "NO_MATCH",

    // SYNONYM fields (null for other actions)
    "taxonomy_id": <int — must match an id from TAXONOMY SKILLS, or null>,
    "taxonomy_name": <string — matched skill name, or null>,

    // NEW_SKILL fields (null for other actions)
    "canonical_name": <string — clean canonical skill name, e.g. "React Native", or null>,
    "normalized_name": <string — snake_case lowercase, e.g. "react_native", or null>,
    "category": <string — must match one of the TAXONOMY CATEGORIES above, or null>,
    "subcategory": <string — must match a subcategory of the chosen category, or null>,
    "skill_type": <"TECHNICAL"|"SOFT"|"DOMAIN"|"TOOL"|"CERTIFICATION"|"LANGUAGE", or null>,

    "confidence": <float 0.0–1.0>,
    "reason": "<one short phrase>"
  }}
]

RULES — SYNONYM:
- Use when the input is clearly a variant, abbreviation, or alias of an existing taxonomy skill
  (e.g. "JS"→"JavaScript", "ReactJS"→"React", "k8s"→"Kubernetes", "Postgres"→"PostgreSQL")
- confidence must be >= 0.85; do not guess at low confidence
- taxonomy_id must exactly match an id from the TAXONOMY SKILLS list

RULES — NEW_SKILL:
- Use when the skill is specific, real, and not yet represented in the taxonomy
  (a new framework, library, platform, domain skill, tool, or certification that deserves its own entry)
- Provide canonical_name (industry-standard form), normalized_name (snake_case), skill_type
- category MUST be chosen from the TAXONOMY CATEGORIES list above — do NOT invent a new category name
- subcategory MUST be chosen from the subcategories listed under the chosen category — do NOT invent a new subcategory
- If no suitable category exists, set category and subcategory to null
- Do NOT use NEW_SKILL for generic verbs, vague traits, or overly broad terms (e.g. "Management", "Testing")
- confidence must be >= 0.80

RULES — NO_MATCH:
- Use when: the input is too vague, a generic verb, a soft trait with no direct phrase, junk data, or a duplicate
- Examples: "Hardworking", "Team Player", "Management", "Development", "Responsible for X"

Return ONLY the JSON array. No markdown, no extra text."""

    def _parse_mapping_response(self, response_text: str, input_skills: list[dict]) -> list[dict]:
        try:
            stripped = response_text.strip()
            try:
                results = json.loads(stripped)
            except json.JSONDecodeError:
                match = re.search(r"\[.*\]", stripped, re.DOTALL)
                if not match:
                    raise ValueError(
                        f"No JSON array found. First 300 chars: {stripped[:300]!r}"
                    )
                results = json.loads(match.group())

            if not isinstance(results, list):
                raise ValueError("Expected a JSON array from LLM")

            input_id_set = {s["id"] for s in input_skills}
            validated: list[dict] = []

            for r in results:
                if not isinstance(r, dict):
                    continue
                skill_id = r.get("id")
                if skill_id not in input_id_set:
                    continue

                action = r.get("action", "NO_MATCH")
                if action not in ("SYNONYM", "NEW_SKILL", "NO_MATCH"):
                    action = "NO_MATCH"

                confidence = float(r.get("confidence", 0.0))
                entry: dict = {
                    "id": skill_id,
                    "action": action,
                    "confidence": confidence,
                    "reason": str(r.get("reason", "")),
                    # SYNONYM fields
                    "taxonomy_id": None,
                    "taxonomy_name": None,
                    # NEW_SKILL fields
                    "canonical_name": None,
                    "normalized_name": None,
                    "category": None,
                    "subcategory": None,
                    "skill_type": None,
                }

                if action == "SYNONYM":
                    if confidence < 0.85 or not r.get("taxonomy_id"):
                        entry["action"] = "NO_MATCH"
                        entry["confidence"] = 0.0
                    else:
                        entry["taxonomy_id"] = r.get("taxonomy_id")
                        entry["taxonomy_name"] = r.get("taxonomy_name")

                elif action == "NEW_SKILL":
                    if confidence < 0.80 or not r.get("canonical_name"):
                        entry["action"] = "NO_MATCH"
                        entry["confidence"] = 0.0
                    else:
                        skill_type = r.get("skill_type", "TECHNICAL")
                        if skill_type not in VALID_SKILL_TYPES:
                            skill_type = "TECHNICAL"
                        canonical = str(r.get("canonical_name", "")).strip()
                        norm = str(r.get("normalized_name") or "").strip().lower().replace(" ", "_") or (
                            canonical.lower().replace(" ", "_").replace("-", "_")
                        )
                        entry["canonical_name"] = canonical
                        entry["normalized_name"] = norm
                        entry["category"] = r.get("category")
                        entry["subcategory"] = r.get("subcategory")
                        entry["skill_type"] = skill_type

                validated.append(entry)

            # Fill in NO_MATCH for any skills the LLM missed
            responded_ids = {r["id"] for r in validated}
            for s in input_skills:
                if s["id"] not in responded_ids:
                    validated.append({
                        "id": s["id"],
                        "action": "NO_MATCH",
                        "confidence": 0.0,
                        "reason": "not returned by LLM",
                        "taxonomy_id": None,
                        "taxonomy_name": None,
                        "canonical_name": None,
                        "normalized_name": None,
                        "category": None,
                        "subcategory": None,
                        "skill_type": None,
                    })

            return validated

        except Exception as e:
            raise ValueError(f"Failed to parse skill mapping response: {e}")

    # ── Categorize existing taxonomy skills ────────────────────────────────────

    async def categorize_skills(
        self,
        skills: list[dict],   # [{id, name, skillType, currentCategory}]
        category_context: list[dict],  # [{name, subcategories}]
    ) -> list[dict]:
        """Assign (or fix) category + subcategory for existing taxonomy skills."""
        if not self.client:
            raise RuntimeError("Gemini API key not configured")

        prompt = self._build_categorize_prompt(skills, category_context)
        last_error: Exception = RuntimeError("No models attempted")

        for attempt, model in enumerate(self.parse_models, start=1):
            try:
                print(f"[SkillCategorize] Attempt {attempt}/{len(self.parse_models)} using model: {model}")
                response = await self._generate_content_async(prompt, model=model)
                return self._parse_categorize_response(response.text, skills)
            except Exception as e:
                print(f"[SkillCategorize] Attempt {attempt} failed ({model}): {e}")
                last_error = e

        raise RuntimeError(f"All Gemini models failed for skill categorization. Last error: {last_error}")

    def _build_categorize_prompt(self, skills: list[dict], category_context: list[dict]) -> str:
        cat_lines = []
        for cat in category_context:
            subcats = ", ".join(cat.get("subcategories", []))
            cat_lines.append(f"  {cat['name']}: {subcats or '(no subcategories)'}")
        categories_str = "\n".join(cat_lines)

        skills_json = json.dumps([
            {"id": s["id"], "name": s["name"], "type": s.get("skillType", "TECHNICAL"), "current_cat": s.get("currentCategory")}
            for s in skills
        ], indent=2)

        return f"""You are a skill taxonomy expert. Assign each skill to the most appropriate category and subcategory from the list below.

VALID CATEGORIES AND THEIR SUBCATEGORIES (you MUST choose from this list only):
{categories_str}

SKILLS TO CATEGORIZE:
{skills_json}

Return a JSON array — one object per skill, same order as input:
[
  {{
    "id": <skill id>,
    "category": "<exact category name from the list above, or null if truly no fit>",
    "subcategory": "<exact subcategory name from the chosen category, or null if none fits>"
  }}
]

RULES:
- category and subcategory MUST be copied verbatim from the list — do NOT invent new names
- Choose the most specific and accurate category for each skill
- If no category fits at all, set both to null
- Return ONLY the JSON array. No markdown, no explanation."""

    def _parse_categorize_response(self, response_text: str, input_skills: list[dict]) -> list[dict]:
        try:
            stripped = response_text.strip()
            try:
                results = json.loads(stripped)
            except json.JSONDecodeError:
                match = re.search(r"\[.*\]", stripped, re.DOTALL)
                if not match:
                    raise ValueError(f"No JSON array found. First 200 chars: {stripped[:200]!r}")
                results = json.loads(match.group())

            if not isinstance(results, list):
                raise ValueError("Expected JSON array")

            input_id_set = {s["id"] for s in input_skills}
            validated = []
            for r in results:
                if not isinstance(r, dict) or r.get("id") not in input_id_set:
                    continue
                validated.append({
                    "id": r["id"],
                    "category": r.get("category"),
                    "subcategory": r.get("subcategory"),
                })

            # Fill nulls for any skills the LLM missed
            responded_ids = {r["id"] for r in validated}
            for s in input_skills:
                if s["id"] not in responded_ids:
                    validated.append({"id": s["id"], "category": None, "subcategory": None})

            return validated
        except Exception as e:
            raise ValueError(f"Failed to parse categorize response: {e}")
