from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.models.database_models import SkillTaxonomy, NonTaxonomySkill, ProfileSkill
from app.models.enums import ProficiencyLevel, SkillSource, SkillType
import json


class TaxonomyService:
    """Service to validate and process skills against taxonomy"""
    
    @staticmethod
    def normalize_skill_name(skill_name: str) -> str:
        """Normalize skill name for comparison"""
        return skill_name.lower().strip()
    
    @staticmethod
    def parse_proficiency_level(level: Optional[str]) -> ProficiencyLevel:
        """
        Convert string proficiency level to enum.
        Defaults to INTERMEDIATE if invalid or missing.
        """
        if not level:
            return ProficiencyLevel.INTERMEDIATE
        
        level_upper = str(level).upper().strip()
        try:
            return ProficiencyLevel[level_upper]
        except KeyError:
            return ProficiencyLevel.INTERMEDIATE
    
    @staticmethod
    async def find_in_taxonomy(db: AsyncSession, skill_name: str) -> Optional[SkillTaxonomy]:
        """Find skill in taxonomy (including synonyms) - ASYNC VERSION"""
        normalized = TaxonomyService.normalize_skill_name(skill_name)
        
        # Direct match
        result = await db.execute(
            select(SkillTaxonomy).filter(SkillTaxonomy.normalized_name == normalized)
        )
        taxonomy_skill = result.scalar_one_or_none()
        if taxonomy_skill:
            return taxonomy_skill
        
        # Search in synonyms
        result = await db.execute(
            select(SkillTaxonomy).filter(func.jsonb_exists(SkillTaxonomy.synonyms, skill_name))
        )
        taxonomy_skills = result.scalars().all()
        return taxonomy_skills[0] if taxonomy_skills else None
    
    @staticmethod
    async def validate_skills(db: AsyncSession, skill_names: List[str]) -> Dict[str, Any]:
        """Validate skills against taxonomy - ASYNC VERSION"""
        matched, unmatched = [], []
        
        for skill_name in skill_names:
            taxonomy_skill = await TaxonomyService.find_in_taxonomy(db, skill_name)
            if taxonomy_skill:
                matched.append({
                    "skill_name": taxonomy_skill.skill_name,
                    "taxonomy_id": taxonomy_skill.id,
                    "category": taxonomy_skill.parent_skill,
                    "normalized_name": taxonomy_skill.normalized_name,
                    "skill_level": taxonomy_skill.skill_level,
                    "skill_weight": taxonomy_skill.skill_weight,
                })
            else:
                unmatched.append(skill_name)
        
        return {"matched": matched, "unmatched": unmatched}
    
    @staticmethod
    async def process_resume_skills(
        db: AsyncSession,
        profile_id: int,
        resume_id: int,
        technical_skills: List[Dict[str, Any]],
        soft_skills: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process and store skills from parsed resume - ASYNC VERSION.
        Handles both taxonomy-matched and unmatched skills.
        """
        results = {
            "technical_matched": [],
            "technical_unmatched": [],
            "soft_skills": []
        }
        now = datetime.utcnow()

        # ------------------- TECHNICAL SKILLS -------------------
        for skill in technical_skills:
    # Handle both dict and string skill inputs
            if isinstance(skill, str):
                skill_name = skill
                proficiency = ProficiencyLevel.INTERMEDIATE
                years_experience = None
            else:
                skill_name = skill.get("name")
                proficiency = TaxonomyService.parse_proficiency_level(skill.get("level"))
                years_experience = skill.get("years_experience")
            
            if not skill_name:
                continue

                
            taxonomy_skill = await TaxonomyService.find_in_taxonomy(db, skill_name)
            proficiency = TaxonomyService.parse_proficiency_level(skill.get("level"))

            if taxonomy_skill:
                # ✅ Taxonomy matched skill
                result = await db.execute(
                    select(ProfileSkill).filter(
                        ProfileSkill.profile_id == profile_id,
                        ProfileSkill.skill_name == taxonomy_skill.skill_name
                    )
                )
                profile_skill = result.scalar_one_or_none()

                if profile_skill:
                    # Update existing
                    profile_skill.category = taxonomy_skill.parent_skill
                    profile_skill.proficiency_level = proficiency
                    profile_skill.years_experience = skill.get("years_experience")
                    profile_skill.verified = True
                    profile_skill.skill_type = SkillType.TECHNICAL
                    profile_skill.updated_at = now
                else:
                    # Create new record with timestamps
                    profile_skill = ProfileSkill(
                        profile_id=profile_id,
                        skill_name=taxonomy_skill.skill_name,
                        category=taxonomy_skill.parent_skill,
                        proficiency_level=proficiency,
                        years_experience=skill.get("years_experience"),
                        source=SkillSource.AI_EXTRACTED,
                        verified=True,
                        skill_type=SkillType.TECHNICAL,
                        created_at=now,
                        updated_at=now
                    )
                    db.add(profile_skill)
                
                results["technical_matched"].append({
                    "skill_name": taxonomy_skill.skill_name,
                    "category": taxonomy_skill.parent_skill,
                    "verified": True
                })
            else:
                # ❌ Unmatched skill — add to non-taxonomy
                result = await db.execute(
                    select(NonTaxonomySkill).filter(
                        NonTaxonomySkill.normalized_name == TaxonomyService.normalize_skill_name(skill_name),
                        NonTaxonomySkill.source == "RESUME",
                        NonTaxonomySkill.source_id == resume_id
                    )
                )
                non_tax_skill = result.scalar_one_or_none()

                if non_tax_skill:
                    non_tax_skill.frequency += 1
                    non_tax_skill.updated_at = now
                else:
                    non_tax_skill = NonTaxonomySkill(
                        skill_name=skill_name,
                        normalized_name=TaxonomyService.normalize_skill_name(skill_name),
                        source="RESUME",
                        source_id=resume_id,
                        frequency=1,
                        created_at=now,
                        updated_at=now
                    )
                    db.add(non_tax_skill)

                # Add unverified profile skill
                result = await db.execute(
                    select(ProfileSkill).filter(
                        ProfileSkill.profile_id == profile_id,
                        ProfileSkill.skill_name == skill_name
                    )
                )
                profile_skill = result.scalar_one_or_none()

                if profile_skill:
                    profile_skill.proficiency_level = proficiency
                    profile_skill.years_experience = skill.get("years_experience")
                    profile_skill.verified = False
                    profile_skill.skill_type = SkillType.TECHNICAL
                    profile_skill.updated_at = now
                else:
                    profile_skill = ProfileSkill(
                        profile_id=profile_id,
                        skill_name=skill_name,
                        proficiency_level=proficiency,
                        years_experience=skill.get("years_experience"),
                        source=SkillSource.AI_EXTRACTED,
                        verified=False,
                        skill_type=SkillType.TECHNICAL,
                        created_at=now,
                        updated_at=now
                    )
                    db.add(profile_skill)

                results["technical_unmatched"].append({
                    "skill_name": skill_name,
                    "verified": False
                })
        
        # ------------------- SOFT SKILLS -------------------
        for skill in soft_skills:
            if isinstance(skill, str):
                skill_name = skill
                proficiency = ProficiencyLevel.INTERMEDIATE
            else:
                skill_name = skill.get("name")
                proficiency = TaxonomyService.parse_proficiency_level(skill.get("level"))
            
            if not skill_name:
                continue

            
            proficiency = TaxonomyService.parse_proficiency_level(skill.get("level"))
            
            result = await db.execute(
                select(ProfileSkill).filter(
                    ProfileSkill.profile_id == profile_id,
                    ProfileSkill.skill_name == skill_name
                )
            )
            profile_skill = result.scalar_one_or_none()

            if profile_skill:
                profile_skill.proficiency_level = proficiency
                profile_skill.skill_type = SkillType.SOFT
                profile_skill.updated_at = now
            else:
                profile_skill = ProfileSkill(
                    profile_id=profile_id,
                    skill_name=skill_name,
                    proficiency_level=proficiency,
                    source=SkillSource.AI_EXTRACTED,
                    verified=False,
                    skill_type=SkillType.SOFT,
                    created_at=now,
                    updated_at=now
                )
                db.add(profile_skill)
            
            results["soft_skills"].append({
                "skill_name": skill_name,
                "skill_type": "SOFT"
            })
        
        # Commit all changes once
        await db.commit()
        return results
