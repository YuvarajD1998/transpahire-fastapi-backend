from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.enums import SkillSource, ProficiencyLevel
from app.crud.resume_crud import ProfileCRUD, WorkExperienceCRUD, EducationCRUD, ProfileSkillCRUD
from app.services.embedding_service import EmbeddingService
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ResumeDataService:
    """Service to handle comprehensive resume data processing"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()

    async def process_parsed_resume_data_with_embeddings(
        self, 
        db: AsyncSession, 
        profile_id: int, 
        resume_id: int,
        parsed_data: Dict[str, Any]
    ) -> None:
        """Process resume data AND generate Gemini embeddings"""
        try:
            # Process resume data
            await self.process_parsed_resume_data(db, profile_id, resume_id, parsed_data)
            
            # Generate embeddings after data processing
            try:
                embedding_vector = await self.embedding_service.generate_profile_embedding(db, profile_id)
                if embedding_vector:
                    logger.info(f"✅ Successfully generated embeddings for profile {profile_id}")
                else:
                    logger.warning(f"⚠️ Failed to generate embeddings for profile {profile_id}")
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in process_parsed_resume_data_with_embeddings: {str(e)}")
            raise

    @staticmethod
    async def process_parsed_resume_data(
        db: AsyncSession, 
        profile_id: int, 
        resume_id: int,
        parsed_data: Dict[str, Any]
    ) -> None:
        """Process and store all parsed resume data into respective tables"""
        try:
            print(f"Processing resume data for profile {profile_id}")
            print(f"Parsed keys: {list(parsed_data.keys())}")

            # --- Extract ---
            experience_data = parsed_data.get("experience", [])
            education_data = parsed_data.get("education", [])
            skills_data = parsed_data.get("skills", [])
            personal_info = parsed_data.get("personal_info", {})
            summary = parsed_data.get("summary")

            print(f"Found {len(experience_data)} experiences, {len(education_data)} education items, {len(skills_data)} skills.")

            # --- Step 1: Update profile info ---
            if personal_info:
                await ResumeDataService.update_profile_info(db, profile_id, personal_info, summary)
                print("✅ Profile updated")

            # --- Step 2: Clear AI-extracted skills ---
            from sqlalchemy import delete, and_
            from app.models.database_models import ProfileSkill

            await db.execute(
                delete(ProfileSkill).where(
                    and_(
                        ProfileSkill.profile_id == profile_id,
                        ProfileSkill.source == SkillSource.AI_EXTRACTED
                    )
                )
            )
            await db.commit()
            print("🧹 Cleared existing AI-extracted skills")

            # --- Step 3: Work Experiences ---
            if experience_data:
                print(f"📁 Processing {len(experience_data)} work experiences...")
                for i, exp_dict in enumerate(experience_data, start=1):
                    try:
                        work_exp = await WorkExperienceCRUD.create_work_experience(
                            db=db,
                            profile_id=profile_id,
                            company=exp_dict.get("company") or "Unknown Company",
                            position=exp_dict.get("position") or "Unknown Position",
                            location=exp_dict.get("location"),
                            start_date=exp_dict.get("start_date"),
                            end_date=exp_dict.get("end_date"),
                            is_current=exp_dict.get("is_current", False),
                            description=exp_dict.get("description"),
                            achievements=exp_dict.get("achievements", []),
                            skills=exp_dict.get("skills", []),
                            resume_id=resume_id,
                        )
                        print(f"✅ Experience {i}: {work_exp.company} - created ID {work_exp.id}")
                    except Exception as e:
                        print(f"❌ Error creating experience {i}: {e}")
                        import traceback; traceback.print_exc()
                        continue
            else:
                print("ℹ️ No work experience found")

            # --- Step 4: Education ---
            if education_data:
                print(f"🎓 Processing {len(education_data)} education entries...")
                for i, edu_dict in enumerate(education_data, start=1):
                    try:
                        education = await EducationCRUD.create_education(
                            db=db,
                            profile_id=profile_id,
                            institution=edu_dict.get("institution") or "Unknown Institution",
                            degree=edu_dict.get("degree") or "Unknown Degree",
                            field=edu_dict.get("field"),
                            start_date=edu_dict.get("start_date"),
                            end_date=edu_dict.get("end_date"),
                            grade=edu_dict.get("grade"),
                            description=edu_dict.get("description"),
                            resume_id=resume_id,
                            source="AI_EXTRACTED"
                        )
                        print(f"✅ Education {i}: {education.institution} - created ID {education.id}")
                    except Exception as e:
                        print(f"❌ Error creating education {i}: {e}")
                        import traceback; traceback.print_exc()
                        continue
            else:
                print("ℹ️ No education data found")

            # --- Step 5: Skills ---
            if skills_data:
                print(f"🧠 Processing {len(skills_data)} skills...")

                # ✅ Normalize input — handle both dicts and strings
                normalized_skills = []
                for s in skills_data:
                    if isinstance(s, str):
                        normalized_skills.append({"name": s})
                    elif isinstance(s, dict):
                        normalized_skills.append(s)
                    else:
                        print(f"⚠️ Skipped invalid skill format: {s}")

                for i, skill_dict in enumerate(normalized_skills, start=1):
                    skill_name = skill_dict.get("name")
                    if not skill_name:
                        continue

                    print(f"Processing skill {i}: {skill_name}")

                    try:
                        proficiency = None
                        prof_value = skill_dict.get("proficiency_level") or skill_dict.get("level")

                        if isinstance(prof_value, str):
                            prof_map = {
                                "BEGINNER": ProficiencyLevel.BEGINNER,
                                "INTERMEDIATE": ProficiencyLevel.INTERMEDIATE,
                                "ADVANCED": ProficiencyLevel.ADVANCED,
                                "EXPERT": ProficiencyLevel.EXPERT,
                            }
                            proficiency = prof_map.get(prof_value.upper(), ProficiencyLevel.INTERMEDIATE)
                        else:
                            proficiency = ProficiencyLevel.INTERMEDIATE

                        skill_obj = await ProfileSkillCRUD.upsert_skill(
                            db=db,
                            profile_id=profile_id,
                            skill_name=skill_name,
                            category=skill_dict.get("category"),
                            proficiency_level=proficiency,
                            years_experience=skill_dict.get("years_experience"),
                            source=SkillSource.AI_EXTRACTED
                        )

                        print(f"✅ Skill {i}: {skill_obj.skill_name} stored (ID {skill_obj.id})")
                    except Exception as e:
                        print(f"❌ Error processing skill {i} ({skill_name}): {str(e)}")
                        import traceback; traceback.print_exc()
                        continue
            else:
                print("ℹ️ No skills found")

            print(f"✅ Successfully processed resume for profile {profile_id}")

        except Exception as e:
            print(f"❌ Error processing resume data for profile {profile_id}: {str(e)}")
            import traceback; traceback.print_exc()
            await db.rollback()
            raise

    @staticmethod
    async def update_profile_info(db: AsyncSession, profile_id: int, personal_info: Dict, summary: str = None):
        """Update profile with personal information"""
        from sqlalchemy import select
        from app.models.database_models import Profile
        
        result = await db.execute(select(Profile).where(Profile.id == profile_id))
        profile = result.scalar_one_or_none()
        if not profile:
            return None

        # --- Basic details ---
        if personal_info.get("name"):
            name_parts = personal_info["name"].split()
            if len(name_parts) >= 2:
                profile.first_name = name_parts[0]
                profile.last_name = " ".join(name_parts[1:])
            elif len(name_parts) == 1:
                profile.first_name = name_parts[0]

        if personal_info.get("phone"):
            profile.phone = personal_info["phone"]

        if personal_info.get("location"):
            profile.location = personal_info["location"]

        if personal_info.get("linkedin"):
            profile.linkedin_url = personal_info["linkedin"]

        if personal_info.get("github"):
            profile.github_url = personal_info["github"]

        if summary:
            profile.bio = summary

        # --- Calculate completeness ---
        fields = ["first_name", "last_name", "phone", "location", "headline", "bio", "linkedin_url"]
        filled = sum(1 for f in fields if getattr(profile, f))
        profile.profile_completeness = int((filled / len(fields)) * 100)

        # --- Update timestamp ---
        profile.updated_at = datetime.utcnow()

        await db.commit()
        await db.refresh(profile)
        return profile
