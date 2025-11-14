from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.enums import SkillSource, ProficiencyLevel
from app.crud.resume_crud import ProfileCRUD, WorkExperienceCRUD, EducationCRUD, ProfileSkillCRUD
from app.services.multi_vector_embedding_service import MultiVectorEmbeddingService
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ResumeDataService:
    """Service to handle comprehensive resume data processing"""
    
    def __init__(self):
        # MultiVectorEmbeddingService will be initialized when needed
        pass

    async def process_parsed_resume_data_with_embeddings(
        self,
        db: AsyncSession,
        profile_id: int,
        resume_id: int,
        parsed_data: dict
    ) -> None:
        """
        ✅ NEW METHOD: Process parsed resume data and generate multi-vector embeddings.
        This is the main entry point for the new multi-vector system.
        """
        try:
            logger.info(f"Processing parsed resume data with multi-vector embeddings for profile {profile_id}")
            
            # 1. Store/update profile data
            await self._store_profile_data(db, profile_id, parsed_data)
            
            # 2. Store work experiences
            await self._store_work_experiences(db, profile_id, resume_id, parsed_data)
            
            # 3. Store education
            await self._store_education(db, profile_id, resume_id, parsed_data)
            
            # 4. Store skills
            await self._store_skills(db, profile_id, parsed_data)
            
            # 5. ✅ NEW: Generate multi-vector embeddings (5 embeddings per candidate)
            embedding_service = MultiVectorEmbeddingService()
            results = await embedding_service.generate_candidate_embeddings(
                db, profile_id, regenerate=True
            )
            
            logger.info(f"✅ Generated multi-vector embeddings for profile {profile_id}: {results}")
            logger.info(f"✅ Successfully processed resume {resume_id} with multi-vector embeddings")
            
        except Exception as e:
            logger.error(f"❌ Error processing resume with embeddings: {e}")
            await db.rollback()
            raise

    async def _store_profile_data(
        self, 
        db: AsyncSession, 
        profile_id: int, 
        parsed_data: Dict
    ) -> None:
        """
        Store/update profile data from parsed resume.
        Updates personal information fields if they don't exist.
        """
        try:
            personal_info = parsed_data.get("personal_info", {})
            summary = parsed_data.get("summary")
            
            # Get existing profile
            from sqlalchemy import select
            from app.models.database_models import Profile
            
            result = await db.execute(select(Profile).where(Profile.id == profile_id))
            profile = result.scalar_one_or_none()
            
            if not profile:
                logger.warning(f"Profile {profile_id} not found")
                return
            
            # --- Extract name ---
            if personal_info.get("name"):
                name_parts = personal_info["name"].split()
                if len(name_parts) >= 2:
                    if not profile.first_name:
                        profile.first_name = name_parts[0]
                    if not profile.last_name:
                        profile.last_name = " ".join(name_parts[1:])
                elif len(name_parts) == 1:
                    if not profile.first_name:
                        profile.first_name = name_parts[0]
            
            # --- Extract contact info ---
            if personal_info.get("phone") and not profile.phone:
                profile.phone = personal_info["phone"]
            
            if personal_info.get("location") and not profile.location:
                profile.location = personal_info["location"]
            
            # --- Extract URLs ---
            if personal_info.get("linkedin") and not profile.linkedin_url:
                profile.linkedin_url = personal_info["linkedin"]
            
            if personal_info.get("github") and not profile.github_url:
                profile.github_url = personal_info["github"]
            
            if personal_info.get("website") and not profile.website_url:
                profile.website_url = personal_info["website"]
            
            # --- Extract summary/bio ---
            if summary and not profile.bio:
                profile.bio = summary[:500]  # Truncate to fit field size
            
            # --- Calculate profile completeness ---
            fields = ["first_name", "last_name", "phone", "location", "headline", "bio", "linkedin_url"]
            filled = sum(1 for f in fields if getattr(profile, f))
            profile.profile_completeness = int((filled / len(fields)) * 100)
            
            # --- Update timestamp ---
            profile.updated_at = datetime.utcnow()
            
            await db.commit()
            await db.refresh(profile)
            
            logger.info(f"✅ Updated profile {profile_id} with parsed data")
            
        except Exception as e:
            logger.error(f"❌ Error storing profile data for profile {profile_id}: {e}")
            await db.rollback()
            raise

    async def _store_work_experiences(
        self,
        db: AsyncSession,
        profile_id: int,
        resume_id: int,
        parsed_data: Dict
    ) -> None:
        """
        Store work experiences from parsed resume data.
        Creates new experiences for this resume.
        """
        try:
            experiences = parsed_data.get("experience", [])
            
            if not experiences:
                logger.info(f"No work experiences found for profile {profile_id}")
                return
            
            logger.info(f"📁 Processing {len(experiences)} work experiences for profile {profile_id}")
            
            # Create new work experiences
            for i, exp_data in enumerate(experiences, start=1):
                try:
                    # Extract company and position (required fields)
                    company = exp_data.get("company", "").strip()
                    position = exp_data.get("position", "").strip()
                    
                    if not company:
                        company = "Unknown Company"
                    if not position:
                        position = "Unknown Position"
                    
                    # Extract optional fields
                    location = exp_data.get("location")
                    start_date = exp_data.get("start_date")
                    end_date = exp_data.get("end_date")
                    is_current = exp_data.get("is_current", False)
                    description = exp_data.get("description")
                    achievements = exp_data.get("achievements", [])
                    skills = exp_data.get("skills", [])
                    
                    # Create work experience entry
                    work_exp = await WorkExperienceCRUD.create_work_experience(
                        db=db,
                        profile_id=profile_id,
                        company=company,
                        position=position,
                        location=location,
                        start_date=start_date,
                        end_date=end_date,
                        is_current=is_current,
                        description=description,
                        achievements=achievements,
                        skills=skills,
                        resume_id=resume_id
                    )
                    
                    logger.info(f"✅ Experience {i}: {position} at {company} - created ID {work_exp.id}")
                    
                except Exception as e:
                    logger.error(f"❌ Error creating experience {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            logger.info(f"✅ Stored {len(experiences)} work experiences for profile {profile_id}")
            
        except Exception as e:
            logger.error(f"❌ Error storing work experiences for profile {profile_id}: {e}")
            await db.rollback()
            raise

    async def _store_education(
        self,
        db: AsyncSession,
        profile_id: int,
        resume_id: int,
        parsed_data: Dict
    ) -> None:
        """
        Store education records from parsed resume data.
        Creates new education records for this resume.
        """
        try:
            educations = parsed_data.get("education", [])
            
            if not educations:
                logger.info(f"No education records found for profile {profile_id}")
                return
            
            logger.info(f"🎓 Processing {len(educations)} education entries for profile {profile_id}")
            
            # Create new education records
            for i, edu_data in enumerate(educations, start=1):
                try:
                    # Extract required fields
                    institution = edu_data.get("institution", "").strip()
                    degree = edu_data.get("degree", "").strip()
                    
                    if not institution:
                        institution = "Unknown Institution"
                    if not degree:
                        degree = "Unknown Degree"
                    
                    # Extract optional fields
                    field = edu_data.get("field")
                    start_date = edu_data.get("start_date")
                    end_date = edu_data.get("end_date")
                    grade = edu_data.get("grade")
                    description = edu_data.get("description")
                    
                    # Create education entry
                    education = await EducationCRUD.create_education(
                        db=db,
                        profile_id=profile_id,
                        institution=institution,
                        degree=degree,
                        field=field,
                        start_date=start_date,
                        end_date=end_date,
                        grade=grade,
                        description=description,
                        resume_id=resume_id,
                        source="AI_EXTRACTED"
                    )
                    
                    logger.info(f"✅ Education {i}: {degree} from {institution} - created ID {education.id}")
                    
                except Exception as e:
                    logger.error(f"❌ Error creating education {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            logger.info(f"✅ Stored {len(educations)} education records for profile {profile_id}")
            
        except Exception as e:
            logger.error(f"❌ Error storing education for profile {profile_id}: {e}")
            await db.rollback()
            raise

    async def _store_skills(
        self,
        db: AsyncSession,
        profile_id: int,
        parsed_data: Dict
    ) -> None:
        """
        Store skills from parsed resume data.
        Handles both naming conventions: technical_skills/soft_skills AND technicalskills/softskills
        """
        try:
            skills_root = parsed_data.get("skills", {})
            
            if not skills_root:
                logger.warning(f"⚠️ No 'skills' key found in parsed data for profile {profile_id}")
                return
            
            # Clear existing AI-extracted skills
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
            logger.info(f"🧹 Cleared existing AI-extracted skills for profile {profile_id}")
            
            # ✅ FIXED: Handle both naming conventions
            skills_to_process = []
            
            # Extract technical skills (try both naming conventions)
            technical_skills = (
                skills_root.get("technical_skills") or 
                skills_root.get("technicalskills") or 
                skills_root.get("technicalSkills") or
                []
            )
            
            if isinstance(technical_skills, list):
                logger.info(f"Found {len(technical_skills)} technical skills")
                for skill in technical_skills:
                    if isinstance(skill, dict) and skill.get("name"):
                        skills_to_process.append({
                            **skill,
                            "category": skill.get("category") or "Technical",
                            "skill_type": "TECHNICAL"
                        })
                    elif isinstance(skill, str):
                        skills_to_process.append({
                            "name": skill,
                            "category": "Technical",
                            "skill_type": "TECHNICAL"
                        })
            else:
                logger.warning(f"⚠️ technical_skills is not a list: {type(technical_skills)}")
            
            # Extract soft skills (try both naming conventions)
            soft_skills = (
                skills_root.get("soft_skills") or 
                skills_root.get("softskills") or 
                skills_root.get("softSkills") or
                []
            )
            
            if isinstance(soft_skills, list):
                logger.info(f"Found {len(soft_skills)} soft skills")
                for skill in soft_skills:
                    if isinstance(skill, dict) and skill.get("name"):
                        skills_to_process.append({
                            **skill,
                            "category": skill.get("category") or "Soft Skill",
                            "skill_type": "SOFT"
                        })
                    elif isinstance(skill, str):
                        skills_to_process.append({
                            "name": skill,
                            "category": "Soft Skill",
                            "skill_type": "SOFT"
                        })
            else:
                logger.warning(f"⚠️ soft_skills is not a list: {type(soft_skills)}")
            
            # ✅ Fallback: Handle flat list structure
            if not skills_to_process and isinstance(skills_root, list):
                logger.info(f"Using fallback: skills_root is a flat list")
                for skill in skills_root:
                    if isinstance(skill, dict) and skill.get("name"):
                        skills_to_process.append(skill)
                    elif isinstance(skill, str):
                        skills_to_process.append({"name": skill, "category": "Technical"})
            
            if not skills_to_process:
                logger.warning(f"⚠️ No skills found in parsed data for profile {profile_id}")
                logger.warning(f"Skills root keys: {list(skills_root.keys()) if isinstance(skills_root, dict) else 'not a dict'}")
                logger.warning(f"Skills root structure: {skills_root}")
                return
            
            logger.info(f"🧠 Processing {len(skills_to_process)} skills for profile {profile_id}")
            
            total_stored = 0
            errors = []
            
            # Process each skill
            for i, skill_dict in enumerate(skills_to_process, start=1):
                skill_name = skill_dict.get("name")
                
                if not skill_name or not skill_name.strip():
                    logger.warning(f"⚠️ Skipping skill {i}: missing or empty name")
                    continue
                
                skill_name = skill_name.strip()
                
                # ✅ Skip invalid skill names (category names)
                invalid_names = [
                    'technicalskills', 'softskills', 'technical', 'soft',
                    'technical_skills', 'soft_skills', 'technicalSkills', 'softSkills'
                ]
                if skill_name.lower() in invalid_names:
                    logger.warning(f"⚠️ Skipping invalid skill name (category): {skill_name}")
                    continue
                
                logger.debug(f"Processing skill {i}/{len(skills_to_process)}: {skill_name}")
                
                try:
                    # Extract skill attributes
                    category = skill_dict.get("category")
                    if not category:
                        category = "Technical" if skill_dict.get("skill_type") == "TECHNICAL" else "Soft Skill"
                    
                    # Handle proficiency level
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
                    elif prof_value is None:
                        proficiency = ProficiencyLevel.INTERMEDIATE
                    else:
                        proficiency = ProficiencyLevel.INTERMEDIATE
                    
                    # Extract years of experience
                    years_experience = skill_dict.get("years_experience")
                    if years_experience:
                        if isinstance(years_experience, str):
                            try:
                                years_experience = int(years_experience)
                            except ValueError:
                                years_experience = None
                        elif not isinstance(years_experience, int):
                            years_experience = None
                    
                    # Upsert skill (creates new or updates existing)
                    skill_obj = await ProfileSkillCRUD.upsert_skill(
                        db=db,
                        profile_id=profile_id,
                        skill_name=skill_name,
                        category=category,
                        proficiency_level=proficiency,
                        years_experience=years_experience,
                        source=SkillSource.AI_EXTRACTED
                    )
                    
                    total_stored += 1
                    logger.info(f"✅ Skill {i}/{len(skills_to_process)}: {skill_obj.skill_name} stored (ID {skill_obj.id})")
                    
                except Exception as e:
                    error_msg = f"❌ Error processing skill {i} ({skill_name}): {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    import traceback
                    traceback.print_exc()
                    # Don't raise - continue processing other skills
                    continue
            
            logger.info(f"✅ Successfully stored {total_stored}/{len(skills_to_process)} skills for profile {profile_id}")
            
            if errors:
                logger.warning(f"⚠️ Encountered {len(errors)} errors during skill processing")
                for error in errors[:5]:  # Log first 5 errors
                    logger.warning(error)
            
            if total_stored == 0:
                logger.error(f"⚠️ No skills were stored! Check skill data structure.")
                logger.error(f"Skills root: {skills_root}")
            
        except Exception as e:
            logger.error(f"❌ Critical error storing skills for profile {profile_id}: {e}")
            import traceback
            traceback.print_exc()
            await db.rollback()
            raise




    # ============================================
    # LEGACY METHOD (Keep for backward compatibility)
    # ============================================
    @staticmethod
    async def process_parsed_resume_data(
        db: AsyncSession, 
        profile_id: int, 
        resume_id: int,
        parsed_data: Dict[str, Any]
    ) -> None:
        """
        ⚠️ LEGACY METHOD: Process and store all parsed resume data.
        
        This method is kept for backward compatibility.
        For new implementations, use process_parsed_resume_data_with_embeddings() instead.
        """
        try:
            logger.warning(f"⚠️ Using legacy process_parsed_resume_data method. Consider migrating to process_parsed_resume_data_with_embeddings()")
            logger.info(f"Processing resume data for profile {profile_id}")
            logger.info(f"Parsed keys: {list(parsed_data.keys())}")

            # --- Extract ---
            experience_data = parsed_data.get("experience", [])
            education_data = parsed_data.get("education", [])
            skills_data = parsed_data.get("skills", [])
            personal_info = parsed_data.get("personal_info", {})
            summary = parsed_data.get("summary")

            logger.info(f"Found {len(experience_data)} experiences, {len(education_data)} education items, {len(skills_data)} skills.")

            # --- Step 1: Update profile info ---
            if personal_info:
                await ResumeDataService.update_profile_info(db, profile_id, personal_info, summary)
                logger.info("✅ Profile updated")

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
            logger.info("🧹 Cleared existing AI-extracted skills")

            # --- Step 3: Work Experiences ---
            if experience_data:
                logger.info(f"📁 Processing {len(experience_data)} work experiences...")
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
                        logger.info(f"✅ Experience {i}: {work_exp.company} - created ID {work_exp.id}")
                    except Exception as e:
                        logger.error(f"❌ Error creating experience {i}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            else:
                logger.info("ℹ️ No work experience found")

            # --- Step 4: Education ---
            if education_data:
                logger.info(f"🎓 Processing {len(education_data)} education entries...")
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
                        logger.info(f"✅ Education {i}: {education.institution} - created ID {education.id}")
                    except Exception as e:
                        logger.error(f"❌ Error creating education {i}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            else:
                logger.info("ℹ️ No education data found")

            # --- Step 5: Skills ---
            if skills_data:
                logger.info(f"🧠 Processing {len(skills_data)} skills...")

                # ✅ Normalize input — handle both dicts and strings
                normalized_skills = []
                for s in skills_data:
                    if isinstance(s, str):
                        normalized_skills.append({"name": s})
                    elif isinstance(s, dict):
                        normalized_skills.append(s)
                    else:
                        logger.warning(f"⚠️ Skipped invalid skill format: {s}")

                for i, skill_dict in enumerate(normalized_skills, start=1):
                    skill_name = skill_dict.get("name")
                    if not skill_name:
                        continue

                    logger.debug(f"Processing skill {i}: {skill_name}")

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

                        logger.info(f"✅ Skill {i}: {skill_obj.skill_name} stored (ID {skill_obj.id})")
                    except Exception as e:
                        logger.error(f"❌ Error processing skill {i} ({skill_name}): {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue
            else:
                logger.info("ℹ️ No skills found")

            logger.info(f"✅ Successfully processed resume for profile {profile_id}")

        except Exception as e:
            logger.error(f"❌ Error processing resume data for profile {profile_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            await db.rollback()
            raise

    @staticmethod
    async def update_profile_info(
        db: AsyncSession, 
        profile_id: int, 
        personal_info: Dict, 
        summary: str = None
    ):
        """
        ⚠️ LEGACY METHOD: Update profile with personal information.
        Kept for backward compatibility with process_parsed_resume_data().
        """
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
