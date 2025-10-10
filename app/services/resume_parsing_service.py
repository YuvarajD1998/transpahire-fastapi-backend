from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.enums import SkillSource, ProficiencyLevel
from app.crud.resume_crud import ProfileCRUD, WorkExperienceCRUD, EducationCRUD, ProfileSkillCRUD
from app.services.embedding_service import EmbeddingService
import logging

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
            # Process resume data as usual
            await self.process_parsed_resume_data(db, profile_id,resume_id, parsed_data)
            
            # Generate embeddings after data processing
            try:
                embedding_vector = await self.embedding_service.generate_profile_embedding(db, profile_id)
                if embedding_vector:
                    logger.info(f"Successfully generated Gemini embeddings for profile {profile_id}")
                else:
                    logger.warning(f"Failed to generate embeddings for profile {profile_id}")
            except Exception as e:
                logger.error(f"Error generating Gemini embeddings: {str(e)}")
                
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
            print(f"Raw parsed data keys: {list(parsed_data.keys())}")
            
            # Extract data arrays
            experience_data = parsed_data.get("experience", [])
            education_data = parsed_data.get("education", [])
            skills_data = parsed_data.get("skills", [])
            personal_info = parsed_data.get("personal_info", {})
            
            print(f"Experience count: {len(experience_data)}")
            print(f"Education count: {len(education_data)}")
            print(f"Skills count: {len(skills_data)}")
            
            # 1. Update Profile with personal information
            if personal_info:
                await ResumeDataService.update_profile_info(db, profile_id, personal_info, parsed_data.get("summary"))
                print("Profile updated successfully")
            
            # 2. Clear existing AI-extracted skills to avoid duplicates
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
            print("Existing AI-extracted skills cleared")
            
            # 3. Process Work Experience
            if experience_data and len(experience_data) > 0:
                print(f"Processing {len(experience_data)} work experiences...")
                for i, exp_dict in enumerate(experience_data):
                    print(f"Processing experience {i+1}: {exp_dict.get('company', 'Unknown')} - {exp_dict.get('position', 'Unknown')}")
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
                        print(f"Work experience {i+1} created with ID {work_exp.id}")
                    except Exception as e:
                        print(f"Error creating work experience {i+1}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue
            else:
                print("No work experience data found in parsed resume")
            
            # 4. Process Education
            if education_data and len(education_data) > 0:
                print(f"Processing {len(education_data)} education entries...")
                for i, edu_dict in enumerate(education_data):
                    print(f"Processing education {i+1}: {edu_dict.get('institution', 'Unknown')} - {edu_dict.get('degree', 'Unknown')}")
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
                            source='AI_EXTRACTED'
                        )
                        print(f"Education {i+1} created with ID {education.id}")
                    except Exception as e:
                        print(f"Error creating education {i+1}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue
            else:
                print("No education data found in parsed resume")
            
            # 5. Process Skills
            if skills_data and len(skills_data) > 0:
                print(f"Processing {len(skills_data)} skills...")
                for i, skill_dict in enumerate(skills_data):
                    skill_name = skill_dict.get("name")
                    if not skill_name:
                        continue
                    
                    print(f"Processing skill {i+1}: {skill_name}")
                    try:
                        # Map proficiency level
                        proficiency = None
                        proficiency_str = skill_dict.get("proficiency_level")
                        if proficiency_str:
                            proficiency_map = {
                                "Beginner": ProficiencyLevel.BEGINNER,
                                "Intermediate": ProficiencyLevel.INTERMEDIATE,
                                "Advanced": ProficiencyLevel.ADVANCED,
                                "Expert": ProficiencyLevel.EXPERT,
                            }
                            proficiency = proficiency_map.get(proficiency_str, ProficiencyLevel.INTERMEDIATE)
                        
                        skill_obj = await ProfileSkillCRUD.upsert_skill(
                            db=db,
                            profile_id=profile_id,
                            skill_name=skill_name,
                            category=skill_dict.get("category"),
                            proficiency_level=proficiency,
                            years_experience=skill_dict.get("years_experience"),
                            source=SkillSource.AI_EXTRACTED
                        )
                        print(f"Skill {i+1} processed with ID {skill_obj.id}")
                    except Exception as e:
                        print(f"Error processing skill {i+1}: {str(e)}")
                        continue
            else:
                print("No skills data found in parsed resume")
            
            print(f"Successfully processed resume data for profile {profile_id}")
            
        except Exception as e:
            print(f"Error processing resume data for profile {profile_id}: {str(e)}")
            import traceback
            traceback.print_exc()
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
        
        # Update profile fields if they don't exist or are empty
        if personal_info.get("name") and not (profile.first_name and profile.last_name):
            name_parts = personal_info["name"].split()
            if len(name_parts) >= 2:
                if not profile.first_name:
                    profile.first_name = name_parts[0]
                if not profile.last_name:
                    profile.last_name = " ".join(name_parts[1:])
            elif len(name_parts) == 1:
                if not profile.first_name:
                    profile.first_name = name_parts[0]
        
        if personal_info.get("phone") and not profile.phone:
            profile.phone = personal_info["phone"]
        
        if personal_info.get("location") and not profile.location:
            profile.location = personal_info["location"]
        
        if personal_info.get("linkedin") and not profile.linkedin_url:
            profile.linkedin_url = personal_info["linkedin"]
        
        if personal_info.get("github") and not profile.github_url:
            profile.github_url = personal_info["github"]
        
        if summary and not profile.bio:
            profile.bio = summary
        
        # Calculate completeness score
        fields = ["first_name", "last_name", "phone", "location", "headline", "bio", "linkedin_url"]
        filled = sum(1 for f in fields if getattr(profile, f))
        profile.profile_completeness = int((filled / len(fields)) * 100)
        
        from datetime import datetime
        profile.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(profile)
        return profile
