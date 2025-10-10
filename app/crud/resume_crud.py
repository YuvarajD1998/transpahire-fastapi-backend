#Resume_crud.py
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy import and_, or_, update, delete
from datetime import datetime
from dateutil import parser as date_parser

from app.models.database_models import (
    Profile, Resume, ResumeCritique, ProfileSkill, 
    WorkExperience, Education  # Add these imports
)
from app.models.enums import ParseStatus, SkillSource, ProficiencyLevel
from app.models.schemas import ProfileUpdateRequest, ParsedResumeData


# Add these new CRUD classes

class WorkExperienceCRUD:
    @staticmethod
    async def create_work_experience(
        db: AsyncSession,
        profile_id: int,
        company: str,
        position: str,
        location: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        is_current: bool = False,
        description: Optional[str] = None,
        achievements: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
        resume_id: Optional[int] = None 
        
        
    ) -> WorkExperience:
        # Parse date strings to datetime objects
        start_dt = None
        end_dt = None
        
        if start_date:
            try:
                start_dt = date_parser.parse(start_date)
            except:
                pass
                
        if end_date:
            try:
                end_dt = date_parser.parse(end_date)
            except:
                pass
        
        work_exp = WorkExperience(
            profile_id=profile_id,
            company=company,
            position=position,
            location=location,
            start_date=start_dt,
            end_date=end_dt,
            is_current=is_current,
            description=description,
            achievements=achievements or [],
            skills=skills or [],
            resume_id=resume_id 
        )
        
        db.add(work_exp)
        await db.commit()
        await db.refresh(work_exp)
        return work_exp
    
    @staticmethod
    async def get_by_profile_id(db: AsyncSession, profile_id: int) -> List[WorkExperience]:
        result = await db.execute(
            select(WorkExperience)
            .where(WorkExperience.profile_id == profile_id)
            .order_by(WorkExperience.start_date.desc().nullslast())
        )
        return result.scalars().all()
    
    @staticmethod
    async def delete_by_profile_id(db: AsyncSession, profile_id: int) -> None:
        await db.execute(
            delete(WorkExperience).where(WorkExperience.profile_id == profile_id)
        )
        await db.commit()


class EducationCRUD:
    @staticmethod
    async def create_education(
        db: AsyncSession,
        profile_id: int,
        institution: str,
        degree: str,
        field: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        grade: Optional[str] = None,
        description: Optional[str] = None,
        resume_id: Optional[int] = None,
        source: str = 'AI_EXTRACTED' 
    ) -> Education:
        # Parse date strings to datetime objects
        start_dt = None
        end_dt = None
        
        if start_date:
            try:
                start_dt = date_parser.parse(start_date)
            except:
                pass
                
        if end_date:
            try:
                end_dt = date_parser.parse(end_date)
            except:
                pass
        
        education = Education(
            profile_id=profile_id,
            institution=institution,
            degree=degree,
            field=field,
            start_date=start_dt,
            end_date=end_dt,
            grade=grade,
            description=description,
            resume_id=resume_id,  # FIXED: Include this field
            source=source         # FIXED: Include this field
        )
        
        db.add(education)
        await db.commit()
        await db.refresh(education)
        return education
    
    @staticmethod
    async def get_by_profile_id(db: AsyncSession, profile_id: int) -> List[Education]:
        result = await db.execute(
            select(Education)
            .where(Education.profile_id == profile_id)
            .order_by(Education.start_date.desc().nullslast())
        )
        return result.scalars().all()
    
    @staticmethod
    async def delete_by_profile_id(db: AsyncSession, profile_id: int) -> None:
        await db.execute(
            delete(Education).where(Education.profile_id == profile_id)
        )
        await db.commit()


class ProfileSkillCRUD:
    @staticmethod
    async def create_skill(
        db: AsyncSession,
        profile_id: int,
        skill_name: str,
        category: Optional[str] = None,
        proficiency_level: Optional[ProficiencyLevel] = None,
        years_experience: Optional[int] = None,
        source: SkillSource = SkillSource.AI_EXTRACTED,
        verified: bool = False
    ) -> ProfileSkill:
        skill = ProfileSkill(
            profile_id=profile_id,
            skill_name=skill_name,
            category=category,
            proficiency_level=proficiency_level,
            years_experience=years_experience,
            source=source,
            verified=verified
        )
        
        db.add(skill)
        await db.commit()
        await db.refresh(skill)
        return skill
    
    @staticmethod
    async def get_by_profile_id(db: AsyncSession, profile_id: int) -> List[ProfileSkill]:
        result = await db.execute(
            select(ProfileSkill)
            .where(ProfileSkill.profile_id == profile_id)
            .order_by(ProfileSkill.category, ProfileSkill.skill_name)
        )
        return result.scalars().all()
    
    @staticmethod
    async def delete_by_profile_id(db: AsyncSession, profile_id: int) -> None:
        await db.execute(
            delete(ProfileSkill).where(ProfileSkill.profile_id == profile_id)
        )
        await db.commit()
    
    @staticmethod
    async def upsert_skill(
        db: AsyncSession,
        profile_id: int,
        skill_name: str,
        category: Optional[str] = None,
        proficiency_level: Optional[ProficiencyLevel] = None,
        years_experience: Optional[int] = None,
        source: SkillSource = SkillSource.AI_EXTRACTED
    ) -> ProfileSkill:
        # Check if skill already exists
        result = await db.execute(
            select(ProfileSkill).where(
                and_(
                    ProfileSkill.profile_id == profile_id,
                    ProfileSkill.skill_name == skill_name
                )
            )
        )
        existing_skill = result.scalar_one_or_none()
        
        if existing_skill:
            # Update existing skill
            if category:
                existing_skill.category = category
            if proficiency_level:
                existing_skill.proficiency_level = proficiency_level
            if years_experience:
                existing_skill.years_experience = years_experience
            existing_skill.updated_at = datetime.utcnow()
            await db.commit()
            await db.refresh(existing_skill)
            return existing_skill
        else:
            # Create new skill
            return await ProfileSkillCRUD.create_skill(
                db, profile_id, skill_name, category, 
                proficiency_level, years_experience, source
            )


# Enhanced ProfileCRUD with new methods
class ProfileCRUD:
    @staticmethod
    async def get_by_user_id(db: AsyncSession, user_id: int) -> Optional[Profile]:
        result = await db.execute(
            select(Profile).where(Profile.user_id == user_id)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def update_profile_from_resume_data(
        db: AsyncSession,
        profile_id: int,
        parsed_data: ParsedResumeData
    ) -> Optional[Profile]:
        """Update profile with extracted resume data"""
        result = await db.execute(
            select(Profile).where(Profile.id == profile_id)
        )
        profile = result.scalar_one_or_none()
        
        if not profile:
            return None
        
        # Extract personal info from parsed data
        personal_info = parsed_data.personal_info or {}
        
        # Update profile fields if they don't exist or are empty
        if personal_info.get('name') and not (profile.first_name and profile.last_name):
            name_parts = personal_info['name'].split()
            if len(name_parts) >= 2:
                if not profile.first_name:
                    profile.first_name = name_parts[0]
                if not profile.last_name:
                    profile.last_name = ' '.join(name_parts[1:])
            elif len(name_parts) == 1:
                if not profile.first_name:
                    profile.first_name = name_parts[0]
        
        if personal_info.get('phone') and not profile.phone:
            profile.phone = personal_info['phone']
        
        if personal_info.get('location') and not profile.location:
            profile.location = personal_info['location']
        
        if personal_info.get('linkedin') and not profile.linkedin_url:
            profile.linkedin_url = personal_info['linkedin']
        
        if personal_info.get('github') and not profile.github_url:
            profile.github_url = personal_info['github']
        
        if parsed_data.summary and not profile.bio:
            profile.bio = parsed_data.summary
        
        # Calculate completeness score
        fields = ["first_name", "last_name", "phone", "location", "headline", "bio", "linkedin_url"]
        filled = sum(1 for f in fields if getattr(profile, f))
        profile.profile_completeness = int((filled / len(fields)) * 100)
        
        profile.updated_at = datetime.utcnow()
        
        await db.commit()
        await db.refresh(profile)
        return profile
    
    @staticmethod
    async def update_profile(
        db: AsyncSession, 
        profile_id: int, 
        update_data: ProfileUpdateRequest
    ) -> Optional[Profile]:
        result = await db.execute(
            select(Profile).where(Profile.id == profile_id)
        )
        profile = result.scalar_one_or_none()
        
        if not profile:
            return None
            
        # Update fields
        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(profile, field, value)
        
        profile.updated_at = datetime.utcnow()
        
        # Calculate completeness score
        fields = ["first_name", "last_name", "phone", "location", "headline", "bio", "linkedin_url"]
        filled = sum(1 for f in fields if getattr(profile, f))
        profile.profile_completeness = int((filled / len(fields)) * 100)
        
        await db.commit()
        await db.refresh(profile)
        return profile
    @staticmethod
    async def get_by_id(db: AsyncSession, profile_id: int) -> Optional[Profile]:
        """Get profile by ID"""
        result = await db.execute(select(Profile).where(Profile.id == profile_id))
        return result.scalar_one_or_none()

class ResumeCRUD:
    @staticmethod
    async def create_resume(
        db: AsyncSession,
        profile_id: int,
        filename: str,
        original_name: str,
        file_path: str,
        file_size: int,
        mimetype: str,
        is_primary: bool = False
    ) -> Resume:
        resume = Resume(
            profile_id=profile_id,
            filename=filename,
            original_name=original_name,
            file_path=file_path,
            file_size=file_size,
            mimetype=mimetype,
            is_primary=is_primary,
            parse_status=ParseStatus.PENDING
        )
        
        db.add(resume)
        await db.commit()
        await db.refresh(resume)
        return resume
    
    @staticmethod
    async def get_resumes_by_profile(db: AsyncSession, profile_id: int) -> List[Resume]:
        result = await db.execute(
            select(Resume)
            .where(Resume.profile_id == profile_id)
            .order_by(Resume.created_at.desc())
        )
        return result.scalars().all()
    
    @staticmethod
    async def get_resume_by_id(
        db: AsyncSession, 
        resume_id: int, 
        profile_id: int
    ) -> Optional[Resume]:
        result = await db.execute(
            select(Resume).where(
                and_(Resume.id == resume_id, Resume.profile_id == profile_id)
            )
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def update_resume(
        db: AsyncSession,
        resume_id: int,
        update_fields: Dict[str, Any]
    ) -> Optional[Resume]:
        result = await db.execute(
            select(Resume).where(Resume.id == resume_id)
        )
        resume = result.scalar_one_or_none()
        
        if not resume:
            return None
            
        for field, value in update_fields.items():
            setattr(resume, field, value)
        
        resume.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(resume)
        return resume
    
    @staticmethod
    async def set_primary_resume(
        db: AsyncSession, 
        profile_id: int, 
        resume_id: int
    ) -> None:
        # First, unset all as primary for this profile
        await db.execute(
            update(Resume)
            .where(Resume.profile_id == profile_id)
            .values(is_primary=False, updated_at=datetime.utcnow())
        )
        
        # Then set the specified resume as primary
        await db.execute(
            update(Resume)
            .where(and_(Resume.id == resume_id, Resume.profile_id == profile_id))
            .values(is_primary=True, updated_at=datetime.utcnow())
        )
        
        await db.commit()
    
    @staticmethod
    async def delete_resume(db: AsyncSession, resume_id: int, profile_id: int) -> bool:
        result = await db.execute(
            delete(Resume).where(
                and_(Resume.id == resume_id, Resume.profile_id == profile_id)
            )
        )
        await db.commit()
        return result.rowcount > 0

class ResumeCritiqueCRUD:
    @staticmethod
    async def create_critique(
        db: AsyncSession,
        resume_id: int,
        critique_data: Dict[str, Any]
    ) -> ResumeCritique:
        # Convert CritiqueSections to dict if it's a Pydantic object
        sections_dict = critique_data["sections"]
        if hasattr(sections_dict, 'dict'):
            sections_dict = sections_dict.dict()
        elif hasattr(sections_dict, 'model_dump'):
            sections_dict = sections_dict.model_dump()
        
        critique = ResumeCritique(
            resume_id=resume_id,
            overall_score=critique_data["overall_score"],
            sections=sections_dict,  # Now it's a proper dict for JSON
            suggestions=critique_data["suggestions"],
            strengths=critique_data["strengths"], 
            weaknesses=critique_data["weaknesses"],
            ai_model=critique_data.get("ai_model", "gemini-1.5-flash")
        )
        
        db.add(critique)
        await db.commit()
        await db.refresh(critique)
        return critique
    @staticmethod
    async def get_critiques_by_resume(
        db: AsyncSession, 
        resume_id: int
    ) -> List[ResumeCritique]:
        """Get all critiques for a specific resume."""
        result = await db.execute(
            select(ResumeCritique)
            .where(ResumeCritique.resume_id == resume_id)
            .order_by(ResumeCritique.created_at.desc())
        )
        return result.scalars().all()
    




