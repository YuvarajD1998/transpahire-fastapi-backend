import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete, and_
from datetime import datetime

from app.models.database_models import Profile, Embedding, WorkExperience, Education, ProfileSkill, User
from app.services.gemini_service_embedding import GeminiService
from app.crud.resume_crud import ProfileCRUD, WorkExperienceCRUD, EducationCRUD, ProfileSkillCRUD

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self.gemini_service = GeminiService()
        self.embedding_model = "text-embedding-004"
        self.embedding_dimensions = 768

    async def generate_profile_embedding(self, db: AsyncSession, profile_id: int) -> Optional[List[float]]:
        """Generate and store embedding for a candidate profile"""
        try:
            logger.info(f"Generating Gemini embedding for profile {profile_id}")
            
            # Get complete profile data
            profile_data = await self.get_complete_profile_data(db, profile_id)
            if not profile_data:
                logger.warning(f"No profile data found for profile {profile_id}")
                return None
            
            # Create embedding text
            embedding_text = self.create_profile_embedding_text(profile_data)
            if not embedding_text.strip():
                logger.warning(f"Empty embedding text for profile {profile_id}")
                return None
            
            # Generate embedding vector
            embedding_vector = await self.gemini_service.generate_profile_embedding(embedding_text)
            
            # Store embedding in database
            await self.store_embedding(db, "PROFILE", profile_id, embedding_vector)
            
            # Update profile embedding status
            await self.update_profile_embedding_status(db, profile_id)
            
            logger.info(f"Successfully generated Gemini embedding for profile {profile_id}")
            return embedding_vector
            
        except Exception as e:
            logger.error(f"Failed to generate Gemini embedding for profile {profile_id}: {str(e)}")
            return None

    async def get_complete_profile_data(self, db: AsyncSession, profile_id: int) -> Optional[Dict]:
        """Get all related data for a profile"""
        try:
            # Get profile
            profile = await ProfileCRUD.get_by_id(db, profile_id)
            if not profile:
                return None
            
            # Get work experiences
            work_experiences = await WorkExperienceCRUD.get_by_profile_id(db, profile_id)
            
            # Get educations
            educations = await EducationCRUD.get_by_profile_id(db, profile_id)
            
            # Get skills
            skills = await ProfileSkillCRUD.get_by_profile_id(db, profile_id)
            
            return {
                "profile": profile,
                "work_experiences": work_experiences,
                "educations": educations,
                "skills": skills
            }
            
        except Exception as e:
            logger.error(f"Error getting complete profile data for {profile_id}: {str(e)}")
            return None

    def create_profile_embedding_text(self, profile_data: Dict) -> str:
        """Create comprehensive text representation of profile for embedding"""
        profile = profile_data["profile"]
        work_experiences = profile_data["work_experiences"]
        educations = profile_data["educations"]
        skills = profile_data["skills"]
        
        text_parts = []
        
        # Personal Information
        if profile.first_name and profile.last_name:
            text_parts.append(f"Name: {profile.first_name} {profile.last_name}")
        
        if profile.headline:
            text_parts.append(f"Professional Headline: {profile.headline}")
        
        if profile.bio:
            text_parts.append(f"Professional Summary: {profile.bio}")
        
        if profile.location:
            text_parts.append(f"Location: {profile.location}")
        
        # Work Experience
        for i, exp in enumerate(work_experiences, 1):
            exp_text = f"Experience {i}: {exp.position} at {exp.company}"
            if exp.location:
                exp_text += f", {exp.location}"
            if exp.start_date:
                exp_text += f" ({exp.start_date.strftime('%Y-%m')}"
                if exp.end_date:
                    exp_text += f" to {exp.end_date.strftime('%Y-%m')})"
                elif exp.is_current:
                    exp_text += " to present)"
                else:
                    exp_text += ")"
            
            if exp.description:
                description = exp.description[:300] + "..." if len(exp.description) > 300 else exp.description
                exp_text += f". Responsibilities: {description}"
            
            text_parts.append(exp_text)
        
        # Education
        for i, edu in enumerate(educations, 1):
            edu_text = f"Education {i}: {edu.degree}"
            if edu.field:
                edu_text += f" in {edu.field}"
            edu_text += f" from {edu.institution}"
            if edu.start_date and edu.end_date:
                edu_text += f" ({edu.start_date.strftime('%Y')}-{edu.end_date.strftime('%Y')})"
            text_parts.append(edu_text)
        
        # Skills
        if skills:
            skill_names = [skill.skill_name for skill in skills]
            text_parts.append(f"Skills: {', '.join(skill_names)}")
        
        return ". ".join(text_parts)

    async def store_embedding(self, db: AsyncSession, entity_type: str, entity_id: int, vector: List[float]) -> None:
        """Store embedding vector in database"""
        try:
            # Delete existing embedding
            await db.execute(
                delete(Embedding).where(
                    and_(
                        Embedding.entity_type == entity_type,
                        Embedding.entity_id == entity_id
                    )
                )
            )
            
            # Create new embedding
            embedding = Embedding(
                entity_type=entity_type,
                entity_id=entity_id,
                vector=vector,
                model_name=self.embedding_model,
                dimensions=self.embedding_dimensions,
                version=1
            )
            
            db.add(embedding)
            await db.commit()
            
            logger.info(f"Stored embedding for {entity_type}:{entity_id}")
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error storing embedding for {entity_type}:{entity_id}: {str(e)}")
            raise

    async def update_profile_embedding_status(self, db: AsyncSession, profile_id: int) -> None:
        """Update profile embedding generation status"""
        try:
            profile = await ProfileCRUD.get_by_id(db, profile_id)
            if profile:
                profile.embeddings_generated = True
                profile.embeddings_version = 1
                profile.updated_at = datetime.utcnow()
                await db.commit()
                logger.info(f"Updated embedding status for profile {profile_id}")
                
        except Exception as e:
            await db.rollback()
            logger.error(f"Error updating embedding status for profile {profile_id}: {str(e)}")
