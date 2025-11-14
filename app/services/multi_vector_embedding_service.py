import logging
from typing import List, Dict, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, and_
from datetime import datetime

from app.models.embedding_models import CandidateEmbedding, JobEmbedding
from app.models.database_models import Profile
from app.services.gemini_service_embedding import GeminiService
from app.services.representation_service import ResumeRepresentationService, JobRepresentationService
from app.crud.resume_crud import ProfileCRUD, WorkExperienceCRUD, EducationCRUD, ProfileSkillCRUD

logger = logging.getLogger(__name__)


class MultiVectorEmbeddingService:
    """
    Service for generating and managing multi-vector embeddings.
    Creates 5 embeddings per candidate and 3 per job for improved matching.
    """
    
    CANDIDATE_EMBEDDING_TYPES = ['summary', 'skills', 'experience', 'education', 'full']
    JOB_EMBEDDING_TYPES = ['jd_summary', 'required_skills', 'responsibilities']
    
    def __init__(self):
        self.gemini_service = GeminiService()
        self.model_name = "text-embedding-004"
        self.dimension = 768
        self.representation_service = ResumeRepresentationService()
        self.job_representation_service = JobRepresentationService()
    
    # =========================================
    # CANDIDATE EMBEDDINGS
    # =========================================
    
    async def generate_candidate_embeddings(
        self, 
        db: AsyncSession, 
        candidate_id: int,
        regenerate: bool = False
    ) -> Dict[str, bool]:
        """
        Generate all 5 embeddings for a candidate profile.
        
        Returns:
            Dict with success status for each embedding type
        """
        logger.info(f"Generating multi-vector embeddings for candidate {candidate_id}")
        
        try:
            # Check if embeddings already exist
            if not regenerate:
                existing = await self._get_candidate_embeddings(db, candidate_id)
                if len(existing) >= 5:
                    logger.info(f"Embeddings already exist for candidate {candidate_id}")
                    return {etype: True for etype in self.CANDIDATE_EMBEDDING_TYPES}
            else:
                # Delete existing embeddings
                await self._delete_candidate_embeddings(db, candidate_id)
            
            # Get complete profile data
            profile_data = await self._get_complete_profile_data(db, candidate_id)
            if not profile_data:
                logger.warning(f"No profile data found for candidate {candidate_id}")
                return {etype: False for etype in self.CANDIDATE_EMBEDDING_TYPES}
            
            results = {}
            
            # Generate each embedding type
            for embedding_type in self.CANDIDATE_EMBEDDING_TYPES:
                try:
                    success = await self._generate_single_candidate_embedding(
                        db, candidate_id, embedding_type, profile_data
                    )
                    results[embedding_type] = success
                except Exception as e:
                    logger.error(f"Failed to generate {embedding_type} embedding for candidate {candidate_id}: {e}")
                    results[embedding_type] = False
            
            # Update profile embedding status
            if any(results.values()):
                await self._update_profile_embedding_status(db, candidate_id)
            
            logger.info(f"Generated embeddings for candidate {candidate_id}: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings for candidate {candidate_id}: {e}")
            return {etype: False for etype in self.CANDIDATE_EMBEDDING_TYPES}
    
    async def _generate_single_candidate_embedding(
        self,
        db: AsyncSession,
        candidate_id: int,
        embedding_type: str,
        profile_data: Dict
    ) -> bool:
        """Generate a single embedding for a specific type."""
        try:
            # Generate text representation
            text = self._get_candidate_text_for_type(embedding_type, profile_data)
            
            if not text or not text.strip():
                logger.warning(f"Empty text for {embedding_type} embedding, candidate {candidate_id}")
                return False
            
            # Generate embedding vector
            vector = await self.gemini_service.generate_embedding(
                text=text,
                dimensions=self.dimension,
                task_type="RETRIEVAL_DOCUMENT"
            )
            
            # Store in database
            await self._store_candidate_embedding(
                db, candidate_id, embedding_type, vector
            )
            
            logger.info(f"Generated {embedding_type} embedding for candidate {candidate_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating {embedding_type} embedding: {e}")
            return False
    
    def _get_candidate_text_for_type(self, embedding_type: str, profile_data: Dict) -> str:
        """Get text representation based on embedding type."""
        if embedding_type == 'summary':
            return self.representation_service.generate_summary_text(profile_data)
        elif embedding_type == 'skills':
            return self.representation_service.generate_skills_text(profile_data)
        elif embedding_type == 'experience':
            return self.representation_service.generate_experience_text(profile_data)
        elif embedding_type == 'education':
            return self.representation_service.generate_education_text(profile_data)
        elif embedding_type == 'full':
            return self.representation_service.generate_full_text(profile_data)
        else:
            return ""
    
    async def _store_candidate_embedding(
        self,
        db: AsyncSession,
        candidate_id: int,
        embedding_type: str,
        vector: List[float],
        job_id: Optional[int] = None
    ) -> None:
        """Store candidate embedding in database."""
        embedding = CandidateEmbedding(
            candidate_id=candidate_id,
            type=embedding_type,
            job_id=job_id,
            vector=vector,
            dimension=len(vector),
            model_name=self.model_name,
            version=1
        )
        
        db.add(embedding)
        await db.commit()
        await db.refresh(embedding)
    
    # =========================================
    # JOB EMBEDDINGS
    # =========================================
    
    async def generate_job_embeddings(
        self,
        db: AsyncSession,
        job_id: int,
        job_data: Dict,
        regenerate: bool = False
    ) -> Dict[str, bool]:
        """
        Generate all 3 embeddings for a job posting.
        
        Returns:
            Dict with success status for each embedding type
        """
        logger.info(f"Generating multi-vector embeddings for job {job_id}")
        
        try:
            # Check if embeddings already exist
            if not regenerate:
                existing = await self._get_job_embeddings(db, job_id)
                if len(existing) >= 3:
                    logger.info(f"Embeddings already exist for job {job_id}")
                    return {etype: True for etype in self.JOB_EMBEDDING_TYPES}
            else:
                # Delete existing embeddings
                await self._delete_job_embeddings(db, job_id)
            
            results = {}
            
            # Generate each embedding type
            for embedding_type in self.JOB_EMBEDDING_TYPES:
                try:
                    success = await self._generate_single_job_embedding(
                        db, job_id, embedding_type, job_data
                    )
                    results[embedding_type] = success
                except Exception as e:
                    logger.error(f"Failed to generate {embedding_type} embedding for job {job_id}: {e}")
                    results[embedding_type] = False
            
            logger.info(f"Generated embeddings for job {job_id}: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings for job {job_id}: {e}")
            return {etype: False for etype in self.JOB_EMBEDDING_TYPES}
    
    async def _generate_single_job_embedding(
        self,
        db: AsyncSession,
        job_id: int,
        embedding_type: str,
        job_data: Dict
    ) -> bool:
        """Generate a single embedding for a specific job type."""
        try:
            # Generate text representation
            text = self._get_job_text_for_type(embedding_type, job_data)
            
            if not text or not text.strip():
                logger.warning(f"Empty text for {embedding_type} embedding, job {job_id}")
                return False
            
            # Generate embedding vector
            vector = await self.gemini_service.generate_embedding(
                text=text,
                dimensions=self.dimension,
                task_type="RETRIEVAL_QUERY"  # Different task type for queries
            )
            
            # Store in database
            await self._store_job_embedding(
                db, job_id, embedding_type, vector
            )
            
            logger.info(f"Generated {embedding_type} embedding for job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating {embedding_type} embedding: {e}")
            return False
    
    def _get_job_text_for_type(self, embedding_type: str, job_data: Dict) -> str:
        """Get text representation based on job embedding type."""
        if embedding_type == 'jd_summary':
            return self.job_representation_service.generate_jd_summary(job_data)
        elif embedding_type == 'required_skills':
            return self.job_representation_service.generate_required_skills_text(job_data)
        elif embedding_type == 'responsibilities':
            return self.job_representation_service.generate_responsibilities_text(job_data)
        else:
            return ""
    
    async def _store_job_embedding(
        self,
        db: AsyncSession,
        job_id: int,
        embedding_type: str,
        vector: List[float]
    ) -> None:
        """Store job embedding in database."""
        embedding = JobEmbedding(
            job_id=job_id,
            type=embedding_type,
            vector=vector,
            dimension=len(vector),
            model_name=self.model_name,
            version=1
        )
        
        db.add(embedding)
        await db.commit()
        await db.refresh(embedding)
    
    # =========================================
    # RETRIEVAL METHODS
    # =========================================
    
    async def find_similar_candidates(
        self,
        db: AsyncSession,
        job_id: int,
        limit: int = 200,
        embedding_type: str = 'summary'
    ) -> List[Tuple[int, float]]:
        """
        Find similar candidates using ANN search on specified embedding type.
        
        Returns:
            List of (candidate_id, similarity_score) tuples
        """
        try:
            # Get job embedding
            job_embedding = await self._get_job_embedding(db, job_id, 'jd_summary')
            if not job_embedding:
                logger.warning(f"No job embedding found for job {job_id}")
                return []
            
            # Perform ANN search using pgvector
            query = f"""
                SELECT 
                    candidate_id,
                    1 - (vector <=> :job_vector) as similarity
                FROM candidate_embeddings
                WHERE type = :embedding_type
                ORDER BY vector <=> :job_vector
                LIMIT :limit
            """
            
            result = await db.execute(
                query,
                {
                    'job_vector': job_embedding.vector,
                    'embedding_type': embedding_type,
                    'limit': limit
                }
            )
            
            candidates = [(row[0], row[1]) for row in result.fetchall()]
            logger.info(f"Found {len(candidates)} similar candidates for job {job_id}")
            return candidates
            
        except Exception as e:
            logger.error(f"Error finding similar candidates: {e}")
            return []
    
    # =========================================
    # HELPER METHODS
    # =========================================
    
    async def _get_complete_profile_data(self, db: AsyncSession, candidate_id: int) -> Optional[Dict]:
        """Get all related data for a profile."""
        try:
            profile = await ProfileCRUD.get_by_id(db, candidate_id)
            if not profile:
                return None
            
            work_experiences = await WorkExperienceCRUD.get_by_profile_id(db, candidate_id)
            educations = await EducationCRUD.get_by_profile_id(db, candidate_id)
            skills = await ProfileSkillCRUD.get_by_profile_id(db, candidate_id)
            
            return {
                "profile": profile,
                "work_experiences": work_experiences,
                "educations": educations,
                "skills": skills
            }
            
        except Exception as e:
            logger.error(f"Error getting profile data: {e}")
            return None
    
    async def _get_candidate_embeddings(
        self, 
        db: AsyncSession, 
        candidate_id: int
    ) -> List[CandidateEmbedding]:
        """Get all existing embeddings for a candidate."""
        result = await db.execute(
            select(CandidateEmbedding).where(
                CandidateEmbedding.candidate_id == candidate_id
            )
        )
        return result.scalars().all()
    
    async def _get_job_embeddings(
        self,
        db: AsyncSession,
        job_id: int
    ) -> List[JobEmbedding]:
        """Get all existing embeddings for a job."""
        result = await db.execute(
            select(JobEmbedding).where(JobEmbedding.job_id == job_id)
        )
        return result.scalars().all()
    
    async def _get_job_embedding(
        self,
        db: AsyncSession,
        job_id: int,
        embedding_type: str
    ) -> Optional[JobEmbedding]:
        """Get specific job embedding by type."""
        result = await db.execute(
            select(JobEmbedding).where(
                and_(
                    JobEmbedding.job_id == job_id,
                    JobEmbedding.type == embedding_type
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def _delete_candidate_embeddings(self, db: AsyncSession, candidate_id: int) -> None:
        """Delete all embeddings for a candidate."""
        await db.execute(
            delete(CandidateEmbedding).where(
                CandidateEmbedding.candidate_id == candidate_id
            )
        )
        await db.commit()
    
    async def _delete_job_embeddings(self, db: AsyncSession, job_id: int) -> None:
        """Delete all embeddings for a job."""
        await db.execute(
            delete(JobEmbedding).where(JobEmbedding.job_id == job_id)
            )
        await db.commit()
    
    async def _update_profile_embedding_status(self, db: AsyncSession, profile_id: int) -> None:
        """Update profile embedding generation status."""
        try:
            profile = await ProfileCRUD.get_by_id(db, profile_id)
            if profile:
                profile.embeddings_generated = True
                profile.embeddings_version = 1
                profile.updated_at = datetime.utcnow()
                await db.commit()
                
        except Exception as e:
            await db.rollback()
            logger.error(f"Error updating profile embedding status: {e}")
