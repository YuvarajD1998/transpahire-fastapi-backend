import logging
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.models.match_schemas import CandidateSimilarityResult, JobSimilarityResult, MatchFeaturesResponse

logger = logging.getLogger(__name__)


class MatchService:
    """
    Performs vector similarity search using pgvector cosine distance.
    Raw SQL is required because SQLAlchemy ORM cannot express the <=> operator.
    """

    async def find_similar_candidates(
        self,
        db: AsyncSession,
        job_id: int,
        embedding_type: str,
        top_k: int,
        min_similarity: float,
    ) -> List[CandidateSimilarityResult]:
        """
        Find candidates most similar to a job's embedding.
        Fetches the job's embedding vector then runs cosine similarity search
        against candidate_embeddings, filtered by min_similarity threshold.
        """
        job_embedding_type = self._map_to_job_embedding_type(embedding_type)

        result = await db.execute(
            text("""
                SELECT vector FROM job_embeddings
                WHERE job_id = :job_id AND type = :embedding_type
                ORDER BY version DESC
                LIMIT 1
            """),
            {"job_id": job_id, "embedding_type": job_embedding_type},
        )
        job_row = result.fetchone()
        if not job_row or job_row.vector is None:
            logger.warning(f"No {job_embedding_type} embedding found for job {job_id}")
            return []

        job_vector_str = job_row.vector

        rows = await db.execute(
            text("""
                SELECT
                    ce.candidate_id,
                    1 - (ce.vector <=> :job_vector::vector) AS similarity
                FROM candidate_embeddings ce
                WHERE
                    ce.type = :candidate_type
                    AND ce.vector IS NOT NULL
                    AND 1 - (ce.vector <=> :job_vector::vector) >= :min_similarity
                ORDER BY ce.vector <=> :job_vector::vector ASC
                LIMIT :top_k
            """),
            {
                "job_vector": job_vector_str,
                "candidate_type": embedding_type,
                "min_similarity": min_similarity,
                "top_k": top_k,
            },
        )

        results = [
            CandidateSimilarityResult(
                candidate_id=row.candidate_id,
                similarity=round(float(row.similarity), 4),
            )
            for row in rows.fetchall()
        ]
        logger.info(f"Found {len(results)} similar candidates for job {job_id} (type={embedding_type})")
        return results

    async def find_similar_jobs(
        self,
        db: AsyncSession,
        candidate_id: int,
        embedding_type: str,
        top_k: int,
        min_similarity: float,
    ) -> List[JobSimilarityResult]:
        """
        Find jobs most similar to a candidate's embedding.
        """
        result = await db.execute(
            text("""
                SELECT vector FROM candidate_embeddings
                WHERE candidate_id = :candidate_id AND type = :embedding_type
                ORDER BY version DESC
                LIMIT 1
            """),
            {"candidate_id": candidate_id, "embedding_type": embedding_type},
        )
        candidate_row = result.fetchone()
        if not candidate_row or candidate_row.vector is None:
            logger.warning(f"No {embedding_type} embedding found for candidate {candidate_id}")
            return []

        candidate_vector_str = candidate_row.vector
        job_embedding_type = self._map_to_job_embedding_type(embedding_type)

        rows = await db.execute(
            text("""
                SELECT
                    je.job_id,
                    1 - (je.vector <=> :candidate_vector::vector) AS similarity
                FROM job_embeddings je
                WHERE
                    je.type = :job_type
                    AND je.vector IS NOT NULL
                    AND 1 - (je.vector <=> :candidate_vector::vector) >= :min_similarity
                ORDER BY je.vector <=> :candidate_vector::vector ASC
                LIMIT :top_k
            """),
            {
                "candidate_vector": candidate_vector_str,
                "job_type": job_embedding_type,
                "min_similarity": min_similarity,
                "top_k": top_k,
            },
        )

        results = [
            JobSimilarityResult(
                job_id=row.job_id,
                similarity=round(float(row.similarity), 4),
            )
            for row in rows.fetchall()
        ]
        logger.info(f"Found {len(results)} similar jobs for candidate {candidate_id}")
        return results

    async def compute_match_features(
        self,
        db: AsyncSession,
        candidate_id: int,
        job_id: int,
    ) -> MatchFeaturesResponse:
        """
        Compute cosine similarities across multiple embedding type pairs.
        Returns semantic similarity features for a candidate-job pair.
        """
        pairs = [
            ('full', 'jd_summary'),
            ('skills', 'required_skills'),
            ('experience', 'responsibilities'),
        ]

        similarities = {}
        for candidate_type, job_type in pairs:
            sim = await self._compute_pair_similarity(db, candidate_id, job_id, candidate_type, job_type)
            similarities[f"{candidate_type}_vs_{job_type}"] = sim

        return MatchFeaturesResponse(
            candidate_id=candidate_id,
            job_id=job_id,
            semantic_similarity=similarities.get('full_vs_jd_summary'),
            skills_similarity=similarities.get('skills_vs_required_skills'),
            experience_similarity=similarities.get('experience_vs_responsibilities'),
        )

    async def _compute_pair_similarity(
        self,
        db: AsyncSession,
        candidate_id: int,
        job_id: int,
        candidate_type: str,
        job_type: str,
    ) -> Optional[float]:
        """Compute cosine similarity between a specific candidate-job embedding pair."""
        result = await db.execute(
            text("""
                SELECT
                    1 - (ce.vector <=> je.vector) AS similarity
                FROM candidate_embeddings ce
                CROSS JOIN job_embeddings je
                WHERE
                    ce.candidate_id = :candidate_id
                    AND ce.type = :candidate_type
                    AND je.job_id = :job_id
                    AND je.type = :job_type
                    AND ce.vector IS NOT NULL
                    AND je.vector IS NOT NULL
                ORDER BY ce.version DESC, je.version DESC
                LIMIT 1
            """),
            {
                "candidate_id": candidate_id,
                "candidate_type": candidate_type,
                "job_id": job_id,
                "job_type": job_type,
            },
        )
        row = result.fetchone()
        if not row:
            return None
        return round(float(row.similarity), 4)

    def _map_to_job_embedding_type(self, candidate_embedding_type: str) -> str:
        """Map candidate embedding type to corresponding job embedding type."""
        mapping = {
            'full': 'jd_summary',
            'summary': 'jd_summary',
            'skills': 'required_skills',
            'experience': 'responsibilities',
            'education': 'jd_summary',
        }
        return mapping.get(candidate_embedding_type, 'jd_summary')
