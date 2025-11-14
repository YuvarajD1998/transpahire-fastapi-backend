from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Index
from sqlalchemy.sql import func
from datetime import datetime
from app.database import Base

# Import pgvector type
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    # Fallback if pgvector not installed
    from sqlalchemy import TEXT as Vector

class CandidateEmbedding(Base):
    __tablename__ = "candidate_embeddings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    candidate_id = Column(Integer, nullable=False, index=True)
    type = Column(String, nullable=False, index=True)  # summary, skills, experience, education, full, job_specific
    job_id = Column(Integer, nullable=True, index=True)
    vector = Column(Vector(768), nullable=True)  # pgvector column
    dimension = Column(Integer, nullable=False)
    model_name = Column(String, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, server_default=func.now(), onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index('idx_candidate_embeddings_candidate_type', 'candidate_id', 'type'),
    )


class JobEmbedding(Base):
    __tablename__ = "job_embeddings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, nullable=False, index=True)
    type = Column(String, nullable=False, index=True)  # jd_summary, required_skills, responsibilities
    vector = Column(Vector(768), nullable=True)
    dimension = Column(Integer, nullable=False)
    model_name = Column(String, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, server_default=func.now(), onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index('idx_job_embeddings_job_type', 'job_id', 'type'),
    )
