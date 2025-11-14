#database-modal.py
from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DateTime, Float, JSON,
    ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from datetime import datetime

from app.models.enums import (
    Role, SubscriptionTier, ParseStatus, SkillSource, 
    ProficiencyLevel, UserStatus, PrivacyMode,SkillType
)

Base = declarative_base()

# Reference existing PostgreSQL enum types (created by Prisma)
privacy_mode_enum = ENUM(PrivacyMode, name='PrivacyMode', create_type=False)
role_enum = ENUM(Role, name='Role', create_type=False)
subscription_tier_enum = ENUM(SubscriptionTier, name='SubscriptionTier', create_type=False)
parse_status_enum = ENUM(ParseStatus, name='ParseStatus', create_type=False)
skill_source_enum = ENUM(SkillSource, name='SkillSource', create_type=False)
proficiency_level_enum = ENUM(ProficiencyLevel, name='ProficiencyLevel', create_type=False)
user_status_enum = ENUM(UserStatus, name='UserStatus', create_type=False)
skill_type_enum = ENUM(SkillType, name='SkillType', create_type=False)


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column("role", role_enum, nullable=False)
    status = Column("status", user_status_enum, nullable=False)
    tenant_id = Column("tenant_id", Integer, ForeignKey("organizations.id"), nullable=True)
    verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, server_default=func.now(), onupdate=datetime.utcnow, nullable=False)
    
    profile = relationship("Profile", back_populates="user", uselist=False)
    subscription = relationship("Subscription", back_populates="user", uselist=False)

class Profile(Base):
    __tablename__ = "profiles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    phone = Column(String, nullable=True)
    location = Column(String, nullable=True)
    headline = Column(String, nullable=True)
    bio = Column(String, nullable=True)
    linkedin_url = Column(String, nullable=True)
    github_url = Column(String, nullable=True)
    website_url = Column(String, nullable=True)
    profile_completeness = Column(Integer, default=0, nullable=False)
    privacy_mode = Column(privacy_mode_enum, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, server_default=func.now(), onupdate=datetime.utcnow, nullable=False)
    embeddings_generated = Column(Boolean, default=False, nullable=False)
    embeddings_version = Column(Integer, nullable=True)
    
    user = relationship("User", back_populates="profile")
    resumes = relationship("Resume", back_populates="profile")
    profile_skills = relationship("ProfileSkill", back_populates="profile")
    work_experiences = relationship("WorkExperience", back_populates="profile")
    educations = relationship("Education", back_populates="profile")

class Resume(Base):
    __tablename__ = "resumes"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    profile_id = Column(Integer, ForeignKey("profiles.id"), nullable=False)
    filename = Column(String, nullable=False)
    original_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    mimetype = Column(String, nullable=False)
    parse_status = Column(parse_status_enum, nullable=False)
    confidence_score = Column(Float, nullable=True)
    is_primary = Column(Boolean, default=False, nullable=False)
    parsed_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, server_default=func.now(), onupdate=datetime.utcnow, nullable=False)
    
    profile = relationship("Profile", back_populates="resumes")
    critiques = relationship("ResumeCritique", back_populates="resume")
    
    __table_args__ = (
        Index("idx_resumes_profile_id_is_primary", "profile_id", "is_primary"),
        Index("idx_resumes_parse_status", "parse_status"),
    )

class ResumeCritique(Base):
    __tablename__ = "resume_critiques"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    resume_id = Column(Integer, ForeignKey("resumes.id"), nullable=False)
    overall_score = Column(Integer, nullable=False)
    sections = Column(JSON, nullable=False)
    suggestions = Column(JSON, nullable=False)
    strengths = Column(JSON, nullable=False)
    weaknesses = Column(JSON, nullable=False)
    ai_model = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, server_default=func.now(), nullable=False)
    
    resume = relationship("Resume", back_populates="critiques")

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    from sqlalchemy import TEXT as Vector

class SkillTaxonomy(Base):
    __tablename__ = "skill_taxonomy"

    id = Column(Integer, primary_key=True, index=True)
    skill_name = Column(String, nullable=False)
    normalized_name = Column(String, unique=True, nullable=False, index=True)
    parent_skill = Column(String, nullable=True, index=True)
    skill_level = Column(Integer, nullable=False, index=True)
    synonyms = Column(JSON, nullable=True)
    related_skills = Column(JSON, nullable=True)
    specializations = Column(JSON, nullable=True)
    industry_relevance = Column(JSON, nullable=True)
    skill_weight = Column(Float, default=0.5)
    
    # pgvector fields
    embedding = Column(Vector(768), nullable=True)
    embedding_model = Column(String, nullable=True)
    embedding_updated_at = Column(DateTime, nullable=True)
    needs_embedding_update = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    profile_skills = relationship("ProfileSkill", back_populates="skill_taxonomy")



class NonTaxonomySkill(Base):
    __tablename__ = "non_taxonomy_skills"

    id = Column(Integer, primary_key=True, index=True)
    skill_name = Column(String, nullable=False)
    normalized_name = Column(String, nullable=False, index=True)
    source = Column(String, nullable=False)
    source_id = Column(Integer, nullable=True)
    frequency = Column(Integer, default=1)
    reviewed = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        Index('idx_normalized_source_sourceid', 'normalized_name', 'source', 'source_id', unique=True),
    )



class ProfileSkill(Base):
    __tablename__ = "profile_skills"

    id = Column(Integer, primary_key=True, index=True)
    profile_id = Column(Integer, ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    skill_taxonomy_id = Column(Integer, ForeignKey("skill_taxonomy.id", ondelete="SET NULL"), nullable=True, index=True)
    skill_name = Column(String, nullable=False)
    category = Column(String, nullable=True)
    proficiency_level = Column(proficiency_level_enum, nullable=True)
    years_experience = Column(Integer, nullable=True)
    source = Column(skill_source_enum, default=SkillSource.AI_EXTRACTED)
    verified = Column(Boolean, default=False)
    skill_type = Column(skill_type_enum, default=SkillType.TECHNICAL)


    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    profile = relationship("Profile", back_populates="profile_skills")
    skill_taxonomy = relationship("SkillTaxonomy", back_populates="profile_skills")  


    __table_args__ = (
        Index('idx_profile_skill_unique', 'profile_id', 'skill_name', unique=True),
    )



class Subscription(Base):
    __tablename__ = "subscriptions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    tier = Column(subscription_tier_enum, nullable=False)
    status = Column(String, default="active", nullable=False)
    stripe_customer_id = Column(String, nullable=True)
    stripe_subscription_id = Column(String, nullable=True)
    current_period_start = Column(DateTime, nullable=True)
    current_period_end = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, server_default=func.now(), onupdate=datetime.utcnow, nullable=False)
    
    user = relationship("User", back_populates="subscription")

class Organization(Base):
    __tablename__ = "organizations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)
    verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, server_default=func.now(), onupdate=datetime.utcnow, nullable=False)

class WorkExperience(Base):
    __tablename__ = "work_experiences"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    profile_id = Column(Integer, ForeignKey("profiles.id"), nullable=False)
    company = Column(String, nullable=False)
    position = Column(String, nullable=False)
    location = Column(String, nullable=True)
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    is_current = Column(Boolean, default=False, nullable=False)
    description = Column(Text, nullable=True)
    achievements = Column(JSON, nullable=True)
    skills = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, server_default=func.now(), onupdate=datetime.utcnow, nullable=False)
    resume_id = Column(Integer, ForeignKey('resumes.id'), nullable=True)
    source = Column(String, default='AI_EXTRACTED') # MANUAL, AI_EXTRACTED, VERIFIED
    
    profile = relationship("Profile", back_populates="work_experiences")
    resume = relationship("Resume") 
    
    __table_args__ = (
        Index("idx_work_experiences_profile_id", "profile_id"),
    )


class Education(Base):
    __tablename__ = "educations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    profile_id = Column(Integer, ForeignKey("profiles.id"), nullable=False)
    institution = Column(String, nullable=False)
    degree = Column(String, nullable=False)
    field = Column(String, nullable=True)
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    grade = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, server_default=func.now(), onupdate=datetime.utcnow, nullable=False)
    resume_id = Column(Integer, ForeignKey('resumes.id'), nullable=True) 
    source = Column(String, default='AI_EXTRACTED') # MANUAL, AI_EXTRACTED, VERIFIED
    
    # Relationship
    
    profile = relationship("Profile", back_populates="educations")
    resume = relationship("Resume")
    
    __table_args__ = (
        Index("idx_educations_profile_id", "profile_id"),
    )

class Embedding(Base):
    __tablename__ = "embeddings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_type = Column(String, nullable=False)
    entity_id = Column(Integer, nullable=False)
    vector = Column(JSON, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    model_name = Column(String, default="text-embedding-004", nullable=False)
    dimensions = Column(Integer, default=768, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, server_default=func.now(), onupdate=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        UniqueConstraint("entity_type", "entity_id", name="uq_embeddings_entity_type_entity_id"),
        Index("idx_embeddings_entity_type", "entity_type"),
        Index("idx_embeddings_entity_id", "entity_id"),
    )