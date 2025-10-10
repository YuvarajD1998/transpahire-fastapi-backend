# enums.py
from enum import Enum

class Role(str, Enum):
    CANDIDATE = "CANDIDATE"
    RECRUITER = "RECRUITER"
    ORG_ADMIN = "ORG_ADMIN"
    ORG_MANAGER = "ORG_MANAGER"
    ORG_RECRUITER = "ORG_RECRUITER"
    PLATFORM_ADMIN = "PLATFORM_ADMIN"


class SubscriptionTier(str, Enum):
    FREE = "FREE"
    BASIC = "BASIC"
    PREMIUM = "PREMIUM"
    


class ApplicationStatus(str, Enum):
    SUBMITTED = "SUBMITTED"
    VIEWED = "VIEWED"
    SHORTLISTED = "SHORTLISTED"
    INTERVIEW_SCHEDULED = "INTERVIEW_SCHEDULED"
    REJECTED = "REJECTED"
    OFFERED = "OFFERED"
    ACCEPTED = "ACCEPTED"
    WITHDRAWN = "WITHDRAWN"


class PrivacyMode(Enum):
    PUBLIC = "PUBLIC"
    LIMITED = "LIMITED"
    ANONYMOUS = "ANONYMOUS"

class VerificationStatus(str, Enum):
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"


class UserStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    SUSPENDED = "SUSPENDED"
    DELETED = "DELETED"


class ParseStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SkillSource(str, Enum):
    MANUAL = "MANUAL"
    AI_EXTRACTED = "AI_EXTRACTED"
    VERIFIED = "VERIFIED"


# Fixed: Match Prisma enum values (UPPERCASE not lowercase)
class ProficiencyLevel(str, Enum):
    BEGINNER = "BEGINNER"
    INTERMEDIATE = "INTERMEDIATE" 
    ADVANCED = "ADVANCED"
    EXPERT = "EXPERT"
    
    @classmethod
    def _missing_(cls, value):
        # Handle case-insensitive matching
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return None

class OrgPlan(str, Enum):
    INDIVIDUAL = "INDIVIDUAL"
    ORG_BASIC = "ORG_BASIC"
    ORG_PREMIUM = "ORG_PREMIUM"
