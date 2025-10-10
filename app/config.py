from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    # Core
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # Database (shared with NestJS Prisma)
    DATABASE_URL: str

    # JWT Secret (must match NestJS service)
    jwt_secret: str
    jwt_refresh_secret: str 

    # Google Gemini (Primary AI Parser)
    GEMINI_API_KEY: Optional[str] = None  
    GEMINI_MODEL: str = "gemini-1.5-flash"  
    
    # OpenAI (First Fallback)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: Optional[str] = "gpt-3.5-turbo"

    # Hugging Face (Second Fallback)
    HUGGINGFACE_API_KEY: Optional[str] = None
    HUGGINGFACE_MODEL_ID: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    # File storage: S3 optional; local fallback
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    AWS_S3_BUCKET: str = "transpahire-resumes"

    # Local upload dir
    UPLOAD_DIR: str = "uploads/resumes"

    # Security / Auth
    SECRET_KEY: str = "your_secret_key_here"  # will be overridden by .env
    ALLOWED_FILE_TYPES: List[str] = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",  # Added .doc support
        "image/jpeg",          # Added image support for OCR
        "image/png",
        "image/tiff",
        "image/bmp"
    ]
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # Increased to 10MB for better resume support

    # AI Processing Settings
    AI_PROCESSING_TIMEOUT: int = 120  # seconds
    AI_MAX_RETRIES: int = 3
    AI_ENABLE_FALLBACK: bool = True
    
    # Gemini-specific settings
    GEMINI_TEMPERATURE: float = 0.1      # Low temperature for consistent parsing
    GEMINI_MAX_TOKENS: int = 2500        # Sufficient for structured resume data
    GEMINI_TOP_P: float = 0.95          # High precision for structured output
    GEMINI_TOP_K: int = 40              # Balanced diversity

    # Rate-limit, CORS, gzip
    CORS_ORIGINS: List[str] = ["*"]

    class Config:
        env_file = ".env"
        # Support both GEMINI_API_KEY and GOOGLE_API_KEY environment variables
        env_prefix = ""
        case_sensitive = False

    @property
    def effective_gemini_api_key(self) -> Optional[str]:
        """Get Gemini API key with fallback to GOOGLE_API_KEY as per Google's documentation."""
        return self.GEMINI_API_KEY or getattr(self, 'GOOGLE_API_KEY', None)


settings = Settings()
