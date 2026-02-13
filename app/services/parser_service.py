import base64
import io
from typing import Optional
from unstructured.partition.auto import partition
import logging

from app.services.gemini_service import GeminiService
from app.models.schemas import ParsedResumeData

logger = logging.getLogger(__name__)

class ParserService:
    def __init__(self):
        self.gemini = GeminiService()

    async def parse_resume(
        self, 
        file_content_base64: str, 
        filename: str, 
        enhance_images: bool = True
    ) -> ParsedResumeData:
        """Parse resume from base64 encoded content."""
        try:
            # Decode base64 content
            file_content = base64.b64decode(file_content_base64)
            
            # Extract text using unstructured
            text = await self._extract_text(file_content, filename)
            
            if not text.strip():
                raise ValueError("No text extracted from file")
            
            # Parse with Gemini
            if self.gemini.is_available():
                return await self.gemini.parse_resume_text(text)
            else:
                raise RuntimeError("Gemini service not available")
                
        except Exception as e:
            logger.error(f"Parsing failed: {e}")
            raise

    async def _extract_text(self, file_content: bytes, filename: str) -> str:
        """Extract text from file using unstructured library."""
        try:
            # Save to temporary file for unstructured
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            
            try:
                # Extract using unstructured
                elements = partition(filename=tmp_path)
                text = "\n\n".join([str(el) for el in elements])
                return text
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise RuntimeError(f"Failed to extract text: {str(e)}")
