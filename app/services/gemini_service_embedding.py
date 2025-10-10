import asyncio
import logging
from typing import List, Optional
from google import genai
from google.genai import types
import numpy as np
from app.config import settings


logger = logging.getLogger(__name__)


class GeminiService:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.embedding_model = "gemini-embedding-001"  # Correct model name
        self.embedding_dimensions_default = 768
        
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            logger.warning("Gemini API key not configured")
            self.client = None
    
    async def generate_embedding(
        self, 
        text: str, 
        dimensions: int = 768,
        task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> List[float]:
        """Generate embedding for given text using Gemini API"""
        if not self.client:
            raise RuntimeError("Gemini API key not configured")
        
        try:
            cleaned_text = text.strip()[:2000]  # Gemini has 2048 token limit
            
            # FIXED: Use correct API format from official documentation
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=cleaned_text,  # Direct string, not 'content=' parameter
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=dimensions
                    )
                )
            )
            
            # Extract embedding values from response (based on official docs)
            if hasattr(result, 'embeddings') and result.embeddings:
                embedding_obj = result.embeddings[0]  # Get first embedding
                if hasattr(embedding_obj, 'values'):
                    embedding_values = embedding_obj.values
                    
                    # Normalize embedding for smaller dimensions (as per docs)
                    if dimensions < 3072:
                        embedding_values_np = np.array(embedding_values)
                        normalized_embedding = embedding_values_np / np.linalg.norm(embedding_values_np)
                        embedding = normalized_embedding.tolist()
                    else:
                        embedding = embedding_values
                else:
                    embedding = embedding_obj
            else:
                raise RuntimeError("No embeddings found in response")
            
            if not embedding or len(embedding) == 0:
                raise RuntimeError("Empty embedding returned from API")
            
            logger.info(f"Generated Gemini embedding with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate Gemini embedding: {str(e)}")
            raise RuntimeError(f"Gemini embedding generation failed: {str(e)}")
    
    async def generate_profile_embedding(self, profile_text: str) -> List[float]:
        """Generate embedding for candidate profile"""
        return await self.generate_embedding(
            profile_text, 
            dimensions=768,
            task_type="RETRIEVAL_DOCUMENT"
        )
    
    def is_available(self) -> bool:
        """Check if Gemini service is available"""
        return self.client is not None
