import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        self.model_name = model_name
        self.model = None
        self.embed_dim = 768
        
    def load_model(self):
        """Lazy load the model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("✅ Embedding model loaded successfully")
    
    def generate_embedding(self, text: str, prefix_type: str = "query") -> List[float]: 
        """Generate embedding for a single text with e5 prefix."""
        self.load_model()
        
        # Truncate text to avoid memory issues
        text = text[:2000]
        
        # Add e5 prefix for better accuracy
        prefixed_text = f"{prefix_type}: {text}"
        
        # Generate embedding
        embedding = self.model.encode(
            prefixed_text,  # ← Changed from 'text' to 'prefixed_text'
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embedding.tolist()

    
    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        self.load_model()
        
        # Truncate all texts
        texts = [text[:2000] for text in texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32
        )
        
        return embeddings.tolist()

# Singleton instance
embedding_service = EmbeddingService()
