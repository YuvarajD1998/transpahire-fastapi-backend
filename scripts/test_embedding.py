import asyncio
import asyncpg
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = "postgresql://transpahire_user:transpahire_password_123@localhost:5432/transpahire_db"
MODEL_NAME = "intfloat/e5-base-v2"

class EmbeddingTester:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        
    def generate_embedding_with_prefix(self, text: str, prefix_type: str = "query") -> np.ndarray:
        """Generate embedding with proper e5 prefix"""
        prefixed_text = f"{prefix_type}: {text}"
        embedding = self.model.encode(prefixed_text, normalize_embeddings=True)
        return embedding
    
    def generate_embedding_without_prefix(self, text: str) -> np.ndarray:
        """Generate embedding without prefix (current problematic approach)"""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    def build_skill_text(self, skill_name: str, context_snippets: List[str] = None) -> str:
        """Build rich text similar to your API method"""
        text = f"Skill: {skill_name}"
        if context_snippets:
            text += f"\nContext: {' '.join(context_snippets)}"
        return text
    
    async def test_taxonomy_vs_temporary(self):
        """Main test comparing taxonomy embeddings vs temporary embeddings"""
        
        # Test cases: (skill_name, context_snippets, expected_similar_skill_in_taxonomy)
        test_cases = [
            ("Python Programming", ["backend development", "FastAPI framework"], "Python"),
            ("JavaScript", ["React development", "frontend"], "JavaScript"),
            ("Data Analysis", ["pandas", "numpy", "statistical analysis"], "Data Analysis"),
            ("Plumbing Installation", ["pipe fitting", "water systems"], "Plumbing Installation"),
            ("Machine Learning", ["scikit-learn", "model training"], "Machine Learning")
        ]
        
        conn = await asyncpg.connect(DATABASE_URL)
        
        results = []
        
        for skill_name, context, expected_match in test_cases:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: {skill_name}")
            logger.info(f"Context: {context}")
            
            # Build text similar to your API
            skill_text = self.build_skill_text(skill_name, context)
            
            # Generate embeddings with different approaches
            temp_emb_no_prefix = self.generate_embedding_without_prefix(skill_text)
            temp_emb_with_prefix = self.generate_embedding_with_prefix(skill_text, "query")
            
            # Test against taxonomy (assuming taxonomy uses passage prefix or no prefix)
            # APPROACH 1: Direct comparison (your current approach)
            matches_no_prefix = await self.find_similar_skills(conn, temp_emb_no_prefix, limit=5)
            
            # APPROACH 2: With proper prefix (recommended)
            matches_with_prefix = await self.find_similar_skills(conn, temp_emb_with_prefix, limit=5)
            
            logger.info(f"\nResults WITHOUT prefix:")
            for i, (name, similarity) in enumerate(matches_no_prefix, 1):
                logger.info(f"  {i}. {name}: {similarity:.4f}")
            
            logger.info(f"\nResults WITH prefix:")
            for i, (name, similarity) in enumerate(matches_with_prefix, 1):
                logger.info(f"  {i}. {name}: {similarity:.4f}")
            
            results.append({
                'skill': skill_name,
                'no_prefix_top_match': matches_no_prefix[0] if matches_no_prefix else None,
                'with_prefix_top_match': matches_with_prefix[0] if matches_with_prefix else None
            })
        
        await conn.close()
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info(f"{'='*60}")
        for result in results:
            logger.info(f"\nSkill: {result['skill']}")
            logger.info(f"  No prefix: {result['no_prefix_top_match']}")
            logger.info(f"  With prefix: {result['with_prefix_top_match']}")
    
    async def find_similar_skills(self, conn, embedding: np.ndarray, limit: int = 5) -> List[Tuple[str, float]]:
        """Find similar skills from taxonomy"""
        vector_str = "[" + ",".join(f"{x:.6f}" for x in embedding.tolist()) + "]"
        
        rows = await conn.fetch("""
            SELECT 
                skill_name,
                1 - (embedding <=> $1::vector) AS similarity
            FROM skill_taxonomy
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """, vector_str, limit)
        
        return [(row['skill_name'], row['similarity']) for row in rows]
    
    async def test_prefix_impact(self):
        """Test the impact of prefixes on embedding similarity"""
        
        logger.info("\n" + "="*60)
        logger.info("PREFIX IMPACT TEST")
        logger.info("="*60)
        
        test_text = "Python programming for backend development"
        
        # Generate all combinations
        emb_no_prefix = self.generate_embedding_without_prefix(test_text)
        emb_query_prefix = self.generate_embedding_with_prefix(test_text, "query")
        emb_passage_prefix = self.generate_embedding_with_prefix(test_text, "passage")
        
        # Compare similarities
        logger.info(f"\nTest text: {test_text}")
        logger.info(f"\nSelf-similarity tests:")
        logger.info(f"  No prefix vs No prefix: {self.cosine_similarity(emb_no_prefix, emb_no_prefix):.4f}")
        logger.info(f"  Query vs Query: {self.cosine_similarity(emb_query_prefix, emb_query_prefix):.4f}")
        logger.info(f"  Passage vs Passage: {self.cosine_similarity(emb_passage_prefix, emb_passage_prefix):.4f}")
        
        logger.info(f"\nCross-similarity tests:")
        logger.info(f"  No prefix vs Query: {self.cosine_similarity(emb_no_prefix, emb_query_prefix):.4f}")
        logger.info(f"  No prefix vs Passage: {self.cosine_similarity(emb_no_prefix, emb_passage_prefix):.4f}")
        logger.info(f"  Query vs Passage: {self.cosine_similarity(emb_query_prefix, emb_passage_prefix):.4f}")
    
    async def test_similarity_thresholds(self):
        """Test to find appropriate similarity thresholds"""
        
        logger.info("\n" + "="*60)
        logger.info("SIMILARITY THRESHOLD TEST")
        logger.info("="*60)
        
        conn = await asyncpg.connect(DATABASE_URL)
        
        # Sample a random skill from taxonomy
        row = await conn.fetchrow("""
            SELECT skill_name, category, subcategory, embedding
            FROM skill_taxonomy
            WHERE embedding IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 1
        """)
        
        if row:
            skill_name = row['skill_name']
            logger.info(f"\nRandom skill: {skill_name}")
            logger.info(f"Category: {row['category']} > {row['subcategory']}")
            
            # Generate temporary embedding for the same skill
            temp_text = self.build_skill_text(skill_name, [row['category'], row['subcategory']])
            temp_emb = self.generate_embedding_with_prefix(temp_text, "query")
            
            # Find matches
            matches = await self.find_similar_skills(conn, temp_emb, limit=10)
            
            logger.info(f"\nTop 10 matches:")
            for i, (name, similarity) in enumerate(matches, 1):
                match_indicator = "✓ EXACT" if name.lower() == skill_name.lower() else ""
                logger.info(f"  {i}. {name}: {similarity:.4f} {match_indicator}")
        
        await conn.close()

async def main():
    tester = EmbeddingTester()
    
    # Run all tests
    await tester.test_prefix_impact()
    await tester.test_taxonomy_vs_temporary()
    await tester.test_similarity_thresholds()

if __name__ == "__main__":
    asyncio.run(main())
