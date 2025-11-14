import os
import asyncio
import asyncpg
import logging
from datetime import datetime
from google import genai
from google.genai import types
import numpy as np
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://transpahire_user:transpahire_password_123@localhost:5432/transpahire_db"
)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAdvFhvGWaIl_ZFlNo98-cBGGRMWd0uHs8")

MODEL = "gemini-embedding-001"
DIMENSIONS = 768
BATCH_SIZE = 50
MAX_TEXT_LEN = 2000

# ============================================
# Text Building Functions
# ============================================

def build_skill_card(row, synonyms_text: str = "") -> str:
    """Build rich text representation of a skill for embedding"""
    
    def parse_json_field(field):
        """Safely parse JSON field"""
        if not field:
            return []
        if isinstance(field, str):
            try:
                return json.loads(field)
            except:
                return []
        return field if isinstance(field, list) else []

    def format_list(items):
        """Format list items for text"""
        if not items:
            return "None"
        return ", ".join(str(x) for x in items[:10])  # Limit to 10 items

    def format_dict(d):
        """Format dictionary for text"""
        if not d:
            return "None"
        if isinstance(d, str):
            try:
                d = json.loads(d)
            except:
                return d
        return ", ".join(f"{k}({v:.2f})" for k, v in list(d.items())[:5])

    # Parse JSON fields
    related_skills = parse_json_field(row.get('related_skills'))
    specializations = parse_json_field(row.get('specializations'))
    industry_relevance = row.get('industry_relevance')
    role_relevance = row.get('role_relevance')

    # Build comprehensive text
    text = f"""
Skill: {row['skill_name']}
Type: {row.get('skill_type', 'TECHNICAL')}
Category: {row.get('category', 'Unknown')} > {row.get('subcategory', 'General')}
Level: {row['skill_level']} (Parent: {row.get('parent_skill', 'None')})

Synonyms: {synonyms_text or 'None'}

Related Skills: {format_list(related_skills)}
Specializations: {format_list(specializations)}

Industry Relevance: {format_dict(industry_relevance)}
Role Relevance: {format_dict(role_relevance)}

Weights: Base={row.get('base_weight', 0.5):.2f}, Technical={row.get('technical_role_weight', 0.8):.2f}, Leadership={row.get('leadership_role_weight', 0.6):.2f}
Market: Demand={row.get('demand_score', 0.5):.2f}, Trending={row.get('trending_score', 0.0):.2f}

Standards: ESCO={row.get('esco_uri', 'N/A')}, O*NET={row.get('onet_code', 'N/A')}
""".strip()

    return text[:MAX_TEXT_LEN]


def build_cluster_card(cluster_row, member_skills: list) -> str:
    """Build text representation of a skill cluster"""
    
    core_skills = [s for s in member_skills if s.get('is_core_skill')]
    optional_skills = [s for s in member_skills if not s.get('is_core_skill')]
    
    text = f"""
Cluster: {cluster_row['cluster_name']}
Type: {cluster_row['cluster_type']}
Description: {cluster_row.get('description', 'No description')}

Core Skills ({len(core_skills)}): {', '.join(s['skill_name'] for s in core_skills[:15])}

Optional Skills ({len(optional_skills)}): {', '.join(s['skill_name'] for s in optional_skills[:10])}

Total Skills: {len(member_skills)}
""".strip()
    
    return text[:MAX_TEXT_LEN]


# ============================================
# Embedding Generation
# ============================================

async def generate_batch_embeddings(client, texts: list) -> list:
    """Generate embeddings for a batch of texts using Gemini"""
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.models.embed_content(
                model=MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=DIMENSIONS
                )
            )
        )

        vectors = []
        for emb in result.embeddings:
            arr = np.array(emb.values)
            # Normalize to unit vector
            normalized = (arr / np.linalg.norm(arr)).tolist()
            vectors.append(normalized)

        return vectors
    
    except Exception as e:
        logger.error(f"❌ Error generating embeddings: {e}")
        raise


# ============================================
# Database Updates
# ============================================

async def update_skill_embedding(conn, skill_id: int, embedding: list, version: int = 1):
    """Update skill taxonomy embedding"""
    await conn.execute("""
        UPDATE skill_taxonomy
        SET embedding = $1::vector,
            embedding_model = $2,
            embedding_dimension = $3,
            embedding_version = $4,
            embedding_updated_at = $5,
            needs_embedding_update = FALSE
        WHERE id = $6
    """, 
        embedding,  # pgvector accepts Python list directly
        MODEL,
        DIMENSIONS,
        version,
        datetime.utcnow(),
        skill_id
    )


async def update_cluster_embedding(conn, cluster_id: int, embedding: list, version: int = 1):
    """Update skill cluster embedding"""
    await conn.execute("""
        UPDATE skill_clusters
        SET embedding = $1::vector,
            embedding_model = $2,
            embedding_version = $3,
            updated_at = $4
        WHERE id = $5
    """, 
        embedding,
        MODEL,
        version,
        datetime.utcnow(),
        cluster_id
    )


# ============================================
# Main Processing Functions
# ============================================

async def embed_skill_taxonomy(conn, client):
    """Generate embeddings for skill taxonomy"""
    
    logger.info("🔍 Fetching skills requiring embeddings...")
    
    # Fetch skills that need embeddings
    skills = await conn.fetch("""
        SELECT 
            st.id,
            st.skill_name,
            st.normalized_name,
            st.skill_type,
            st.category,
            st.subcategory,
            st.parent_skill,
            st.skill_level,
            st.related_skills,
            st.specializations,
            st.industry_relevance,
            st.role_relevance,
            st.base_weight,
            st.technical_role_weight,
            st.leadership_role_weight,
            st.demand_score,
            st.trending_score,
            st.esco_uri,
            st.onet_code,
            st.embedding_version
        FROM skill_taxonomy st
        WHERE st.embedding IS NULL 
           OR st.needs_embedding_update = TRUE
        ORDER BY st.id ASC
    """)
    
    if not skills:
        logger.info("✅ No skills require embedding updates")
        return
    
    total = len(skills)
    logger.info(f"📊 Found {total} skills requiring embeddings")
    
    # Fetch synonyms for all skills at once
    skill_ids = [s['id'] for s in skills]
    synonyms_map = {}
    
    if skill_ids:
        synonyms = await conn.fetch("""
            SELECT skill_taxonomy_id, 
                   array_agg(synonym) as synonyms
            FROM skill_synonyms
            WHERE skill_taxonomy_id = ANY($1::int[])
            GROUP BY skill_taxonomy_id
        """, skill_ids)
        
        for row in synonyms:
            synonyms_map[row['skill_taxonomy_id']] = row['synonyms']
    
    # Process in batches
    success_count = 0
    error_count = 0
    
    for i in range(0, total, BATCH_SIZE):
        batch = skills[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        
        logger.info(f"⚡ Processing batch {batch_num}/{(total + BATCH_SIZE - 1) // BATCH_SIZE} ({len(batch)} skills)")
        
        try:
            # Build text representations
            texts = []
            for skill in batch:
                skill_dict = dict(skill)
                synonyms_text = ", ".join(synonyms_map.get(skill['id'], []))
                text = build_skill_card(skill_dict, synonyms_text)
                texts.append(text)
            
            # Generate embeddings
            vectors = await generate_batch_embeddings(client, texts)
            
            # Update database
            for skill, vector in zip(batch, vectors):
                version = skill['embedding_version'] or 1
                await update_skill_embedding(conn, skill['id'], vector, version)
                success_count += 1
            
            logger.info(f"  ✅ Batch {batch_num} completed")
            
            # Rate limiting
            await asyncio.sleep(0.3)
            
        except Exception as e:
            logger.error(f"  ❌ Error in batch {batch_num}: {e}")
            error_count += len(batch)
    
    logger.info(f"\n📈 Skill Taxonomy Summary: {success_count} success, {error_count} errors")


async def embed_skill_clusters(conn, client):
    """Generate embeddings for skill clusters"""
    
    logger.info("\n🔍 Fetching skill clusters...")
    
    # Fetch clusters without embeddings
    clusters = await conn.fetch("""
        SELECT 
            sc.id,
            sc.cluster_name,
            sc.cluster_type,
            sc.description,
            sc.embedding_version
        FROM skill_clusters sc
        WHERE sc.embedding IS NULL
        ORDER BY sc.id ASC
    """)
    
    if not clusters:
        logger.info("✅ No clusters require embeddings")
        return
    
    total = len(clusters)
    logger.info(f"📊 Found {total} clusters requiring embeddings")
    
    success_count = 0
    error_count = 0
    
    for i in range(0, total, BATCH_SIZE):
        batch = clusters[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        
        logger.info(f"⚡ Processing cluster batch {batch_num}/{(total + BATCH_SIZE - 1) // BATCH_SIZE}")
        
        try:
            texts = []
            
            for cluster in batch:
                # Fetch member skills
                members = await conn.fetch("""
                    SELECT 
                        st.skill_name,
                        scm.is_core_skill,
                        scm.weight
                    FROM skill_cluster_members scm
                    JOIN skill_taxonomy st ON scm.skill_taxonomy_id = st.id
                    WHERE scm.cluster_id = $1
                    ORDER BY scm.is_core_skill DESC, scm.weight DESC
                """, cluster['id'])
                
                member_list = [dict(m) for m in members]
                text = build_cluster_card(dict(cluster), member_list)
                texts.append(text)
            
            # Generate embeddings
            vectors = await generate_batch_embeddings(client, texts)
            
            # Update database
            for cluster, vector in zip(batch, vectors):
                version = cluster['embedding_version'] or 1
                await update_cluster_embedding(conn, cluster['id'], vector, version)
                success_count += 1
            
            logger.info(f"  ✅ Cluster batch {batch_num} completed")
            await asyncio.sleep(0.3)
            
        except Exception as e:
            logger.error(f"  ❌ Error in cluster batch {batch_num}: {e}")
            error_count += len(batch)
    
    logger.info(f"\n📈 Skill Clusters Summary: {success_count} success, {error_count} errors")


# ============================================
# Statistics & Validation
# ============================================

async def print_statistics(conn):
    """Print embedding statistics"""
    
    logger.info("\n📊 Embedding Statistics:")
    
    # Skill taxonomy stats
    stats = await conn.fetchrow("""
        SELECT 
            COUNT(*) as total,
            COUNT(embedding) as with_embedding,
            COUNT(*) FILTER (WHERE needs_embedding_update = TRUE) as needs_update,
            COUNT(*) FILTER (WHERE status = 'ACTIVE') as active
        FROM skill_taxonomy
    """)
    
    logger.info(f"\n  Skill Taxonomy:")
    logger.info(f"    Total: {stats['total']}")
    logger.info(f"    With embeddings: {stats['with_embedding']} ({100*stats['with_embedding']/stats['total']:.1f}%)")
    logger.info(f"    Needs update: {stats['needs_update']}")
    logger.info(f"    Active: {stats['active']}")
    
    # Cluster stats
    cluster_stats = await conn.fetchrow("""
        SELECT 
            COUNT(*) as total,
            COUNT(embedding) as with_embedding
        FROM skill_clusters
    """)
    
    logger.info(f"\n  Skill Clusters:")
    logger.info(f"    Total: {cluster_stats['total']}")
    logger.info(f"    With embeddings: {cluster_stats['with_embedding']}")
    
    # Synonym stats
    synonym_count = await conn.fetchval("SELECT COUNT(*) FROM skill_synonyms")
    logger.info(f"\n  Synonyms: {synonym_count}")


# ============================================
# Main Entry Point
# ============================================

async def main():
    """Main execution function"""
    
    logger.info("🚀 Starting Enhanced Embedding Generation")
    logger.info(f"📝 Model: {MODEL}")
    logger.info(f"📏 Dimensions: {DIMENSIONS}")
    logger.info(f"📦 Batch size: {BATCH_SIZE}\n")
    
    # Initialize clients
    client = genai.Client(api_key=GEMINI_API_KEY)
    conn = await asyncpg.connect(DATABASE_URL)
    
    try:
        # Print initial stats
        await print_statistics(conn)
        
        # Process skill taxonomy
        await embed_skill_taxonomy(conn, client)
        
        # Process skill clusters
        await embed_skill_clusters(conn, client)
        
        # Print final stats
        await print_statistics(conn)
        
        logger.info("\n✅ All embeddings generated successfully!")
        
    except Exception as e:
        logger.error(f"\n💥 Fatal error: {e}")
        raise
    
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
