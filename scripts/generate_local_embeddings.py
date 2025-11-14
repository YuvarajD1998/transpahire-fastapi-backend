import os
import asyncio
import asyncpg
import numpy as np
import json
import logging
from datetime import datetime
from tqdm.asyncio import tqdm
from sentence_transformers import SentenceTransformer

# ============================================
# CONFIGURATION
# ============================================
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://transpahire_user:transpahire_password_123@localhost:5432/transpahire_db"
)
MODEL_NAME = "intfloat/e5-base-v2"
EMBED_DIM = 768
BATCH_SIZE = 64

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("embedding-generator")

# ============================================
# BUILD TEXT DESCRIPTION
# ============================================

def build_skill_card(row, synonyms, related, specializations):
    """Constructs a rich semantic text for each skill row."""
    def fmt_dict(d):
        if not d:
            return "None"
        if isinstance(d, str):
            try:
                d = json.loads(d)
            except Exception:
                return d
        return ", ".join(f"{k}({v:.2f})" for k, v in list(d.items())[:5])

    text = f"""
Skill: {row['skill_name']}
Type: {row.get('skill_type', 'TECHNICAL')}
Category: {row.get('category', 'N/A')} > {row.get('subcategory', 'N/A')}
Level: {row.get('skill_level', 0)} (Parent: {row.get('parent_skill', 'None')})

Synonyms: {', '.join(synonyms) if synonyms else 'None'}
Related Skills: {', '.join(related) if related else 'None'}
Specializations: {', '.join(specializations) if specializations else 'None'}

Industry Relevance: {fmt_dict(row.get('industry_relevance'))}
Role Relevance: {fmt_dict(row.get('role_relevance'))}

Weights:
  Base={row.get('base_weight', 0.5):.2f},
  Technical={row.get('technical_role_weight', 0.8):.2f},
  Leadership={row.get('leadership_role_weight', 0.6):.2f},
  Managerial={row.get('managerial_role_weight', 0.7):.2f}

Market:
  Demand={row.get('demand_score', 0.5):.2f},
  Trending={row.get('trending_score', 0.0):.2f}

Ontology:
  ESCO={row.get('esco_uri', 'N/A')},
  O*NET={row.get('onet_code', 'N/A')}
""".strip()

    return text[:2000]


# ============================================
# MAIN EMBEDDING FUNCTION
# ============================================

async def embed_taxonomy():
    logger.info(f"🚀 Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    logger.info("✅ Model loaded successfully")

    conn = await asyncpg.connect(DATABASE_URL)

    logger.info("🔍 Fetching skills that need embeddings...")
    rows = await conn.fetch("""
        SELECT 
            st.id,
            st.skill_name,
            st.skill_type,
            st.category,
            st.subcategory,
            st.parent_id,
            p.skill_name AS parent_skill,
            st.skill_level,
            st.industry_relevance,
            st.role_relevance,
            st.base_weight,
            st.technical_role_weight,
            st.leadership_role_weight,
            st.managerial_role_weight,
            st.demand_score,
            st.trending_score,
            st.esco_uri,
            st.onet_code,
            st.embedding_version
        FROM skill_taxonomy st
        LEFT JOIN skill_taxonomy p ON st.parent_id = p.id
        WHERE st.embedding IS NULL OR st.needs_embedding_update = TRUE
        ORDER BY st.id ASC
    """)

    total = len(rows)
    if not total:
        logger.info("✅ No skills need embedding updates.")
        await conn.close()
        return

    logger.info(f"📊 Found {total} skills requiring embeddings")

    # Preload synonyms, related skills, and specializations
    skill_ids = [r["id"] for r in rows]
    synonyms_map, related_map, spec_map = await fetch_skill_metadata(conn, skill_ids)

    # Process in batches
    for i in tqdm(range(0, total, BATCH_SIZE), desc="Embedding batches"):
        batch = rows[i:i + BATCH_SIZE]
        texts = []
        for r in batch:
            sid = r["id"]
            synonyms = synonyms_map.get(sid, [])
            related = related_map.get(sid, [])
            specs = spec_map.get(sid, [])
            text = build_skill_card(dict(r), synonyms, related, specs)
            texts.append(text)

        # Generate embeddings
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        # Update DB
        for r, emb in zip(batch, embeddings):
            # Convert numpy array → PostgreSQL vector literal
            vector_str = "[" + ",".join(f"{x:.6f}" for x in emb.tolist()) + "]"

            await conn.execute("""
                UPDATE skill_taxonomy
                SET embedding = $1::vector,
                    embedding_model = $2,
                    embedding_dimension = $3,
                    embedding_updated_at = $4,
                    needs_embedding_update = FALSE
                WHERE id = $5
            """, vector_str, MODEL_NAME, EMBED_DIM, datetime.utcnow(), r["id"])


    logger.info("✅ All embeddings generated successfully!")
    await conn.close()


# ============================================
# FETCH SYNONYMS & RELATIONS
# ============================================

async def fetch_skill_metadata(conn, skill_ids):
    """Fetch all synonyms, related skills, and specializations for given skill IDs."""
    synonyms_map = {}
    related_map = {}
    spec_map = {}

    # Synonyms
    synonyms = await conn.fetch("""
        SELECT skill_taxonomy_id, array_agg(synonym) AS synonyms
        FROM skill_synonyms
        WHERE skill_taxonomy_id = ANY($1::int[])
        GROUP BY skill_taxonomy_id
    """, skill_ids)
    for s in synonyms:
        synonyms_map[s["skill_taxonomy_id"]] = s["synonyms"]

    # Related Skills
    related = await conn.fetch("""
        SELECT sr.source_skill_id, array_agg(st.skill_name) AS related
        FROM skill_relations sr
        JOIN skill_taxonomy st ON sr.target_skill_id = st.id
        WHERE sr.source_skill_id = ANY($1::int[])
          AND sr.relation_type IN ('SIMILAR_TO', 'COMMONLY_WITH')
        GROUP BY sr.source_skill_id
    """, skill_ids)
    for r in related:
        related_map[r["source_skill_id"]] = r["related"]

    # Specializations
    specs = await conn.fetch("""
        SELECT sr.source_skill_id, array_agg(st.skill_name) AS specs
        FROM skill_relations sr
        JOIN skill_taxonomy st ON sr.target_skill_id = st.id
        WHERE sr.source_skill_id = ANY($1::int[])
          AND sr.relation_type = 'SPECIALIZATION_OF'
        GROUP BY sr.source_skill_id
    """, skill_ids)
    for s in specs:
        spec_map[s["source_skill_id"]] = s["specs"]

    return synonyms_map, related_map, spec_map


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    asyncio.run(embed_taxonomy())
