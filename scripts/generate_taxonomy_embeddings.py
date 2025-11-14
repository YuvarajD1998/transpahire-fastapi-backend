import os
import asyncio
import asyncpg
import logging
from datetime import datetime
from google import genai
from google.genai import types
import numpy as np
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = "postgresql://transpahire_user:transpahire_password_123@localhost:5432/transpahire_db"
GEMINI_API_KEY = "AIzaSyAdvFhvGWaIl_ZFlNo98-cBGGRMWd0uHs8"

MODEL = "gemini-embedding-001"
DIMENSIONS = 768
BATCH_SIZE = 50
MAX_TEXT_LEN = 1800


def build_skill_card(row):
    def join(v):
        return ", ".join(v) if v else "None"

    return f"""
{row['skill_name']}.
This is a skill at level {row['skill_level']} under category {row['parent_skill'] or 'None'}.
Synonyms: {join(row['synonyms'])}.
Related skills: {join(row['related_skills'])}.
Specializations: {join(row['specializations'])}.
Industries: {join(row['industry_relevance'])}.
Skill weight: {row['skill_weight']}.
""".strip()[:MAX_TEXT_LEN]


# ✅ Correct Batch Embedding Using THIS SDK
async def generate_batch_embeddings(client, texts):
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.models.embed_content(
            model=MODEL,
            contents=texts,  # ✅ LIST of strings
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=DIMENSIONS
            )
        )
    )

    # ✅ result.embeddings = list of embeddings
    vectors = []
    for emb in result.embeddings:
        arr = np.array(emb.values)
        normalized = (arr / np.linalg.norm(arr)).tolist()
        vectors.append(normalized)

    return vectors


# ✅ DB Update
async def update_row(conn, skill_id, embedding):
    await conn.execute("""
        UPDATE skill_taxonomy
        SET embedding = $1,
            embedding_model = $2,
            embedding_updated_at = $3,
            needs_embedding_update = FALSE
        WHERE id = $4
    """, json.dumps(embedding), MODEL, datetime.utcnow(), skill_id)


# ✅ Main process
async def embed_taxonomy():
    client = genai.Client(api_key=GEMINI_API_KEY)
    conn = await asyncpg.connect(DATABASE_URL)

    rows = await conn.fetch("""
        SELECT id, skill_name, parent_skill, skill_level,
               synonyms, related_skills, specializations,
               industry_relevance, skill_weight
        FROM skill_taxonomy
        WHERE embedding IS NULL OR needs_embedding_update = TRUE
        ORDER BY id ASC
    """)

    total = len(rows)
    logger.info(f"✅ Found {total} taxonomy rows requiring embeddings")

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i+BATCH_SIZE]
        texts = [build_skill_card(dict(r)) for r in batch]

        logger.info(f"⚡ Processing batch {i//BATCH_SIZE + 1} ({len(batch)} rows)")

        vectors = await generate_batch_embeddings(client, texts)

        for row, vector in zip(batch, vectors):
            await update_row(conn, row["id"], vector)

        await asyncio.sleep(0.2)  # smooth rate limiting

    logger.info("✅ All embeddings generated successfully!")
    await conn.close()


if __name__ == "__main__":
    asyncio.run(embed_taxonomy())
