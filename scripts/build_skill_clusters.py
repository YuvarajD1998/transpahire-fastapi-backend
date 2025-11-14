#!/usr/bin/env python3
"""
build_skill_clusters.py

- Clusters existing skill embeddings per category/subcategory.
- Inserts rows into `skill_clusters` and `skill_cluster_members`.
- Skips SkillEndorsement (user requested).
- Uses HDBSCAN (preferred) or KMeans fallback.

Behavior:
- This script does NOT insert updated_at (Option A). Ensure your DB has:
    ALTER TABLE skill_clusters ALTER COLUMN updated_at SET DEFAULT NOW();
  (so updated_at isn't NULL on inserts)
- The script expects skill_taxonomy.embedding column to contain vectors (pgvector or array literal).
"""
import os
import argparse
import asyncio
import asyncpg
import json
import math
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
from tqdm import tqdm

# ML libs
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine as scipy_cosine

# hdbscan may be heavy but preferred
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False

# UMAP for dimensionality reduction (optional)
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# ---------- logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("skill-clusterer")

# ---------- defaults ----------
DEFAULT_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://transpahire_user:transpahire_password_123@localhost:5432/transpahire_db"
)
DEFAULT_MIN_CLUSTER_SIZE = 12
DEFAULT_BATCH_SIZE = 512  # DB insert batch size for members
MIN_SKIP_BUCKET = 5       # buckets with fewer than this will be skipped entirely

# ---------- helper utils ----------
def normalize_embedding(raw) -> Optional[np.ndarray]:
    """
    Normalize various embedding representations returned by asyncpg into numpy array.
    Handles:
      - list/tuple of floats
      - memoryview / bytes (if pgvector returns), tries to decode JSON-like
      - string like '[0.123, 0.234, ...]'
    """
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        return raw
    if isinstance(raw, (list, tuple)):
        try:
            arr = np.array(raw, dtype=float)
            return arr
        except Exception:
            return None
    if isinstance(raw, str):
        try:
            s = raw.strip()
            if s.startswith('['):
                lst = json.loads(s)
                return np.array(lst, dtype=float)
            # comma separated?
            parts = [p.strip() for p in s.split(',') if p.strip()]
            return np.array([float(p) for p in parts], dtype=float)
        except Exception:
            return None
    # fallback: try to convert bytes
    try:
        b = bytes(raw)
        s = b.decode('utf-8')
        if s.startswith('['):
            return np.array(json.loads(s), dtype=float)
    except Exception:
        pass
    return None

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity in [-1,1]. If zero vectors, fallback to 0."""
    if a is None or b is None:
        return 0.0
    try:
        # ensure shapes
        a = np.asarray(a, dtype=float).reshape(1, -1)
        b = np.asarray(b, dtype=float).reshape(1, -1)
        sim = cosine_similarity(a, b)[0, 0]
        if np.isnan(sim):
            return 0.0
        return float(sim)
    except Exception:
        # fallback
        try:
            return 1.0 - scipy_cosine(np.asarray(a).ravel(), np.asarray(b).ravel())
        except Exception:
            return 0.0

def vector_literal(arr: np.ndarray) -> str:
    """Convert numpy vector to Postgres vector literal string: '[0.123,0.456,...]'"""
    return "[" + ",".join(f"{float(x):.6f}" for x in arr.tolist()) + "]"

# ---------- DB helpers ----------
async def fetch_categories(conn: asyncpg.Connection) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Returns distinct (category, subcategory) pairs where embeddings exist.
    """
    rows = await conn.fetch("""
        SELECT DISTINCT category, subcategory
        FROM skill_taxonomy
        WHERE embedding IS NOT NULL
        ORDER BY category NULLS LAST, subcategory NULLS LAST
    """)
    pairs = [(r['category'], r['subcategory']) for r in rows]
    return pairs

async def fetch_skill_rows(conn: asyncpg.Connection, category: Optional[str], subcategory: Optional[str]):
    """
    Fetch skill rows (id, normalized_name, skill_name, embedding, demand_score, trending_score)
    filtered by category/subcategory.
    """
    if category is None:
        q = """
            SELECT id, normalized_name, skill_name, embedding, demand_score, trending_score
            FROM skill_taxonomy
            WHERE embedding IS NOT NULL
            ORDER BY id
        """
        rows = await conn.fetch(q)
    else:
        if subcategory is None:
            q = """
                SELECT id, normalized_name, skill_name, embedding, demand_score, trending_score
                FROM skill_taxonomy
                WHERE category = $1 AND embedding IS NOT NULL
                ORDER BY id
            """
            rows = await conn.fetch(q, category)
        else:
            q = """
                SELECT id, normalized_name, skill_name, embedding, demand_score, trending_score
                FROM skill_taxonomy
                WHERE category = $1 AND subcategory = $2 AND embedding IS NOT NULL
                ORDER BY id
            """
            rows = await conn.fetch(q, category, subcategory)
    return rows

async def insert_cluster(conn: asyncpg.Connection, cluster_name: str, description: str, cluster_type: str,
                         centroid: np.ndarray, embedding_model: str, embedding_version: int) -> int:
    """
    Inserts into skill_clusters and returns cluster id.
    NOTE: This insert intentionally does NOT set updated_at (Option A).
    Ensure your DB has updated_at default: ALTER TABLE skill_clusters ALTER COLUMN updated_at SET DEFAULT now();
    """
    vec = vector_literal(centroid)
    r = await conn.fetchrow("""
        INSERT INTO skill_clusters (
            cluster_name,
            description,
            cluster_type,
            embedding,
            embedding_model,
            embedding_version,
            created_at
        )
        VALUES ($1, $2, $3, $4::vector, $5, $6, NOW())
        RETURNING id;
    """, cluster_name, description, cluster_type, vec, embedding_model, embedding_version)
    return r['id']

async def insert_cluster_members_batch(conn: asyncpg.Connection, rows: List[Tuple[int,int,float,bool]]):
    """
    rows: list of (cluster_id, skill_taxonomy_id, weight, is_core_skill)
    Uses executemany to insert/update members.
    """
    if not rows:
        return
    
    # Remove timestamp from payload since columns don't exist
    try:
        await conn.executemany("""
            INSERT INTO skill_cluster_members
              (cluster_id, skill_taxonomy_id, weight, is_core_skill)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (cluster_id, skill_taxonomy_id) DO UPDATE
              SET weight = EXCLUDED.weight,
                  is_core_skill = EXCLUDED.is_core_skill
        """, rows)
    except Exception as e:
        logger.warning("Bulk insert failed, falling back to single inserts: %s", e)
        for (c, s, w, ic) in rows:
            try:
                await conn.execute("""
                    INSERT INTO skill_cluster_members
                      (cluster_id, skill_taxonomy_id, weight, is_core_skill)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (cluster_id, skill_taxonomy_id) DO UPDATE
                      SET weight = EXCLUDED.weight,
                          is_core_skill = EXCLUDED.is_core_skill
                """, c, s, w, ic)
            except Exception as e2:
                logger.error("Failed to insert member %s: %s", (c, s), e2)


# ---------- clustering logic ----------
def run_hdbscan_cluster(X: np.ndarray, min_cluster_size: int = 12) -> np.ndarray:
    """Return labels array (len = n_samples). Noise label = -1."""
    if not HDBSCAN_AVAILABLE:
        raise RuntimeError("HDBSCAN not available")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=None, prediction_data=False)
    labels = clusterer.fit_predict(X)
    return labels

def run_kmeans_cluster(X: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be > 0")
    k = min(k, X.shape[0])
    if k == 1:
        # all in one cluster
        return np.zeros(X.shape[0], dtype=int)
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = km.fit_predict(X)
    return labels

def reduce_dim_if_needed(X: np.ndarray, n_components: int = 64) -> np.ndarray:
    if X.shape[1] <= n_components or not UMAP_AVAILABLE:
        return X
    try:
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        Xr = reducer.fit_transform(X)
        return Xr
    except Exception as e:
        logger.warning("UMAP failed; continuing with original dims: %s", e)
        return X

# ---------- weight formula ----------
def compute_weights_for_cluster(vectors: np.ndarray, centroid: np.ndarray, demand_scores: List[float],
                                trending_scores: List[float], alpha=0.7, beta=0.2, gamma=0.1) -> np.ndarray:
    """
    vectors: NxD, centroid: D
    demand_scores/trending_scores: lists length N (may contain None)
    Returns normalized weights in 0..1
    """
    sims = []
    for v in vectors:
        s = cosine(v, centroid)  # in [-1,1]
        # Normalize to [0,1]
        s01 = (s + 1.0) / 2.0
        sims.append(s01)
    sims = np.array(sims, dtype=float)

    ds = np.array([float(x) if x is not None else 0.0 for x in demand_scores], dtype=float)
    ts_raw = np.array([float(x) if x is not None else 0.0 for x in trending_scores], dtype=float)
    # trending_score expected -1..1 -> normalize to 0..1
    ts = (ts_raw + 1.0) / 2.0

    raw = alpha * sims + beta * ds + gamma * ts
    # normalize 0..1
    if raw.max() > 0:
        norm = (raw - raw.min()) / (raw.max() - raw.min())
    else:
        norm = raw
    return np.clip(norm, 0.0, 1.0)

# ---------- main pipeline ----------
async def process_bucket(conn: asyncpg.Connection, category: Optional[str], subcategory: Optional[str],
                         min_cluster_size: int, embedding_model: str, embedding_version: int,
                         dry_run: bool, batch_size: int):
    label = f"{category or 'ALL'} / {subcategory or 'ALL'}"
    logger.info("Processing bucket: %s", label)

    rows = await fetch_skill_rows(conn, category, subcategory)
    n = len(rows)
    logger.info("Found %d skills with embeddings in this bucket", n)

    # CASE 1: No skills
    if n == 0:
        logger.info("No skills in this bucket — skipping.")
        return

    # CASE 2: Extremely small buckets - skip entirely
    if n < MIN_SKIP_BUCKET:
        logger.info("Too few skills (%d). Skipping bucket.", n)
        return

    # CASE 3: Small bucket (MIN_SKIP_BUCKET <= n < min_cluster_size) → make 1 cluster
    if n < min_cluster_size:
        logger.info("Small bucket (%d < %d). Creating one single cluster directly.", n, min_cluster_size)

        usable_rows = []
        embeddings = []
        demand_scores = []
        trending_scores = []
        for r in rows:
            emb = normalize_embedding(r['embedding'])
            if emb is None:
                continue
            usable_rows.append(r)
            embeddings.append(emb)
            demand_scores.append(r.get('demand_score'))
            trending_scores.append(r.get('trending_score'))

        if not embeddings:
            logger.info("No usable embeddings after normalization in small bucket — skipping.")
            return

        X = np.stack(embeddings, axis=0)
        centroid = X.mean(axis=0)

        weights = compute_weights_for_cluster(X, centroid, demand_scores, trending_scores)

        is_core = weights >= 0.8
        top_k = max(1, int(len(weights) * 0.2))
        top_idx = np.argsort(weights)[-top_k:]
        is_core[top_idx] = True

        cluster_name = f"AUTO_CLUSTER: {category or 'ALL'} / {subcategory or 'ALL'}"
        description = f"Auto-generated single cluster (bucket too small: n={len(weights)})"
        cluster_type = "THEMATIC"

        if dry_run:
            cluster_id = -1
            logger.info("[DRY RUN] Would create cluster: %s (size=%d)", cluster_name, len(weights))
        else:
            cluster_id = await insert_cluster(conn, cluster_name, description, cluster_type, centroid, embedding_model, embedding_version)
            logger.info("Created cluster id=%d: %s (size=%d)", cluster_id, cluster_name, len(weights))

        member_rows = [(cluster_id, r['id'], float(w), bool(core)) for r, w, core in zip(usable_rows, weights, is_core)]

        if not dry_run:
            for i in range(0, len(member_rows), batch_size):
                await insert_cluster_members_batch(conn, member_rows[i:i+batch_size])

        logger.info("✔️ Created SINGLE cluster for small bucket.")
        return

    # Collect embeddings and metadata for larger buckets
    ids = []
    names = []
    embeddings = []
    demand_scores = []
    trending_scores = []
    for r in rows:
        emb = normalize_embedding(r['embedding'])
        if emb is None:
            continue
        ids.append(r['id'])
        names.append(r.get('normalized_name') or r.get('skill_name'))
        embeddings.append(emb)
        demand_scores.append(r.get('demand_score'))
        trending_scores.append(r.get('trending_score'))

    if not embeddings:
        logger.info("No usable embeddings after normalization in bucket %s", label)
        return

    X = np.stack(embeddings, axis=0)
    D = X.shape[1]

    # Optional reduction for speed if many rows
    if X.shape[0] > 2000 and UMAP_AVAILABLE:
        logger.info("Reducing dimensionality with UMAP for speed (n=%d -> dim<=64)...", X.shape[0])
        Xr = reduce_dim_if_needed(X, n_components=min(64, D))
    else:
        Xr = X

    # Choose clustering algorithm
    labels = None
    try:
        if HDBSCAN_AVAILABLE:
            logger.info("Running HDBSCAN (min_cluster_size=%d)...", min_cluster_size)
            labels = run_hdbscan_cluster(Xr, min_cluster_size=min_cluster_size)
            # If HDBSCAN returns only noise -> skip
            if set(labels.tolist()) == {-1}:
                logger.info("HDBSCAN returned only noise — switching to KMeans fallback.")

                # Compute fallback k
                n_samples = X.shape[0]
                k = max(2, round(n_samples / 20))

                logger.info("Running fallback KMeans with k=%d", k)
                labels = run_kmeans_cluster(Xr, k=k)

        else:
            raise RuntimeError("HDBSCAN not available")
    except Exception as e:
        logger.warning("HDBSCAN failed or unavailable: %s. Falling back to KMeans heuristic.", e)
        k_guess = max(2, round(X.shape[0] / 50))
        k = min(k_guess, X.shape[0])
        logger.info("KMeans with k=%d", k)
        labels = run_kmeans_cluster(Xr, k=k)

    unique_labels = sorted(set(labels.tolist()))
    logger.info("Clustering produced %d labels (including noise). Unique labels: %s", len(unique_labels), unique_labels[:10])

    # Map label -> member indices
    label_to_indices: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        label_to_indices.setdefault(int(lab), []).append(idx)

    # For each cluster label (ignore small noise groups)
    created_clusters = 0
    for lab in sorted(label_to_indices.keys()):
        indices = label_to_indices[lab]
        if lab == -1:
            # skip tiny noise groups
            if len(indices) < (min_cluster_size // 2):
                logger.info("Skipping small noise group (size=%d)", len(indices))
                continue
            # else treat noise group as cluster
            logger.info("Treating large noise group as cluster (size=%d)", len(indices))

        cluster_vecs = X[np.array(indices)]
        centroid = np.mean(cluster_vecs, axis=0)

        cluster_demands = [demand_scores[i] for i in indices]
        cluster_trends = [trending_scores[i] for i in indices]
        weights = compute_weights_for_cluster(cluster_vecs, centroid, cluster_demands, cluster_trends)

        # determine core skills
        top_k = max(1, math.ceil(len(weights) * 0.05))
        top_indices = np.argsort(weights)[-top_k:]
        is_core = [False] * len(weights)
        for j, w in enumerate(weights):
            if w >= 0.8:
                is_core[j] = True
        for ti in top_indices:
            is_core[ti] = True

        cluster_name = f"AUTO: {category or 'ALL'} / {subcategory or 'ALL'} - {lab}"
        description = f"Auto-generated cluster for {category or 'ALL'} / {subcategory or 'ALL'} (label={lab}, size={len(indices)})."
        cluster_type = "THEMATIC"

        if dry_run:
            created_cluster_id = -1 * (lab + 1)
            logger.info("[DRY RUN] Would create cluster: %s (size=%d)", cluster_name, len(indices))
        else:
            try:
                created_cluster_id = await insert_cluster(conn, cluster_name, description, cluster_type, centroid, embedding_model, embedding_version)
                logger.info("Created cluster id=%d: %s (size=%d)", created_cluster_id, cluster_name, len(indices))
            except Exception as e:
                logger.exception("Failed to create cluster for label %s: %s", lab, e)
                continue

        # Build and insert members
        member_rows = []
        for local_idx, idx_in_bucket in enumerate(indices):
            skill_id = ids[idx_in_bucket]
            w = float(weights[local_idx])
            ic = bool(is_core[local_idx])
            member_rows.append((created_cluster_id, skill_id, w, ic))

        if dry_run:
            logger.info("[DRY RUN] Would insert %d cluster members for cluster %s", len(member_rows), cluster_name)
        else:
            for start in range(0, len(member_rows), batch_size):
                batch = member_rows[start:start+batch_size]
                await insert_cluster_members_batch(conn, batch)

        created_clusters += 1

    logger.info("Bucket %s done. Created %d clusters.", label, created_clusters)

# ---------- main entry ----------
async def main():
    parser = argparse.ArgumentParser(description="Build Skill Clusters from existing embeddings")
    parser.add_argument("--db", type=str, default=DEFAULT_DB_URL, help="Postgres DATABASE_URL")
    parser.add_argument("--category", type=str, default=None, help="Process only this category")
    parser.add_argument("--subcategory", type=str, default=None, help="Process only this subcategory")
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE, help="Min cluster size for HDBSCAN")
    parser.add_argument("--embedding-model", type=str, default="intfloat/e5-base-v2", help="Embedding model name meta")
    parser.add_argument("--embedding-version", type=int, default=1, help="Embedding version meta")
    parser.add_argument("--dry-run", action="store_true", help="Do not insert into DB; only simulate")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Insert batch size for members")
    args = parser.parse_args()

    logger.info("Starting clustering pipeline. HDBSCAN available: %s, UMAP available: %s", HDBSCAN_AVAILABLE, UMAP_AVAILABLE)
    conn = await asyncpg.connect(args.db)

    try:
        # If specific category provided, process only that bucket
        if args.category:
            buckets = [(args.category, args.subcategory)]
        else:
            buckets = await fetch_categories(conn)

        logger.info("Buckets to process: %d", len(buckets))
        for (cat, subcat) in buckets:
            await process_bucket(conn, cat, subcat, args.min_cluster_size, args.embedding_model, args.embedding_version, args.dry_run, args.batch_size)

    finally:
        await conn.close()
        logger.info("DB connection closed. Pipeline finished.")

if __name__ == "__main__":
    asyncio.run(main())
