# TranspaHire FastAPI Backend

FastAPI microservice for resume parsing, profile management, and embedding generation; paired with a NestJS primary backend that owns auth and the Prisma DB.

## Run / Test

```bash
# Activate venv first (always required)
source transpahire_env/bin/activate

# Dev server (hot reload requires DEBUG=True in .env)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Production
python app/main.py

# Docs only available when DEBUG=True
# http://localhost:8000/docs
```

No test suite exists yet. Test manually via `/docs` with `DEBUG=True`.

## LLM Fallback Chain

Order is fixed — never reorder:
1. **Gemini** (primary) — `gemini_service.py`
2. **OpenAI** (first fallback) — `openai_service.py`
3. **HuggingFace** (second fallback) — `huggingface_service.py`
4. **Regex** — `_enhanced_fallback_parse()` in `file_service.py`
5. **Minimal** — bare `ParsedResumeData` with `confidence_score=0.1`

## Coding Rules

- **Routers are thin.** No business logic in routers. All logic lives in services.
- **All Gemini SDK calls are blocking** — must be wrapped in `loop.run_in_executor(None, ...)`. See `_generate_content_async()` and `gemini_service_embedding.py`. Calling them directly will block the event loop.
- **`parse-resume` must never raise HTTP 500.** The router catches all exceptions and returns `ParseResponse(success=False, ...)`. Do not let any exception propagate out of that handler.
- **e5-base-v2 requires a prefix on every text.** `embedding_service_local.py` prepends `"query: "` or a caller-supplied prefix to every string before encoding. Omitting the prefix silently degrades embedding quality.
- **Text truncation limits are not negotiable:**
  - Embeddings: `text[:2000]` (both local e5 and Gemini embedding services)
  - Resume parsing: `text[:settings.RESUME_TEXT_CHAR_CAP]` (currently `15000` chars, not `50000`)
- **Skills cap at 40.** Enforced in `GeminiService._normalize_parsed_data()` and `SkillCleaner.clean()`. Do not raise this without updating both places.
- **`unstructured` PDF strategy is `fast`, not `hi_res`.** `hi_res` loads Detectron2/YOLO (4–8 GB RAM) and OOMs on this machine. Do not change the strategy.

## NEVER Do

- **Do not fix `get_user_from_database()`** in `dependencies.py`. It returns hardcoded mock data. This is intentional and tracked — a real DB query will replace it when NestJS auth integration is complete.
- **Do not reorder the LLM fallback chain.** Each step exists for cost and reliability reasons.
- **Do not add `use_hi_res=True`** anywhere in `file_service.py`. See RAM note above.
- **Do not call Gemini SDK methods outside `run_in_executor`.** The SDK is synchronous and will stall the async event loop.

## Active Gotchas

- `RESUME_TEXT_CHAR_CAP` in `config.py` is `15000`, but comments elsewhere in the codebase say `50000`. `15000` is the current truth.
- `gemini_service_embedding.py` and `gemini_service.py` are two separate classes both named `GeminiService` — one for parsing, one for embeddings. Import carefully.
- `generate_embedding()` in `embeddings_local.py` calls `embedding_service.generate_embedding(..., normalize=True)` but the `EmbeddingService` signature uses `prefix_type`, not `normalize`. This mismatch exists; do not add a `normalize` param silently.
- JWT auth decodes without strict subject validation (`"verify_sub": False`) because NestJS issues integer user IDs that fail PyJWT's default string check.
- `MultiVectorEmbeddingService.CANDIDATE_EMBEDDING_TYPES` has exactly 5 types. The `len(existing) >= 5` guard in `generate_candidate_embeddings()` depends on this count. Adding a new type requires updating both the list **and** the count check.

## WIP — Do Not Touch

- `get_user_from_database()` in `dependencies.py` — mock stub, intentionally incomplete.
- `app/services/unstructured_service.py` — appears to be a duplicate/earlier draft of `UnstructuredExtractor` inside `file_service.py`; not imported anywhere. Do not delete or merge yet.
- `app/routers/embeddings.py` (the Gemini-backed one) — partially wired; the multi-vector flow through `MultiVectorEmbeddingService` is not fully tested end-to-end.
