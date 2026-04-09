# TranspaHire – FastAPI LLM & Embedding Microservice

> **Maintenance note:** Update only the relevant section when code changes. Do not rewrite the full document.

---

## 1. Project Overview

### Purpose
A Python microservice within the TranspaHire platform. The NestJS backend handles core business logic; this service handles all AI-heavy work:

- **Resume parsing** – extract structured candidate data from uploaded files using LLMs
- **Embedding generation** – produce semantic vectors for candidates and jobs to power downstream matching

### Core Responsibilities
- Accept file uploads (PDF, DOCX, images) and return structured `ParsedResumeData` JSON
- Generate local embeddings (sentence-transformers, no API key required)
- Generate multi-vector embeddings (Gemini API) for candidates and jobs, stored in pgvector
- Validate JWT tokens issued by the NestJS service for protected routes

### High-level Architecture
```
Client / NestJS
      │
      ▼
FastAPI (this service)
      │
      ├── /resumes/parse-resume        → Unstructured → Gemini / OpenAI / HF / regex
      ├── /embeddings/generate-*       → sentence-transformers (local, no auth)
      └── /embeddings/candidates|jobs  → Gemini API → pgvector DB (JWT required)
```

---

## 2. Tech Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI (async) |
| Language | Python 3.10+ |
| Primary LLM | Google Gemini (`gemini-1.5-flash`) |
| Embedding (remote) | Gemini `gemini-embedding-001` (768-dim) |
| Embedding (local) | `sentence-transformers` / `intfloat/e5-base-v2` (768-dim) |
| Document extraction | `unstructured` (primary), PyPDF2 / python-docx / pytesseract (legacy fallbacks) |
| Vector DB | PostgreSQL + `pgvector` extension |
| ORM | SQLAlchemy async (`asyncpg` driver) |
| Image preprocessing | Pillow |
| Auth | PyJWT (HS256, validates tokens from NestJS) |
| Rate limiting | `slowapi` |
| Config | `pydantic-settings` |
| CORS / GZip | FastAPI middleware |

---

## 3. Folder Structure

```
app/
├── main.py                  # App factory, router registration, middleware setup
├── config.py                # Settings loaded from .env via pydantic-settings
├── database.py              # Async SQLAlchemy engine + session factory (DatabaseManager)
├── dependencies.py          # JWT auth (get_current_user), subscription guard
│
├── routers/                 # FastAPI route handlers (thin – delegate to services)
│   ├── health.py            # GET /health
│   ├── resumes.py           # POST /resumes/parse-resume
│   ├── embeddings_local.py  # POST /embeddings/generate-embedding|generate-batch-embeddings
│   └── embeddings.py        # POST|GET /embeddings/candidates|jobs (auth required)
│
├── services/                # Business logic
│   ├── file_service.py      # ResumeParserService + UnstructuredExtractor + FileService
│   ├── gemini_service.py    # Gemini LLM – resume parsing (primary)
│   ├── openai_service.py    # OpenAI LLM – resume parsing (first fallback)
│   ├── huggingface_service.py  # HuggingFace LLM – resume parsing (second fallback)
│   ├── unstructured_service.py # Standalone unstructured wrapper (unused directly)
│   ├── embedding_service_local.py    # Local sentence-transformer embeddings (singleton)
│   ├── multi_vector_embedding_service.py  # Gemini multi-vector embeddings (DB-backed)
│   ├── gemini_service_embedding.py   # Gemini embedding API client
│   └── representation_service.py    # Text generation for each embedding type
│
├── models/                  # Data models
│   ├── schemas.py           # Pydantic request/response schemas
│   ├── embedding_models.py  # SQLAlchemy ORM: CandidateEmbedding, JobEmbedding
│   ├── database_models.py   # SQLAlchemy ORM: Profile, WorkExperience, Education, etc.
│   └── enums.py             # ParseStatus, ProficiencyLevel, SkillSource, Role, etc.
│
├── crud/
│   └── resume_crud.py       # ProfileCRUD, WorkExperienceCRUD, EducationCRUD, ProfileSkillCRUD
│
├── middleware/
│   └── errorhandler.py      # ErrorHandlerMiddleware + setup_exception_handlers
│
└── utils/
    ├── file_utils.py        # validate_upload() – checks file type & size
    ├── json_utils.py        # JSON helpers
    └── text_utils.py        # Text processing helpers
```

---

## 4. API Documentation

Base prefix: `/api/v1`

---

### `GET /health`
- **Auth:** None
- **Response:** `{"status": "ok"}`

---

### `POST /resumes/parse-resume`
- **Auth:** None
- **Content-Type:** `multipart/form-data`

**Request fields:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | File | Yes | – | Resume file (PDF, DOCX, DOC, JPG, PNG, TIFF, BMP) |
| `resume_id` | int | Yes | – | ID assigned by caller (returned in response for correlation) |
| `enhance_images` | bool | No | `true` | Apply OCR preprocessing (contrast/sharpness boost) |

**Response: `ParseResponse`**
```json
{
  "success": true,
  "resume_id": 42,
  "parsed_data": {
    "personal_info": { "name": "...", "email": "...", "phone": "...", "linkedin": "...", "github": "..." },
    "summary": "...",
    "skills": {
      "technical_skills": [{ "name": "Python", "type": "TECHNICAL", "group": "Programming", ... }],
      "soft_skills": [{ "name": "Communication", ... }]
    },
    "experience": [{ "company": "...", "position": "...", "start_date": "YYYY-MM-DD", ... }],
    "education": [{ "institution": "...", "degree": "...", "field": "...", ... }],
    "certifications": [...],
    "projects": [...],
    "languages": [...],
    "confidence_score": 0.95
  },
  "error": null,
  "confidence_score": 0.95
}
```

**Internal flow:**
1. `ResumeParserService.parse()` →
2. `UnstructuredExtractor.extract_with_unstructured()` → raw text (or legacy PyPDF2/docx/OCR fallback) →
3. `GeminiService.parse_resume_text()` → structured JSON →
4. Parsed into `ParsedResumeData` and returned

**On failure:** Returns `ParseResponse(success=False, error="...", confidence_score=0.0)` — does **not** raise HTTP 500.

---

### `POST /embeddings/generate-embedding`
- **Auth:** None
- **Content-Type:** `application/json`

**Request: `EmbeddingRequest`**
```json
{ "text": "Skill: Python\nContext: Used for backend development with FastAPI" }
```
- `text`: 1–5000 characters

**Response: `EmbeddingResponse`**
```json
{ "success": true, "embedding": [0.012, -0.034, ...], "dimension": 768, "error": null }
```

**Internal flow:** `EmbeddingService.generate_embedding()` → prepends `"query: "` prefix → `SentenceTransformer("intfloat/e5-base-v2")` → 768-dim normalized vector.

---

### `POST /embeddings/generate-batch-embeddings`
- **Auth:** None
- **Content-Type:** `application/json`

**Request: `BatchEmbeddingRequest`**
```json
{ "texts": ["text 1", "text 2", "..."] }
```
- `texts`: 1–100 items

**Response: `BatchEmbeddingResponse`**
```json
{ "success": true, "embeddings": [[...], [...]], "dimension": 768, "count": 2, "error": null }
```

**Internal flow:** `EmbeddingService.generate_batch_embeddings()` → batch_size=32 → returns list of 768-dim vectors.

---

### `POST /embeddings/candidates/{candidate_id}/generate`
- **Auth:** Bearer JWT (CANDIDATE role)
- **Query params:** `regenerate: bool = false`

**Response:**
```json
{ "message": "Embedding generation started", "candidate_id": 1, "regenerate": false }
```

Runs asynchronously in `BackgroundTasks`. Generates 5 embedding types: `summary`, `skills`, `experience`, `education`, `full`.

**Internal flow:** `MultiVectorEmbeddingService.generate_candidate_embeddings()` → fetches profile data via CRUD → `ResumeRepresentationService` generates text per type → `GeminiService (embedding).generate_embedding()` → stored in `candidate_embeddings` table.

---

### `GET /embeddings/candidates/{candidate_id}/status`
- **Auth:** Bearer JWT (CANDIDATE role)

**Response:**
```json
{
  "candidate_id": 1,
  "total_embeddings": 5,
  "embedding_types": ["summary", "skills", "experience", "education", "full"],
  "has_all_embeddings": true,
  "model_name": "text-embedding-004",
  "dimension": 768
}
```

---

### `POST /embeddings/jobs/{job_id}/generate`
- **Auth:** Bearer JWT (CANDIDATE role)
- **Body:** Job data dict
- **Query params:** `regenerate: bool = false`

Generates 3 embedding types: `jd_summary`, `required_skills`, `responsibilities`. Runs in background.

---

### `GET /embeddings/jobs/{job_id}/status`
- **Auth:** Bearer JWT (CANDIDATE role)

Same shape as candidate status. `has_all_embeddings` is true when count ≥ 3.

---

## 5. Core Modules & Logic

### 5.1 Parsing Pipeline (`file_service.py`)

**`UnstructuredExtractor`** – primary document extraction:
- Uses `unstructured` library with `hi_res` strategy for PDFs/images
- Categorizes elements (titles, headings, paragraphs, tables, lists, images)
- Outputs `{"raw_text": "...", "headings": [...], "tables": [...], ...}`
- Validates DOCX files by checking for `word/document.xml` in ZIP structure
- Falls back to `_legacy_extract_text()` on failure (PyPDF2 → OCR via pytesseract)

**`ResumeParserService`** – LLM orchestration:
```
Step 1: Text extraction
  UnstructuredExtractor (hi_res)
    └─ on failure → _legacy_extract_text (PyPDF2 / python-docx / pytesseract)

Step 2: LLM parsing
  GeminiService.parse_resume_text()     [confidence_score=0.95]
    └─ on failure → OpenAIService       [confidence_score varies]
         └─ on failure → HuggingFaceService
              └─ on failure → _enhanced_fallback_parse (regex) [confidence_score=0.6]
                   └─ on total failure → minimal ParsedResumeData [confidence_score=0.1]
```

**`GeminiService` (parsing)** – structured prompt engineering:
- Sends resume text (truncated to 50,000 chars) with a strict JSON schema prompt
- Temperature=0.1, max_output_tokens=65536
- Parses response JSON into full `ParsedResumeData` including certifications, projects, languages
- Normalizes `skills_used` in experience entries (handles both string and object forms)

### 5.2 Local Embedding Pipeline (`embedding_service_local.py`)

- **Model:** `intfloat/e5-base-v2` (768-dim), lazy-loaded singleton
- **Prefix:** `"query: "` prepended to all texts (e5 model requirement)
- **Truncation:** 2000 chars per text
- **Normalization:** `normalize_embeddings=True`
- **Batch:** `batch_size=32`, no progress bar

### 5.3 Multi-vector Embedding Pipeline (`multi_vector_embedding_service.py`)

- **Model:** `gemini-embedding-001` (768-dim), called via `gemini_service_embedding.py`
- **Text generation:** `ResumeRepresentationService` / `JobRepresentationService` in `representation_service.py`
- **Candidate types (5):** `summary`, `skills`, `experience`, `education`, `full`
- **Job types (3):** `jd_summary`, `required_skills`, `responsibilities`
- **Storage:** pgvector tables `candidate_embeddings` / `job_embeddings` (see `embedding_models.py`)
- **Task type:** `RETRIEVAL_DOCUMENT` for candidates, `RETRIEVAL_QUERY` for jobs
- **Normalization:** normalized when `dimensions < 3072`
- **Deduplication:** skips generation if embeddings already exist (unless `regenerate=true`)

---

## 6. Data Flow

### Resume Parsing
```
Client
  → POST /api/v1/resumes/parse-resume (multipart)
  → ResumeParserService.parse(file_bytes, filename)
      → UnstructuredExtractor → raw_text
      → GeminiService → JSON string → ParsedResumeData
  ← ParseResponse { success, resume_id, parsed_data, confidence_score }
```

### Local Embedding
```
Client
  → POST /api/v1/embeddings/generate-embedding { "text": "..." }
  → EmbeddingService.generate_embedding(text, prefix_type="query")
      → SentenceTransformer.encode(["query: " + text])
  ← EmbeddingResponse { embedding: float[], dimension: 768 }
```

### Multi-vector Candidate Embedding
```
Client (NestJS)
  → POST /api/v1/embeddings/candidates/{id}/generate  [Bearer JWT]
  → BackgroundTask: MultiVectorEmbeddingService.generate_candidate_embeddings(db, id)
      → ProfileCRUD + WorkExperienceCRUD + EducationCRUD + ProfileSkillCRUD → profile_data
      → for each type in [summary, skills, experience, education, full]:
          → ResumeRepresentationService.generate_{type}_text(profile_data)
          → GeminiService.generate_embedding(text, task_type="RETRIEVAL_DOCUMENT")
          → INSERT INTO candidate_embeddings (pgvector)
  ← { "message": "Embedding generation started" }  (immediate)
```

---

## 7. Configuration & Environment

All settings are in `app/config.py` via `pydantic-settings`. Source: `.env` file.

### Required
| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL URL (e.g., `postgresql://user:pass@host/db`) |
| `JWT_SECRET` | Must match NestJS access token secret (HS256) |
| `JWT_REFRESH_SECRET` | Must match NestJS refresh token secret |

### LLM Keys (at least one recommended)
| Variable | Description | Default |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini key (`GOOGLE_API_KEY` also accepted) | – |
| `OPENAI_API_KEY` | OpenAI fallback | – |
| `HUGGINGFACE_API_KEY` | HuggingFace fallback | – |

### LLM Tuning
| Variable | Default | Description |
|---|---|---|
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model for parsing |
| `GEMINI_TEMPERATURE` | `0.1` | Low for consistent structured output |
| `GEMINI_MAX_TOKENS` | `2500` | Output token budget |
| `AI_PROCESSING_TIMEOUT` | `120` | Seconds |
| `AI_MAX_RETRIES` | `3` | Max retry attempts |
| `AI_ENABLE_FALLBACK` | `true` | Enable LLM fallback chain |

### Storage (optional – defaults to local)
| Variable | Default | Description |
|---|---|---|
| `UPLOAD_DIR` | `uploads/resumes` | Local upload path |
| `AWS_ACCESS_KEY_ID` | – | S3 uploads |
| `AWS_SECRET_ACCESS_KEY` | – | S3 uploads |
| `AWS_S3_BUCKET` | `transpahire-resumes` | S3 bucket name |

### Other
| Variable | Default | Description |
|---|---|---|
| `DEBUG` | `false` | Enables Swagger UI at `/docs` and `/redoc`; enables SQL echo |
| `MAX_FILE_SIZE` | `10485760` (10MB) | Max upload size |
| `ALLOWED_FILE_TYPES` | PDF, DOCX, DOC, JPEG, PNG, TIFF, BMP | Allowed MIME types |

---

## 8. Error Handling & Logging

### Middleware
- **`ErrorHandlerMiddleware`** – catches all unhandled exceptions; returns `{"detail": "Internal server error"}` (HTTP 500). Logs full traceback via `logger.exception`.
- **`setup_exception_handlers`** – registered handlers for:
  - `HTTPException` → `{detail}` with original status code
  - `RequestValidationError` → `{"detail": "Validation error", "errors": [...]}` (HTTP 422)
  - `StarletteHTTPException` → `{detail}` with original status code

### Service-level Error Handling

| Situation | Behavior |
|---|---|
| Unstructured extraction fails | Falls back silently to legacy PyPDF2/OCR |
| Gemini parsing fails | Falls back to OpenAI → HuggingFace → regex |
| All LLM parsers fail | Returns `ParsedResumeData` with `confidence_score=0.1` |
| `parse_resume` endpoint exception | Returns `ParseResponse(success=False, error=...)` — **no HTTP 500** |
| Embedding endpoint exception | Raises HTTP 500 |
| Multi-vector per-type failure | Logs error, marks that type `False`, continues other types |

### Logging
- Format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- Level: `INFO` globally
- Each service uses `logging.getLogger(__name__)`

> **TODO:** `get_user_from_database()` in `dependencies.py` currently returns mock data — real DB lookup is not implemented.

---

## 9. Performance Considerations

| Concern | Approach |
|---|---|
| Blocking LLM calls | `asyncio.get_event_loop().run_in_executor(None, ...)` wraps all Gemini calls |
| Local model cold start | `EmbeddingService` uses lazy loading — model loads on first request |
| Token limits | Text truncated to 2000 chars for embeddings, 50000 chars for parsing prompt |
| Batch embeddings | `batch_size=32` in sentence-transformers; 1–100 texts per request |
| Multi-vector generation | Runs in `BackgroundTasks` — returns HTTP 200 immediately |
| DB connection pool | `pool_size=10`, `max_overflow=20`, `pool_recycle=3600`, `pool_pre_ping=True` |
| Response size | GZip middleware enabled (`minimum_size=1000`) |
| Rate limiting | `slowapi` on remote IP address |
| Temp files | `UnstructuredExtractor` writes to `temp_extraction/`, deletes after processing |

---

## 10. Extensibility Guidelines

### Add a new LLM parser provider
1. Create `app/services/your_provider_service.py` implementing `async def parse_resume_text(self, text: str) -> ParsedResumeData`
2. Import and instantiate in `ResumeParserService.__init__()` in `file_service.py`
3. Add a `try/except` fallback block in `ResumeParserService.parse()` after HuggingFace

### Add a new API endpoint
1. Add the handler to the relevant router file in `app/routers/`
2. If creating a new router, register it in `app/main.py` with `app.include_router(...)`
3. Add request/response schemas to `app/models/schemas.py` if needed
4. Document the endpoint in **Section 4** of this file

### Add a new candidate embedding type
1. Add the type string to `MultiVectorEmbeddingService.CANDIDATE_EMBEDDING_TYPES`
2. Add a corresponding branch in `_get_candidate_text_for_type()`
3. Implement `generate_{type}_text()` in `ResumeRepresentationService` in `representation_service.py`
4. Update the status endpoint check (`has_all_embeddings: len >= N`)

### Add a new job embedding type
Same pattern as candidate — use `JOB_EMBEDDING_TYPES`, `_get_job_text_for_type()`, and `JobRepresentationService`.

### Change the local embedding model
Update `model_name` in `EmbeddingService.__init__()` in `embedding_service_local.py`. Verify `embed_dim` matches the new model's output dimension.

---

## Appendix: Key Schema Relationships

```
ParsedResumeData
  ├── personal_info: Dict
  ├── summary: str
  ├── skills: ParsedSkill
  │     ├── technical_skills: List[ParsedTechnicalSkill]  ← extends EnrichedSkill
  │     └── soft_skills: List[ParsedSoftSkill]
  ├── experience: List[ParsedExperience]
  │     └── skills_used: List[EnrichedSkill]
  ├── education: List[ParsedEducation]
  ├── certifications: List[ParsedCertification]
  ├── projects: List[ParsedProject]
  ├── languages: List[ParsedLanguage]
  └── confidence_score: float (0.0–1.0)

CandidateEmbedding (DB table: candidate_embeddings)
  ├── candidate_id, type, job_id (nullable)
  ├── vector: pgvector(768)
  └── model_name, dimension, version

JobEmbedding (DB table: job_embeddings)
  ├── job_id, type
  ├── vector: pgvector(768)
  └── model_name, dimension, version
```
