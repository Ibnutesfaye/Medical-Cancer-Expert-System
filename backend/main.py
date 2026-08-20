"""
main_v2.py — Production-ready FastAPI entry point with MySQL integration.

Run with:
    venv\\Scripts\\python.exe main_v2.py
"""

# ── Fix OpenBLAS memory allocation error on Windows ───────────────────────────
# Must be set BEFORE any numpy/faiss/sentence-transformers imports.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ── DB setup ──────────────────────────────────────────────────────────────────
from db.database import create_tables, engine
from services.user_service import create_user, get_user_by_username
from schemas.user import UserCreate
from sqlalchemy.orm import Session

# ── Existing AI services (unchanged) ─────────────────────────────────────────
from config import config
from vector_db_faiss import VectorDatabase
from ingestion import IngestionService
from llm_service import LLMService
from rag_pipeline import RAGPipeline
from image_analyzer import get_image_analyzer

# ── Routers ───────────────────────────────────────────────────────────────────
from routes.auth import router as auth_router
from routes.chat import router as chat_router
from routes.documents import router as documents_router
from routes.images import router as images_router
from routes.admin import router as admin_router

# ── Singleton AI services (used by route modules via import) ──────────────────
vector_db = VectorDatabase(persist_directory=config.vector_db_path)

ingestion_service = IngestionService(
    vector_db=vector_db,
    chunk_size=config.rag.chunk_size,
    overlap=config.rag.chunk_overlap,
    embedding_model_name=config.embedding_model,
)

llm_service = LLMService(
    groq_api_key=config.groq_api_key,
    openai_api_key=config.openai_api_key,
    model_name=config.rag.model_name,
    temperature=config.rag.temperature,
    max_tokens=config.rag.max_tokens,
)

rag_pipeline = RAGPipeline(
    vector_db=vector_db,
    llm_service=llm_service,
    embedding_model_name=config.embedding_model,
    top_k=config.rag.top_k,
    similarity_threshold=config.rag.similarity_threshold,
    fallback_enabled=config.fallback_enabled,
)

startup_time = time.time()


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Create MySQL tables
    create_tables()
    print("SUCCESS: MySQL tables created / verified")

    # 2. Seed default admin user if not exists
    with Session(engine) as db:
        if config.admin_password and not get_user_by_username(db, config.admin_username):
            create_user(db, UserCreate(
                username=config.admin_username,
                password=config.admin_password,
                is_admin=True,
                full_name="System Admin",
            ))
            print(f"SUCCESS: Admin user '{config.admin_username}' seeded")
        elif not config.admin_password:
            print("INFO: ADMIN_PASSWORD is unset; default admin seeding skipped")

    # 3. Build image dataset index in background
    def _build_index():
        try:
            get_image_analyzer()._build_index()
        except Exception as e:
            print(f"WARNING: Dataset index build error: {e}")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _build_index)

    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Medical Cancer RAG Chatbot API",
    description="Production-ready API with MySQL, FAISS, JWT, and ResNet18",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(images_router)
app.include_router(admin_router)


# ── Legacy compatibility aliases ──────────────────────────────────────────────
# Keep old endpoints working so the existing frontend doesn't break
from routes.auth import router as _auth
from fastapi import Depends, UploadFile, File
from schemas.user import LoginRequest, LoginResponse, UserRead
from db.database import get_db
from sqlalchemy.orm import Session as _Session
from services import user_service as _us


@app.post("/auth/login", response_model=LoginResponse, include_in_schema=False)
def legacy_login(request: LoginRequest, db: _Session = Depends(get_db)):
    result = _us.login_user(db, request.username, request.password)
    return LoginResponse(
        access_token=result["access_token"],
        token_type="bearer",
        expires_in=result["expires_in"],
        user=UserRead.model_validate(result["user"]),
    )


@app.post("/auth/logout", include_in_schema=False)
def legacy_logout():
    return {"message": "Logged out successfully"}


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
def health():
    components: dict = {}
    try:
        components["vector_db"] = {"status": "up", "chunks": vector_db.count()}
    except Exception as e:
        components["vector_db"] = {"status": "down", "error": str(e)}
    try:
        from db.database import engine as _engine
        with _engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        components["mysql"] = {"status": "up"}
    except Exception as e:
        components["mysql"] = {"status": "down", "error": str(e)}
    try:
        provider = "groq" if llm_service.use_groq else "openai"
        components["llm"] = {"status": "up", "provider": provider}
    except Exception as e:
        components["llm"] = {"status": "down", "error": str(e)}

    all_up = all(v.get("status") == "up" for v in components.values())
    return {
        "status": "healthy" if all_up else "degraded",
        "version": "2.0.0",
        "uptime_seconds": int(time.time() - startup_time),
        "components": components,
    }


@app.get("/metrics", tags=["system"])
def metrics():
    return rag_pipeline.external_search.get_metrics()


# ── Serve built frontend ──────────────────────────────────────────────────────
_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(_dist, "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        skip = ("auth/", "chat", "documents", "images", "admin", "health", "metrics", "assets/")
        if any(full_path.startswith(p) for p in skip):
            raise HTTPException(status_code=404)
            
        file_path = os.path.join(_dist, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
            
        return FileResponse(os.path.join(_dist, "index.html"))


if __name__ == "__main__":
    import uvicorn
    # Intentional for the container/server entry point; network access is
    # restricted by the deployment firewall/reverse proxy.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)  # nosec B104
