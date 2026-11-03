"""main.py – LexiScan FastAPI application entry point."""

import logging

from fastapi import FastAPI

from app.database import init_db

# Configure root logger so all modules use the same format.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="LexiScan – Automated Legal Parsing & Intelligence API",
    description=(
        "Upload legal contracts (PDF / TXT), analyse key clauses and risk flags "
        "with Google Gemini, and ask natural-language questions about any uploaded "
        "contract using RAG-powered chat."
    ),
    version="2.0.0",
)


@app.on_event("startup")
def startup():
    init_db()


# --- Routers ---
from app.routes.contracts import router as contracts_router
from app.routes.analysis import router as analysis_router
from app.routes.chat import router as chat_router

app.include_router(contracts_router)
app.include_router(analysis_router)
app.include_router(chat_router)


@app.get("/")
def root():
    return {
        "app": "Lexiscan - Automated Legal Parsing & Intelligence API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
        "features": [
            "Contract upload (PDF / TXT)",
            "Gemini-powered clause analysis & risk flagging",
            "RAG-powered contract Q&A (POST /contracts/{id}/ask)",
            "Contract re-indexing (POST /contracts/{id}/reindex)",
        ],
    }
