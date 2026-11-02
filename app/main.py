from fastapi import FastAPI 
from app.database import init_db
app = FastAPI()

@app.on_event("startup")
def startup():
    init_db()

from app.routes.contracts import router as contracts_router
app.include_router(contracts_router)

@app.get("/")
def root():
    return {
        "app": "Lexiscan - Automated Legal Parsing & Intelligence API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }

from app.routes.analysis import router as analysis_router
app.include_router(analysis_router)