from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class Contract(BaseModel):
    id: Optional[str] = None
    file_name: str
    original_name: str
    text_content: str
    page_count: int = 1
    word_count: int = 0
    status: str = "Uploaded"
    upload_date: datetime = Field(default_factory=datetime.now)
    # RAG indexing state: "pending" | "indexed" | "failed"
    rag_status: str = "pending"


class ClauseAnalysis(BaseModel):
    clause_title: str
    clause_text: str
    explanation: str
    is_standard: bool


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskFlag(BaseModel):
    risk_title: str
    description: str
    risk_level: RiskLevel
    recommendation: str
    clause_reference: str = ""


class AnalysisResult(BaseModel):
    id: Optional[str] = None
    contract_id: str
    analysis_date: str = ""
    summary: str = ""
    contract_type: str = ""
    key_clauses: list[ClauseAnalysis] = []
    risk_flags: list[RiskFlag] = []
    overall_risk_level: RiskLevel = RiskLevel.LOW
    recommendations: list[str] = []

    def model_post_init(self, __context):
        if not self.analysis_date:
            self.analysis_date = datetime.now().isoformat()


# ---------------------------------------------------------------------------
# RAG / Chat models
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    """Request body for POST /contracts/{contract_id}/ask."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The question to answer based on the uploaded contract.",
    )


class SourceReference(BaseModel):
    """A single retrieved contract passage used to ground an answer."""
    page: int
    text: str
    score: float


class AskResponse(BaseModel):
    """Response for POST /contracts/{contract_id}/ask."""
    contract_id: str
    question: str
    answer: str
    sources: List[SourceReference] = []


class ReindexResponse(BaseModel):
    """Response for POST /contracts/{contract_id}/reindex."""
    contract_id: str
    message: str
    chunks_indexed: int
    rag_status: str
