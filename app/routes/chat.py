"""chat.py – Contract-specific RAG chat endpoints.

Endpoints
---------
POST /contracts/{contract_id}/ask
    Ask a natural-language question about a specific uploaded contract.
    The system retrieves relevant passages from Qdrant (filtered strictly by
    contract_id) and passes them as grounding context to Gemini.

POST /contracts/{contract_id}/reindex
    Trigger re-indexing of an existing contract's text content.
    Useful for contracts uploaded before RAG was enabled, or to recover from
    a previous indexing failure.

Security note
-------------
There is currently no per-user ownership check on contracts (the base
application does not implement authentication).  Before exposing these
endpoints publicly, add user authentication and verify that the requesting
user owns the requested contract_id.
"""

import logging

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, HTTPException

from app.config import GEMINI_API_KEY
from app.database import contracts_collection
from app.models import AskResponse, QuestionRequest, ReindexResponse, SourceReference
from app.services.rag_service import answer_contract_question, index_contract

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/contracts",
    tags=["contract-chat"],
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _validate_and_fetch(contract_id: str) -> dict:
    """Validate *contract_id* format and return the MongoDB document.

    Raises
    ------
    HTTPException 400
        If *contract_id* is not a valid MongoDB ObjectId.
    HTTPException 404
        If no contract with that id exists in MongoDB.
    """
    try:
        oid = ObjectId(contract_id)
    except (InvalidId, Exception):
        raise HTTPException(status_code=400, detail="Invalid contract ID format")

    doc = contracts_collection.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Contract not found")
    return doc


# ---------------------------------------------------------------------------
# POST /contracts/{contract_id}/ask
# ---------------------------------------------------------------------------

@router.post("/{contract_id}/ask", response_model=AskResponse)
async def ask_contract_question(contract_id: str, body: QuestionRequest):
    """Answer a question about a specific contract using RAG.

    The endpoint:
    1. Validates the contract ID and verifies the contract exists.
    2. Checks that RAG indexing has completed (``rag_status == "indexed"``).
    3. Retrieves the top-5 most relevant passages from Qdrant (filtered by
       contract_id — chunks from other contracts are never retrieved).
    4. Builds a grounded prompt and calls Google Gemini.
    5. Returns the answer with page-level source references.

    Example request
    ---------------
    ``POST /contracts/507f1f77bcf86cd799439011/ask``

    ```json
    { "question": "What is the termination notice period?" }
    ```
    """
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key is not configured on the server.",
        )

    doc = _validate_and_fetch(contract_id)
    rag_status = doc.get("rag_status", "pending")

    if rag_status == "pending":
        raise HTTPException(
            status_code=409,
            detail=(
                "This contract is still being indexed. "
                "Please wait a moment and try again."
            ),
        )
    if rag_status == "failed":
        raise HTTPException(
            status_code=409,
            detail=(
                "RAG indexing failed for this contract. "
                "Use POST /contracts/{contract_id}/reindex to retry."
            ),
        )
    if rag_status != "indexed":
        raise HTTPException(
            status_code=409,
            detail=f"Unexpected RAG status '{rag_status}'. Cannot answer questions.",
        )

    try:
        result = answer_contract_question(
            contract_id=contract_id, question=body.question
        )
    except RuntimeError as exc:
        # Gemini API errors, missing API key, empty response, etc.
        logger.error("answer_contract_question failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Unexpected error in ask endpoint for contract %s: %s",
            contract_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

    sources = [SourceReference(**s) for s in result.get("sources", [])]

    return AskResponse(
        contract_id=contract_id,
        question=body.question,
        answer=result["answer"],
        sources=sources,
    )


# ---------------------------------------------------------------------------
# POST /contracts/{contract_id}/reindex
# ---------------------------------------------------------------------------

@router.post("/{contract_id}/reindex", response_model=ReindexResponse)
async def reindex_contract(contract_id: str):
    """Re-index an existing contract's text content into Qdrant.

    This is useful for:
    - Contracts that were uploaded before RAG was enabled.
    - Contracts whose indexing previously failed (``rag_status == "failed"``).
    - Forcing a fresh index after a Qdrant data wipe.

    Page metadata
    -------------
    If the original page-level extraction data is not available (contracts
    uploaded before the parser was updated), the entire text is treated as
    a single page and page references will be marked as approximate (page 1).
    """
    doc = _validate_and_fetch(contract_id)

    text_content = doc.get("text_content", "").strip()
    if not text_content:
        raise HTTPException(
            status_code=422,
            detail="No text content found in this contract record. Cannot re-index.",
        )

    # Build pages list.  If the original page breakdown was saved (future
    # uploads) use it; otherwise fall back to treating everything as page 1.
    # Note: page-level data is not persisted in MongoDB by the current upload
    # handler (only the joined text is stored), so existing contracts will
    # always use the single-page fallback.
    pages = [{"page": 1, "text": text_content}]

    # Mark as pending before starting so concurrent calls do not query an
    # inconsistent state.
    contracts_collection.update_one(
        {"_id": ObjectId(contract_id)},
        {"$set": {"rag_status": "pending"}},
    )

    try:
        chunks_indexed = index_contract(contract_id=contract_id, pages=pages)
        contracts_collection.update_one(
            {"_id": ObjectId(contract_id)},
            {"$set": {"rag_status": "indexed"}},
        )
        rag_status = "indexed"
        msg = f"Re-indexing complete. {chunks_indexed} chunks indexed."
        logger.info("Reindex succeeded for contract %s (%d chunks).", contract_id, chunks_indexed)
    except Exception as exc:
        contracts_collection.update_one(
            {"_id": ObjectId(contract_id)},
            {"$set": {"rag_status": "failed"}},
        )
        rag_status = "failed"
        chunks_indexed = 0
        msg = f"Re-indexing failed: {exc}"
        logger.error(
            "Reindex failed for contract %s: %s", contract_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail="Re-indexing failed. Check server logs.")

    return ReindexResponse(
        contract_id=contract_id,
        message=msg,
        chunks_indexed=chunks_indexed,
        rag_status=rag_status,
    )

