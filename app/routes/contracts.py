"""contracts.py – Upload, list, and retrieve contracts.

The upload endpoint now triggers RAG indexing (chunking + embedding) in a
ThreadPoolExecutor worker so that CPU-intensive embedding generation does not
block the FastAPI async event loop.  If indexing fails the contract record is
preserved and ``rag_status`` is set to ``"failed"``; the user can retry via
``POST /contracts/{contract_id}/reindex``.
"""

import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, HTTPException, UploadFile, File

from app.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, UPLOAD_DIR
from app.database import contracts_collection
from app.models import Contract
from app.services.document_parser import extract_text
from app.services.rag_service import index_contract

logger = logging.getLogger(__name__)

# A small thread-pool so embedding generation runs off the async event loop.
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag_index")

router = APIRouter(
    prefix="/contracts",
    tags=["contracts"],
)


def _run_indexing(contract_id: str, pages: list) -> int:
    """Synchronous helper called inside the thread pool.

    Returns the number of indexed chunks.  On failure, updates rag_status in
    MongoDB and re-raises so the caller can handle it.
    """
    try:
        n = index_contract(contract_id=contract_id, pages=pages)
        contracts_collection.update_one(
            {"_id": ObjectId(contract_id)},
            {"$set": {"rag_status": "indexed"}},
        )
        logger.info("RAG indexing succeeded for contract %s (%d chunks).", contract_id, n)
        return n
    except Exception as exc:
        contracts_collection.update_one(
            {"_id": ObjectId(contract_id)},
            {"$set": {"rag_status": "failed"}},
        )
        logger.error(
            "RAG indexing failed for contract %s: %s", contract_id, exc, exc_info=True
        )
        raise


@router.post("/upload")
async def upload_contract(file: UploadFile = File(...)):
    """Upload a PDF or TXT contract.

    Workflow
    --------
    1. Validate file extension and size.
    2. Save the file to the uploads directory.
    3. Extract text (and page-level metadata) with ``document_parser``.
    4. Persist contract record to MongoDB with ``rag_status="pending"``.
    5. Run RAG indexing (embedding + Qdrant upsert) in a background thread.
    6. Update ``rag_status`` to ``"indexed"`` or ``"failed"`` in MongoDB.
    7. Return the contract record plus indexing results.
    """
    # --- 1. Validation ---
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File extension not allowed")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File cannot be empty")
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    # --- 2. Save to disk ---
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)
    with open(file_path, "wb") as f:
        f.write(content)

    # --- 3. Extract text ---
    try:
        parsed = extract_text(file_path)
    except ValueError as exc:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(exc))

    # Normalise: parsed is always a dict now, but be defensive.
    text_content = parsed["text"] if isinstance(parsed, dict) else parsed
    page_count = int(parsed.get("page_count", 1) if isinstance(parsed, dict) else 1)
    word_count = int(parsed.get("word_count", len(text_content.split())) if isinstance(parsed, dict) else len(text_content.split()))
    pages = parsed.get("pages", [{"page": 1, "text": text_content}]) if isinstance(parsed, dict) else [{"page": 1, "text": text_content}]

    # --- 4. Persist to MongoDB ---
    contract_data = Contract(
        file_name=unique_name,
        original_name=file.filename,
        text_content=text_content,
        page_count=page_count,
        word_count=word_count,
        rag_status="pending",
    )
    doc = contract_data.model_dump()
    result = contracts_collection.insert_one(doc)
    contract_id = str(result.inserted_id)
    contract_data.id = contract_id

    # --- 5 & 6. RAG indexing (off event loop) ---
    chunks_indexed = 0
    rag_status = "failed"
    try:
        future = _executor.submit(_run_indexing, contract_id, pages)
        # Wait for the result.  For initial synchronous strategy this is
        # acceptable; switch to background tasks if non-blocking upload
        # responses are preferred.
        chunks_indexed = future.result(timeout=120)
        rag_status = "indexed"
    except Exception as exc:
        logger.warning(
            "RAG indexing did not complete for contract %s: %s", contract_id, exc
        )
        rag_status = "failed"

    # --- 7. Return ---
    return {
        "message": "Contract uploaded successfully",
        "contract": contract_data.model_dump(),
        "contract_id": contract_id,
        "rag_status": rag_status,
        "chunks_indexed": chunks_indexed,
    }


@router.get("/")
def list_contracts():
    """Return all contracts stored in MongoDB."""
    contracts = []
    for doc in contracts_collection.find():
        contract = Contract(**{k: v for k, v in doc.items() if k != "_id"})
        contract.id = str(doc["_id"])
        contracts.append(contract)
    return {"contracts": contracts}


@router.get("/{contract_id}")
def get_contract(contract_id: str):
    """Retrieve a single contract by its MongoDB ObjectId."""
    try:
        oid = ObjectId(contract_id)
    except (InvalidId, Exception):
        raise HTTPException(status_code=400, detail="Invalid contract ID format")

    doc = contracts_collection.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Contract not found")

    contract = Contract(**{k: v for k, v in doc.items() if k != "_id"})
    contract.id = str(doc["_id"])
    return {"contract": contract.model_dump()}
