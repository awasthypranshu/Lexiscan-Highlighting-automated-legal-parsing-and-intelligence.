from app.models import Contract
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
import os
import uuid
from app.config import ALLOWED_EXTENSIONS,MAX_FILE_SIZE_MB,UPLOAD_DIR
from app.services.document_parser import extract_text
from app.database import contracts_collection
from bson import ObjectId

router = APIRouter(
    prefix="/contracts",
    tags=["contracts"]
)

@router.post("/upload")
async def upload_contract(file: UploadFile=File(...)):
    #reading current file and reading content of that file and saving it on server
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400,detail="File extension not allowed")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File cannot be empty")
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)
    with open(file_path, "wb") as f:
        f.write(content)
    
    try:
        parsed = extract_text(file_path)
    except ValueError as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(e))

    contract_data = Contract(
        file_name=unique_name,
        original_name=file.filename,
        text_content=parsed["text"] if isinstance(parsed,dict) else parsed,
        page_count=int(parsed["page_count"] if isinstance(parsed,dict) else 1),
        word_count=int(parsed["word_count"] if isinstance(parsed,dict) else len(parsed.split()))
    )
    
    
    doc = contract_data.model_dump()
    result = contracts_collection.insert_one(doc)
    contract_data.id = str(result.inserted_id)

    return {
        "message":"Contract uploaded successfully",
        "contract":contract_data.model_dump(),
        "contract_id":str(result.inserted_id)
    }

@router.get("/")
def list_contracts():
    contracts = []
    for doc in contracts_collection.find():
        contract = Contract(**doc)
        contract.id = str(doc["_id"])
        contracts.append(contract)
    return {"contracts":contracts}

@router.get("{contract_id}")
def get_contract(contract_id:str):
    """Get contract by id"""
    contract = contracts_collection.find_one({"_id":ObjectId(contract_id)})
    if not contract:
        raise HTTPException(status_code=404,detail="Contract not found")
    contract["_id"] = str(contract["_id"])
    return {"contract":contract.model_dump()}
    
    
    