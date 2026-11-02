from fastapi import APIRouter, HTTPException
from app.config import GEMINI_API_KEY
from bson import ObjectId
from app.database import contracts_collection,analysis_collection
from app.services.gemini_analyse import analyse_contract

router = APIRouter(
    prefix="/analysis",
    tags=["analysis"]
)

@router.get("/contract/{contract_id}")
async def analyze_contract(contract_id:str):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500,detail="Gemini API key not configured")
    
    contract = contracts_collection.find_one({"_id":ObjectId(contract_id)})

    if not contract:
        raise HTTPException(status_code=404,detail="Contract not found")
    
    contracts_collection.update_one({"_id":ObjectId(contract_id)}, {"$set": {"analysis_status":"in_progress"}})

    result = await analyse_contract(contract_id,contract["text_content"])

    doc = result.model_dump()
    contracts_collection.update_one({"_id":ObjectId(contract_id)}, {"$set": {"analysis_status":"completed"}})
    insert_result = analysis_collection.insert_one(doc)
    result.id = str(insert_result.inserted_id)

    return{
        "message":"Analysis completed successfully",
        "result":result,
        "id":result.id
    }

    

    
    