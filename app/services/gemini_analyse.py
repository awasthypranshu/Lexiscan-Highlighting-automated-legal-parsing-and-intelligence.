import httpx #python library to send http req. to other applications
import json
from app.config import GEMINI_API_KEY
from app.services.prompt import CONTRACT_ANALYSIS_PROMPT_TEMPLATE
from app.models import AnalysisResult,ClauseAnalysis,RiskFlag,RiskLevel
from google import genai

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-4-argon:generateContent"

async def analyse_contract(contract_id:str,text_content:str):
    prompt = CONTRACT_ANALYSIS_PROMPT_TEMPLATE.format(contract_text=text_content)
    client = genai.Client()

    interaction = client.chats.send_message(
        model = "gemini-4-argon",
        input=prompt,
    )
    
    raw_text = interaction.text

    raw_text = raw_text.strip()
    #Remove markdown code block markers
    if raw_text.startswith("```") and raw_text.endswith("```"):
        raw_text = raw_text[3:-3].strip()
    
    #handle empty response
    if not raw_text:
        raise ValueError("Gemini returned empty response")
    
    raw_text = raw_text.strip()
    analysys_data = json.loads(raw_text)

    key_clauses = [ClauseAnalysis(**clause) for clause in analysys_data.get("key_clauses",[])]
    risk_flags = [RiskFlag(**risk) for risk in analysys_data.get("risk_flags",[])]

    result = AnalysisResult(
        contract_id=contract_id,
        summary=analysys_data.get("summary"," "),
        contract_type=analysys_data.get("contract_type"," "),
        key_clauses=key_clauses,
        risk_flags=risk_flags,
        overall_risk_level=analysys_data.get("overall_risk_level",RiskLevel.LOW),
        recommendations=analysys_data.get("recommendations",[])
    )

    return result


        

        
        