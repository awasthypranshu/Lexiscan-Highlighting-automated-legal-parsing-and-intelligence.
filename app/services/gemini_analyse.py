"""gemini_analyse.py – Gemini-powered contract analysis.

Uses ``google-genai`` (google.genai) which is the current Google Gen AI SDK.
The correct API for non-streaming content generation is:

    client.models.generate_content(model=..., contents=...)

The previous code used ``client.chats.send_message(model=..., input=...)``
which does not exist in google-genai 2.x and would raise AttributeError at
runtime.  This has been corrected to use ``models.generate_content``.

The Gemini model identifier (``GEMINI_MODEL``) is read from ``app.config``
so it can be overridden via the ``GEMINI_MODEL`` environment variable without
touching the source code.
"""

import json
import logging

from google import genai

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.models import AnalysisResult, ClauseAnalysis, RiskFlag, RiskLevel
from app.services.prompt import CONTRACT_ANALYSIS_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


async def analyse_contract(contract_id: str, text_content: str) -> AnalysisResult:
    """Analyse a contract with Gemini and return a structured AnalysisResult.

    Parameters
    ----------
    contract_id:
        The MongoDB ObjectId string of the contract.
    text_content:
        The full extracted text of the contract.

    Returns
    -------
    AnalysisResult
        Pydantic model populated from Gemini's JSON response.

    Raises
    ------
    ValueError
        If Gemini returns an empty or unparseable response.
    RuntimeError
        If the Gemini API call itself fails.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not configured.")

    prompt = CONTRACT_ANALYSIS_PROMPT_TEMPLATE.format(contract_text=text_content)

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        raw_text = (response.text or "").strip()
    except Exception as exc:
        logger.error("Gemini API call failed during analysis: %s", exc, exc_info=True)
        raise RuntimeError(f"Gemini API error: {exc}") from exc

    if not raw_text:
        raise ValueError("Gemini returned an empty response for contract analysis.")

    # Strip markdown code-fence if present.
    if raw_text.startswith("```") and raw_text.endswith("```"):
        raw_text = raw_text[3:-3].strip()
    # Also handle ```json ... ``` fences.
    if raw_text.startswith("json"):
        raw_text = raw_text[4:].strip()

    try:
        analysis_data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.error(
            "Failed to parse Gemini JSON response: %s\nRaw: %s", exc, raw_text[:500]
        )
        raise ValueError(f"Gemini returned non-JSON content: {exc}") from exc

    key_clauses = [
        ClauseAnalysis(**clause) for clause in analysis_data.get("key_clauses", [])
    ]
    risk_flags = [
        RiskFlag(**risk) for risk in analysis_data.get("risk_flags", [])
    ]

    result = AnalysisResult(
        contract_id=contract_id,
        summary=analysis_data.get("summary", ""),
        contract_type=analysis_data.get("contract_type", ""),
        key_clauses=key_clauses,
        risk_flags=risk_flags,
        overall_risk_level=analysis_data.get("overall_risk_level", RiskLevel.LOW),
        recommendations=analysis_data.get("recommendations", []),
    )
    return result
