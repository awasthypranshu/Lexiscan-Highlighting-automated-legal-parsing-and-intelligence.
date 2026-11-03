"""test_api_endpoints.py – Integration tests for the FastAPI HTTP endpoints.

Tests use FastAPI's TestClient with a fully mocked RAG service and a
mocked MongoDB collection so no external services (MongoDB, Qdrant, Gemini)
are required.

Run with:
    cd "API Contract"
    python -m pytest tests/test_api_endpoints.py -v
"""

import os
import sys
import tempfile
import unittest.mock as mock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
# App + client fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    """Create a TestClient with mocked DB and RAG singletons."""
    import app.database as db_module
    from app.main import app

    # Patch init_db so tests don't need a real MongoDB.
    with mock.patch.object(db_module, "init_db", return_value=None):
        with TestClient(app) as c:
            yield c


# ─────────────────────────────────────────────────────────────────────────────
# /contracts/upload
# ─────────────────────────────────────────────────────────────────────────────

class TestUploadEndpoint:
    def _txt_file(self, content="Sample contract text. " * 50):
        return ("test_contract.txt", content.encode(), "text/plain")

    def test_upload_txt_success(self, client):
        """Upload a TXT file – should return 200 with contract_id."""
        import app.routes.contracts as cr
        import app.database as db_module
        from bson import ObjectId

        fake_id = ObjectId()

        with mock.patch.object(db_module, "contracts_collection") as mock_col:
            mock_col.insert_one.return_value = mock.MagicMock(inserted_id=fake_id)
            with mock.patch.object(cr, "index_contract", return_value=5) as mock_idx:
                with mock.patch.object(cr._executor, "submit") as mock_submit:
                    future = mock.MagicMock()
                    future.result.return_value = 5
                    mock_submit.return_value = future

                    name, content, ct = self._txt_file()
                    response = client.post(
                        "/contracts/upload",
                        files={"file": (name, content, ct)},
                    )

        assert response.status_code == 200
        data = response.json()
        assert "contract_id" in data
        assert data["rag_status"] in ("indexed", "failed", "pending")

    def test_upload_unsupported_extension_returns_400(self, client):
        response = client.post(
            "/contracts/upload",
            files={"file": ("contract.docx", b"dummy content", "application/octet-stream")},
        )
        assert response.status_code == 400

    def test_upload_empty_file_returns_400(self, client):
        response = client.post(
            "/contracts/upload",
            files={"file": ("contract.txt", b"", "text/plain")},
        )
        assert response.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# /contracts/{contract_id}/ask
# ─────────────────────────────────────────────────────────────────────────────

class TestAskEndpoint:
    _valid_id = "507f1f77bcf86cd799439011"

    def _mock_doc(self, rag_status="indexed"):
        return {
            "_id": __import__("bson").ObjectId(self._valid_id),
            "file_name": "test.txt",
            "original_name": "test.txt",
            "text_content": "Some contract text.",
            "page_count": 1,
            "word_count": 3,
            "status": "Uploaded",
            "rag_status": rag_status,
            "upload_date": __import__("datetime").datetime.now(),
        }

    def test_invalid_contract_id_returns_400(self, client):
        import app.config as cfg
        with mock.patch.object(cfg, "GEMINI_API_KEY", "fake_key"):
            response = client.post(
                "/contracts/not_an_objectid/ask",
                json={"question": "What is the notice period?"},
            )
        assert response.status_code == 400

    def test_missing_contract_returns_404(self, client):
        import app.config as cfg
        # Patch the contracts_collection as used inside app.routes.chat
        with mock.patch.object(cfg, "GEMINI_API_KEY", "fake_key"):
            with mock.patch("app.routes.chat.contracts_collection") as mock_col:
                mock_col.find_one.return_value = None
                response = client.post(
                    f"/contracts/{self._valid_id}/ask",
                    json={"question": "What is the notice period?"},
                )
        assert response.status_code == 404

    def test_unindexed_contract_returns_409(self, client):
        import app.config as cfg
        with mock.patch.object(cfg, "GEMINI_API_KEY", "fake_key"):
            with mock.patch("app.routes.chat.contracts_collection") as mock_col:
                mock_col.find_one.return_value = self._mock_doc(rag_status="pending")
                response = client.post(
                    f"/contracts/{self._valid_id}/ask",
                    json={"question": "What is the notice period?"},
                )
        assert response.status_code == 409

    def test_failed_index_returns_409(self, client):
        import app.config as cfg
        with mock.patch.object(cfg, "GEMINI_API_KEY", "fake_key"):
            with mock.patch("app.routes.chat.contracts_collection") as mock_col:
                mock_col.find_one.return_value = self._mock_doc(rag_status="failed")
                response = client.post(
                    f"/contracts/{self._valid_id}/ask",
                    json={"question": "What is the notice period?"},
                )
        assert response.status_code == 409

    def test_ask_returns_answer_and_sources(self, client):
        import app.config as cfg
        import app.routes.chat as chat_module

        with mock.patch.object(cfg, "GEMINI_API_KEY", "fake_key"):
            with mock.patch("app.routes.chat.contracts_collection") as mock_col:
                mock_col.find_one.return_value = self._mock_doc(rag_status="indexed")
                with mock.patch.object(
                    chat_module,
                    "answer_contract_question",
                    return_value={
                        "answer": "The notice period is 30 days. [Page 1]",
                        "sources": [{"page": 1, "text": "Either party may terminate...", "score": 0.82}],
                    },
                ):
                    response = client.post(
                        f"/contracts/{self._valid_id}/ask",
                        json={"question": "What is the notice period?"},
                    )

        assert response.status_code == 200
        data = response.json()
        assert data["contract_id"] == self._valid_id
        assert "answer" in data
        assert "sources" in data
        assert len(data["sources"]) > 0

    def test_missing_api_key_returns_500(self, client):
        # GEMINI_API_KEY is imported directly into app.routes.chat, so we must
        # patch it at its bound location there.
        with mock.patch("app.routes.chat.GEMINI_API_KEY", None):
            response = client.post(
                f"/contracts/{self._valid_id}/ask",
                json={"question": "What is the notice period?"},
            )
        assert response.status_code == 500

    def test_short_question_returns_422(self, client):
        import app.config as cfg
        with mock.patch.object(cfg, "GEMINI_API_KEY", "fake_key"):
            response = client.post(
                f"/contracts/{self._valid_id}/ask",
                json={"question": "Hi"},
            )
        # "Hi" is 2 chars, below min_length=3
        assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# /contracts/{contract_id}/reindex
# ─────────────────────────────────────────────────────────────────────────────

class TestReindexEndpoint:
    _valid_id = "507f1f77bcf86cd799439011"

    def _mock_doc(self):
        return {
            "_id": __import__("bson").ObjectId(self._valid_id),
            "file_name": "test.txt",
            "original_name": "test.txt",
            "text_content": "Reindex contract text. " * 50,
            "page_count": 1,
            "word_count": 100,
            "status": "Uploaded",
            "rag_status": "failed",
            "upload_date": __import__("datetime").datetime.now(),
        }

    def test_invalid_id_returns_400(self, client):
        response = client.post("/contracts/bad_id/reindex")
        assert response.status_code == 400

    def test_missing_contract_returns_404(self, client):
        with mock.patch("app.routes.chat.contracts_collection") as mock_col:
            mock_col.find_one.return_value = None
            response = client.post(f"/contracts/{self._valid_id}/reindex")
        assert response.status_code == 404

    def test_reindex_success_returns_chunk_count(self, client):
        import app.routes.chat as chat_module

        with mock.patch("app.routes.chat.contracts_collection") as mock_col:
            mock_col.find_one.return_value = self._mock_doc()
            mock_col.update_one.return_value = mock.MagicMock()
            with mock.patch.object(chat_module, "index_contract", return_value=12):
                response = client.post(f"/contracts/{self._valid_id}/reindex")

        assert response.status_code == 200
        data = response.json()
        assert data["chunks_indexed"] == 12
        assert data["rag_status"] == "indexed"


# ─────────────────────────────────────────────────────────────────────────────
# /analysis/contract/{contract_id}  (Backward-compatibility check)
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalysisEndpoint:
    _valid_id = "507f1f77bcf86cd799439011"

    def test_existing_analysis_route_intact(self, client):
        """Verify that existing contract analysis functionality remains intact."""
        from app.models import AnalysisResult, ClauseAnalysis, RiskFlag, RiskLevel
        from bson import ObjectId

        fake_analysis = AnalysisResult(
            contract_id=self._valid_id,
            summary="Standard NDA agreement between parties.",
            contract_type="Non-Disclosure Agreement",
            key_clauses=[
                ClauseAnalysis(
                    clause_title="Confidentiality",
                    clause_text="All disclosures are confidential.",
                    explanation="Protects company secrets.",
                    is_standard=True,
                )
            ],
            risk_flags=[
                RiskFlag(
                    risk_title="Indefinite duration",
                    description="Obligation never expires.",
                    risk_level=RiskLevel.MEDIUM,
                    recommendation="Limit to 3 years.",
                    clause_reference="Clause 5",
                )
            ],
            overall_risk_level=RiskLevel.MEDIUM,
            recommendations=["Add expiry date."],
        )

        with mock.patch("app.routes.analysis.GEMINI_API_KEY", "fake_key"):
            with mock.patch("app.routes.analysis.contracts_collection") as mock_cc:
                with mock.patch("app.routes.analysis.analysis_collection") as mock_ac:
                    mock_cc.find_one.return_value = {
                        "_id": ObjectId(self._valid_id),
                        "text_content": "Agreement text...",
                    }
                    mock_cc.update_one.return_value = mock.MagicMock()
                    mock_ac.insert_one.return_value = mock.MagicMock(inserted_id=ObjectId())

                    with mock.patch("app.routes.analysis.analyse_contract", return_value=fake_analysis):
                        response = client.get(f"/analysis/contract/{self._valid_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Analysis completed successfully"
        assert "result" in data
        assert data["result"]["contract_type"] == "Non-Disclosure Agreement"

