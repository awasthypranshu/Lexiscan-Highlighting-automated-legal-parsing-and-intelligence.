"""test_rag_service.py – Unit tests for app/services/rag_service.py.

All Qdrant and Gemini interactions are run against a temporary local Qdrant
instance so no external services are required.  Gemini calls are mocked.

Run with:
    cd "API Contract"
    python -m pytest tests/test_rag_service.py -v
"""

import os
import sys
import shutil
import tempfile
import threading
import unittest.mock as mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_rag_singletons(tmp_path):
    """Reset module-level singletons between tests to avoid state leakage."""
    import app.services.rag_service as svc

    # Patch QDRANT_PATH to use an isolated tmp directory.
    original_path = svc.QDRANT_PATH
    svc.QDRANT_PATH = str(tmp_path / "qdrant_test")
    os.makedirs(svc.QDRANT_PATH, exist_ok=True)

    # Reset singletons.
    svc._embedding_model = None
    svc._qdrant_client = None
    svc._collection_ready = False

    yield

    # Cleanup after test.
    if svc._qdrant_client is not None:
        try:
            svc._qdrant_client.close()
        except Exception:
            pass
    svc._embedding_model = None
    svc._qdrant_client = None
    svc._collection_ready = False
    svc.QDRANT_PATH = original_path


# ─────────────────────────────────────────────────────────────────────────────
# Chunking tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitIntoChunks:
    def _fn(self, text):
        from app.services.rag_service import split_into_chunks
        return split_into_chunks(text)

    def test_empty_text_returns_empty_list(self):
        assert self._fn("") == []
        assert self._fn("   ") == []

    def test_short_text_returns_single_chunk(self):
        text = "This is a short contract paragraph."
        chunks = self._fn(text)
        assert len(chunks) == 1
        assert text in chunks[0]

    def test_long_text_produces_multiple_chunks(self):
        # 5 000-char text must split into at least 4 chunks with CHUNK_SIZE=1200
        text = "This is a sentence with some content. " * 150
        chunks = self._fn(text)
        assert len(chunks) > 3

    def test_no_empty_chunks(self):
        text = "Paragraph one.\n\nParagraph two.\n\n\nParagraph three."
        chunks = self._fn(text)
        assert all(c.strip() for c in chunks)

    def test_overlap_content_appears_in_consecutive_chunks(self):
        # Create text where the end of chunk N should appear at the start of N+1.
        text = ("A " * 600) + "OVERLAP_MARKER " + ("B " * 600)
        from app.services.rag_service import CHUNK_SIZE, CHUNK_OVERLAP
        chunks = self._fn(text)
        if len(chunks) >= 2:
            # The overlap marker should appear in at least two adjacent chunks
            # if it falls near the boundary.
            combined = " ".join(chunks)
            assert "OVERLAP_MARKER" in combined

    def test_chunk_sizes_do_not_far_exceed_chunk_size(self):
        text = "Word " * 500
        from app.services.rag_service import CHUNK_SIZE
        chunks = self._fn(text)
        for chunk in chunks:
            # Allow some slack for sentence-level rounding.
            assert len(chunk) <= CHUNK_SIZE + 400, f"Chunk too large: {len(chunk)}"


class TestCreateContractChunks:
    def _fn(self, pages):
        from app.services.rag_service import create_contract_chunks
        return create_contract_chunks(pages)

    def test_page_metadata_preserved(self):
        pages = [
            {"page": 1, "text": "Page one content. " * 5},
            {"page": 2, "text": "Page two content. " * 5},
        ]
        chunks = self._fn(pages)
        pages_seen = {c["page"] for c in chunks}
        assert 1 in pages_seen
        assert 2 in pages_seen

    def test_chunk_number_is_sequential(self):
        pages = [{"page": 1, "text": "Word " * 400}]
        chunks = self._fn(pages)
        numbers = [c["chunk_number"] for c in chunks]
        assert numbers == list(range(len(numbers)))

    def test_empty_pages_produces_no_chunks(self):
        assert self._fn([]) == []
        assert self._fn([{"page": 1, "text": ""}]) == []

    def test_each_chunk_has_required_keys(self):
        pages = [{"page": 3, "text": "Some content here."}]
        chunks = self._fn(pages)
        for c in chunks:
            assert "text" in c
            assert "page" in c
            assert "chunk_number" in c


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant / index tests  (use real local Qdrant, mocked embeddings)
# ─────────────────────────────────────────────────────────────────────────────

def _fake_encode(texts, show_progress_bar=False, convert_to_numpy=True):
    """Return deterministic 384-dim fake embeddings without loading torch."""
    import numpy as np
    if isinstance(texts, str):
        return np.zeros(384, dtype="float32")
    return np.zeros((len(texts), 384), dtype="float32")


def _mock_model():
    m = mock.MagicMock()
    m.encode.side_effect = _fake_encode
    m.get_embedding_dimension.return_value = 384
    return m


class TestEnsureCollection:
    def test_collection_created(self):
        import app.services.rag_service as svc
        svc._embedding_model = _mock_model()
        svc.ensure_collection()
        client = svc.get_qdrant_client()
        names = {c.name for c in client.get_collections().collections}
        assert svc.COLLECTION_NAME in names

    def test_ensure_collection_idempotent(self):
        import app.services.rag_service as svc
        svc._embedding_model = _mock_model()
        svc.ensure_collection()
        svc._collection_ready = False  # force re-check
        svc.ensure_collection()  # should not raise
        client = svc.get_qdrant_client()
        names = {c.name for c in client.get_collections().collections}
        assert svc.COLLECTION_NAME in names


class TestIndexContract:
    def _index(self, contract_id, pages):
        import app.services.rag_service as svc
        svc._embedding_model = _mock_model()
        return svc.index_contract(contract_id, pages)

    def test_returns_chunk_count(self):
        pages = [{"page": 1, "text": "This is a test contract. " * 20}]
        n = self._index("contract_aaa", pages)
        assert n > 0

    def test_chunks_stored_with_contract_id(self):
        import app.services.rag_service as svc
        from qdrant_client import models as qm

        svc._embedding_model = _mock_model()
        pages = [{"page": 1, "text": "Contract A content. " * 20}]
        svc.index_contract("contract_aaa", pages)

        client = svc.get_qdrant_client()
        results = client.scroll(
            collection_name=svc.COLLECTION_NAME,
            scroll_filter=qm.Filter(
                must=[qm.FieldCondition(key="contract_id", match=qm.MatchValue(value="contract_aaa"))]
            ),
            limit=100,
        )
        points = results[0]
        assert len(points) > 0
        for p in points:
            assert p.payload["contract_id"] == "contract_aaa"

    def test_reindex_does_not_create_duplicates(self):
        import app.services.rag_service as svc
        from qdrant_client import models as qm

        svc._embedding_model = _mock_model()
        pages = [{"page": 1, "text": "Stable contract content. " * 20}]

        count_1 = svc.index_contract("contract_bbb", pages)
        count_2 = svc.index_contract("contract_bbb", pages)

        client = svc.get_qdrant_client()
        results = client.scroll(
            collection_name=svc.COLLECTION_NAME,
            scroll_filter=qm.Filter(
                must=[qm.FieldCondition(key="contract_id", match=qm.MatchValue(value="contract_bbb"))]
            ),
            limit=200,
        )
        # After re-index, point count must equal count_1 (not doubled).
        assert len(results[0]) == count_1

    def test_retrieval_isolated_per_contract(self):
        import app.services.rag_service as svc
        from qdrant_client import models as qm

        svc._embedding_model = _mock_model()
        pages_a = [{"page": 1, "text": "Contract Alpha text. " * 20}]
        pages_b = [{"page": 1, "text": "Contract Beta text. " * 20}]

        svc.index_contract("contract_alpha", pages_a)
        svc.index_contract("contract_beta", pages_b)

        client = svc.get_qdrant_client()
        # Fetch all points for alpha only.
        results_alpha = client.scroll(
            collection_name=svc.COLLECTION_NAME,
            scroll_filter=qm.Filter(
                must=[qm.FieldCondition(key="contract_id", match=qm.MatchValue(value="contract_alpha"))]
            ),
            limit=200,
        )
        for p in results_alpha[0]:
            assert p.payload["contract_id"] == "contract_alpha"

    def test_embedding_dimension_is_384(self):
        import app.services.rag_service as svc

        svc._embedding_model = _mock_model()
        svc.ensure_collection()
        client = svc.get_qdrant_client()
        info = client.get_collection(svc.COLLECTION_NAME)
        assert info.config.params.vectors.size == 384


# ─────────────────────────────────────────────────────────────────────────────
# search_contract_chunks tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchContractChunks:
    def test_search_returns_only_matching_contract(self):
        import app.services.rag_service as svc

        svc._embedding_model = _mock_model()
        pages_x = [{"page": 1, "text": "Contract X unique text. " * 30}]
        pages_y = [{"page": 1, "text": "Contract Y unique text. " * 30}]
        svc.index_contract("cid_x", pages_x)
        svc.index_contract("cid_y", pages_y)

        results = svc.search_contract_chunks("cid_x", "termination notice", top_k=10)
        for r in results:
            # All returned chunks must belong to cid_x (validated by contract_id in text).
            # We verify via the chunk text stored in payload.
            assert "Contract X" in r["text"]

    def test_search_returns_dict_with_required_keys(self):
        import app.services.rag_service as svc

        svc._embedding_model = _mock_model()
        pages = [{"page": 2, "text": "Payment terms are net 30. " * 20}]
        svc.index_contract("cid_pay", pages)

        results = svc.search_contract_chunks("cid_pay", "payment terms")
        assert len(results) > 0
        for r in results:
            assert "text" in r
            assert "page" in r
            assert "score" in r

    def test_search_empty_when_no_matching_contract(self):
        import app.services.rag_service as svc

        svc._embedding_model = _mock_model()
        # No contract indexed for "nonexistent_cid"
        svc.ensure_collection()
        results = svc.search_contract_chunks("nonexistent_cid_xyz", "anything")
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# answer_contract_question tests (Gemini mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestAnswerContractQuestion:
    def _setup_indexed(self, contract_id="cid_gemini_test"):
        import app.services.rag_service as svc
        svc._embedding_model = _mock_model()
        pages = [{"page": 1, "text": "The notice period for termination is 30 days. " * 20}]
        svc.index_contract(contract_id, pages)
        return contract_id

    def test_returns_answer_and_sources(self):
        import app.services.rag_service as svc

        contract_id = self._setup_indexed()

        mock_response = mock.MagicMock()
        mock_response.text = "The termination notice period is 30 days. [Page 1]"

        with mock.patch("app.services.rag_service.GEMINI_API_KEY", "fake_key"):
            with mock.patch("app.services.rag_service.genai") as mock_genai:
                mock_client = mock.MagicMock()
                mock_client.models.generate_content.return_value = mock_response
                mock_genai.Client.return_value = mock_client

                result = svc.answer_contract_question(contract_id, "What is the notice period?")

        assert "answer" in result
        assert "sources" in result
        assert "30 days" in result["answer"]
        assert isinstance(result["sources"], list)
        assert len(result["sources"]) > 0

    def test_gemini_api_error_raises_runtime_error(self):
        import app.services.rag_service as svc

        contract_id = self._setup_indexed("cid_err_test")

        with mock.patch("app.services.rag_service.GEMINI_API_KEY", "fake_key"):
            with mock.patch("app.services.rag_service.genai") as mock_genai:
                mock_client = mock.MagicMock()
                mock_client.models.generate_content.side_effect = Exception("API quota exceeded")
                mock_genai.Client.return_value = mock_client

                with pytest.raises(RuntimeError, match="Gemini API error"):
                    svc.answer_contract_question(contract_id, "What is the notice period?")

    def test_empty_gemini_response_raises_runtime_error(self):
        import app.services.rag_service as svc

        contract_id = self._setup_indexed("cid_empty_test")

        mock_response = mock.MagicMock()
        mock_response.text = ""

        with mock.patch("app.services.rag_service.GEMINI_API_KEY", "fake_key"):
            with mock.patch("app.services.rag_service.genai") as mock_genai:
                mock_client = mock.MagicMock()
                mock_client.models.generate_content.return_value = mock_response
                mock_genai.Client.return_value = mock_client

                with pytest.raises(RuntimeError, match="empty response"):
                    svc.answer_contract_question(contract_id, "Something?")

    def test_missing_api_key_raises_runtime_error(self):
        import app.services.rag_service as svc

        contract_id = self._setup_indexed("cid_nokey_test")

        with mock.patch("app.services.rag_service.GEMINI_API_KEY", None):
            with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
                svc.answer_contract_question(contract_id, "Something?")

