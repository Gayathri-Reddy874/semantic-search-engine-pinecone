from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.pinecone_client import SearchHit
from src.search_service import SemanticSearchService


@pytest.fixture
def settings():
    return Settings(
        pinecone_api_key="test-key",
        default_top_k=3,
        max_top_k=10,
    )


@pytest.fixture
def service(settings):
    with patch("src.search_service.PineconeSearchClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        svc = SemanticSearchService(settings)
        yield svc, mock_client


def test_ingest_text_parses_and_uploads(service):
    svc, mock_client = service
    mock_client.upload_documents.return_value = 2

    count = svc.ingest_text("Q1\nA1\nQ2\nA2")

    assert count == 2
    uploaded_docs = mock_client.upload_documents.call_args[0][0]
    assert len(uploaded_docs) == 2


def test_search_uses_default_top_k_when_not_specified(service):
    svc, mock_client = service
    mock_client.search.return_value = [SearchHit(id="0", text="x", score=0.9)]

    svc.search("hello")

    mock_client.search.assert_called_once_with("hello", top_k=3)


def test_search_clamps_top_k_to_configured_max(service):
    svc, mock_client = service
    mock_client.search.return_value = []

    svc.search("hello", top_k=999)

    mock_client.search.assert_called_once_with("hello", top_k=10)


def test_index_stats_delegates_to_client(service):
    svc, mock_client = service
    mock_client.get_stats.return_value = {"total_vector_count": 42}

    stats = svc.index_stats()

    assert stats == {"total_vector_count": 42}
