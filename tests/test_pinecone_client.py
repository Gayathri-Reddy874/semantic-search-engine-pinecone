from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.document_loader import Document
from src.exceptions import VectorStoreQueryError, VectorStoreUpsertError
from src.pinecone_client import PineconeSearchClient


@pytest.fixture
def settings():
    return Settings(pinecone_api_key="test-key", pinecone_index_name="test-index")


@pytest.fixture
def mock_pc_and_index():
    """Patch the Pinecone SDK class and return (mock_pc, mock_index)."""
    with patch("src.pinecone_client.Pinecone") as mock_pinecone_cls:
        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.list_indexes.return_value = [{"name": "test-index"}]
        mock_pc.Index.return_value = mock_index
        mock_pinecone_cls.return_value = mock_pc
        yield mock_pc, mock_index


def test_uses_existing_index_without_creating(settings, mock_pc_and_index):
    mock_pc, _ = mock_pc_and_index
    PineconeSearchClient(settings)
    mock_pc.create_index_for_model.assert_not_called()


def test_creates_index_when_missing(settings, mock_pc_and_index):
    mock_pc, _ = mock_pc_and_index
    mock_pc.list_indexes.return_value = []

    PineconeSearchClient(settings)

    mock_pc.create_index_for_model.assert_called_once()


def test_upload_documents_batches_and_returns_count(settings, mock_pc_and_index):
    _, mock_index = mock_pc_and_index
    client = PineconeSearchClient(settings)
    docs = [Document(id=str(i), question="Q", answer="A") for i in range(5)]

    count = client.upload_documents(docs)

    assert count == 5
    mock_index.upsert_records.assert_called_once()


def test_upload_documents_empty_list_is_noop(settings, mock_pc_and_index):
    _, mock_index = mock_pc_and_index
    client = PineconeSearchClient(settings)

    count = client.upload_documents([])

    assert count == 0
    mock_index.upsert_records.assert_not_called()


def test_upload_documents_wraps_sdk_errors(settings, mock_pc_and_index):
    _, mock_index = mock_pc_and_index
    mock_index.upsert_records.side_effect = RuntimeError("network down")
    client = PineconeSearchClient(settings)

    with pytest.raises(VectorStoreUpsertError):
        client.upload_documents([Document(id="0", question="Q", answer="A")])


def test_search_returns_parsed_hits(settings, mock_pc_and_index):
    _, mock_index = mock_pc_and_index
    mock_hit = {"_id": "0", "fields": {"text": "Q: x\nA: y"}, "_score": 0.87}
    mock_result = MagicMock()
    mock_result.result.hits = [mock_hit]
    mock_index.search.return_value = mock_result

    client = PineconeSearchClient(settings)
    hits = client.search("what is x?", top_k=3)

    assert len(hits) == 1
    assert hits[0].score == 0.87
    assert hits[0].text == "Q: x\nA: y"


def test_search_empty_query_returns_empty_list(settings, mock_pc_and_index):
    client = PineconeSearchClient(settings)
    assert client.search("   ") == []


def test_search_wraps_sdk_errors(settings, mock_pc_and_index):
    _, mock_index = mock_pc_and_index
    mock_index.search.side_effect = RuntimeError("timeout")

    client = PineconeSearchClient(settings)

    with pytest.raises(VectorStoreQueryError):
        client.search("a real query")

