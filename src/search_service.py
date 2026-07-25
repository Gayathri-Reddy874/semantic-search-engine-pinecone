"""
Service layer that orchestrates document parsing and vector-store
operations. This is the only module the UI layer (app.py) talks to —
it never touches Pinecone or file parsing directly.
"""

from src.config import Settings
from src.document_loader import parse_qa_text
from src.logger import get_logger
from src.pinecone_client import PineconeSearchClient, SearchHit

logger = get_logger(__name__)


class SemanticSearchService:
    """High-level API: ingest raw text, then search it semantically."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = PineconeSearchClient(settings)

    def ingest_text(self, raw_text: str) -> int:
        """Parse raw Q&A text and upload it to the vector store.

        Returns the number of documents uploaded.
        """
        documents = parse_qa_text(raw_text)
        return self._client.upload_documents(documents)

    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Run a semantic search, clamping top_k to the configured max."""
        effective_top_k = min(
            top_k or self._settings.default_top_k, self._settings.max_top_k
        )
        return self._client.search(query, top_k=effective_top_k)

    def index_stats(self) -> dict:
        """Expose vector store statistics for display in the UI."""
        return self._client.get_stats()
