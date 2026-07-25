"""
Thin, testable wrapper around the Pinecone SDK.

Encapsulates connection setup, index creation-if-missing, batched
upserts, and querying — with retries on transient failures and
translation of SDK exceptions into the app's own exception types so
callers never need to import `pinecone` directly.
"""

from dataclasses import dataclass

from pinecone import Pinecone
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import Settings
from src.document_loader import Document
from src.exceptions import (
    VectorStoreConnectionError,
    VectorStoreQueryError,
    VectorStoreUpsertError,
)
from src.logger import get_logger

logger = get_logger(__name__)

# Pinecone's integrated-inference upsert/search endpoints raise these
# transient error types under load; retry on them specifically rather
# than on every exception.
_TRANSIENT_ERRORS = (TimeoutError, ConnectionError)

_UPSERT_BATCH_SIZE = 96  # stays comfortably under Pinecone's per-request limits


@dataclass(frozen=True)
class SearchHit:
    """A single search result."""

    id: str
    text: str
    score: float


class PineconeSearchClient:
    """Manages a single Pinecone index configured for integrated embeddings."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._pc = self._connect()
        self._index = self._get_or_create_index()

    def _connect(self) -> Pinecone:
        try:
            return Pinecone(api_key=self._settings.pinecone_api_key)
        except Exception as exc:  # SDK raises plain Exception on bad auth
            raise VectorStoreConnectionError(
                f"Failed to initialize Pinecone client: {exc}"
            ) from exc

    def _get_or_create_index(self):
        index_name = self._settings.pinecone_index_name
        try:
            existing = {idx["name"] for idx in self._pc.list_indexes()}
            if index_name not in existing:
                logger.info("Index '%s' not found; creating it.", index_name)
                self._pc.create_index_for_model(
                    name=index_name,
                    cloud=self._settings.pinecone_cloud,
                    region=self._settings.pinecone_region,
                    embed={
                        "model": self._settings.embedding_model,
                        "field_map": {"text": "text"},
                    },
                )
            return self._pc.Index(index_name)
        except Exception as exc:
            raise VectorStoreConnectionError(
                f"Failed to open or create index '{index_name}': {exc}"
            ) from exc

    @retry(
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _upsert_batch(self, records: list[dict]) -> None:
        self._index.upsert_records(
            namespace=self._settings.pinecone_namespace,
            records=records,
        )

    def upload_documents(self, documents: list[Document]) -> int:
        """Upsert documents in batches. Returns the count uploaded."""
        if not documents:
            return 0

        records = [{"id": doc.id, "text": doc.text} for doc in documents]
        uploaded = 0

        try:
            for start in range(0, len(records), _UPSERT_BATCH_SIZE):
                batch = records[start : start + _UPSERT_BATCH_SIZE]
                self._upsert_batch(batch)
                uploaded += len(batch)
                logger.info(
                    "Upserted batch of %d (total %d/%d).",
                    len(batch),
                    uploaded,
                    len(records),
                )
        except Exception as exc:
            raise VectorStoreUpsertError(
                f"Failed to upload documents after {uploaded}/{len(records)} "
                f"succeeded: {exc}"
            ) from exc

        return uploaded

    @retry(
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        """Run a semantic search and return ranked hits."""
        if not query or not query.strip():
            return []

        try:
            results = self._index.search(
                namespace=self._settings.pinecone_namespace,
                query={"top_k": top_k, "inputs": {"text": query}},
            )
        except Exception as exc:
            raise VectorStoreQueryError(f"Search failed for query '{query}': {exc}") from exc

        hits = results.result.hits
        return [
            SearchHit(id=hit["id"], text=hit["fields"]["text"], score=hit["score"])
            for hit in hits
        ]

    def get_stats(self) -> dict:
        """Return index statistics (vector count, dimension, etc.) as a plain dict."""
        try:
            stats = self._index.describe_index_stats()
            return stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
        except Exception as exc:
            raise VectorStoreConnectionError(f"Failed to fetch index stats: {exc}") from exc