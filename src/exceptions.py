"""Custom exception hierarchy for the semantic search engine.

Using specific exception types (instead of bare Exception/ValueError)
lets callers — including the Streamlit UI — react appropriately to
each failure mode instead of catching and swallowing everything.
"""


class SemanticSearchError(Exception):
    """Base class for all application-specific errors."""


class DocumentParsingError(SemanticSearchError):
    """Raised when an uploaded file cannot be parsed into Q&A documents."""


class EmptyDocumentSetError(SemanticSearchError):
    """Raised when a parsed file yields zero valid documents."""


class VectorStoreConnectionError(SemanticSearchError):
    """Raised when the Pinecone client cannot be initialized or reached."""


class VectorStoreUpsertError(SemanticSearchError):
    """Raised when writing documents to the vector store fails."""


class VectorStoreQueryError(SemanticSearchError):
    """Raised when a search query against the vector store fails."""

