"""
Parses uploaded Q&A text files into structured Document objects.

Kept independent of Streamlit and Pinecone so it can be unit-tested
in isolation and reused (e.g. in a CLI or batch ingestion job).
"""

from dataclasses import dataclass

from src.exceptions import DocumentParsingError, EmptyDocumentSetError
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Document:
    """A single Q&A record ready for embedding and storage."""

    id: str
    question: str
    answer: str

    @property
    def text(self) -> str:
        """Combined text representation used for embedding."""
        return f"Q: {self.question}\nA: {self.answer}"


def parse_qa_text(raw_text: str) -> list[Document]:
    """Parse raw text into a list of Q&A Documents.

    Expected format: alternating non-empty lines of question / answer.
    Blank lines are ignored. Raises DocumentParsingError on malformed
    input and EmptyDocumentSetError if no documents result.
    """
    if raw_text is None:
        raise DocumentParsingError("File content is empty (None).")

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    if not lines:
        raise EmptyDocumentSetError("Uploaded file contains no non-empty lines.")

    if len(lines) % 2 != 0:
        logger.warning(
            "Odd number of lines (%d); the final question has no matching answer "
            "and will be dropped.",
            len(lines),
        )
        lines = lines[:-1]

    if not lines:
        raise EmptyDocumentSetError(
            "No complete Q&A pairs could be formed from the uploaded file."
        )

    documents = [
        Document(id=str(i // 2), question=lines[i], answer=lines[i + 1])
        for i in range(0, len(lines), 2)
    ]

    logger.info("Parsed %d Q&A document(s) from uploaded file.", len(documents))
    return documents


def load_documents_from_path(filepath: str, encoding: str = "utf-8") -> list[Document]:
    """Read a file from disk and parse it into Documents.

    Raises DocumentParsingError if the file cannot be read (bad
    encoding, missing file, permissions, etc.).
    """
    try:
        with open(filepath, encoding=encoding) as f:
            raw_text = f.read()
    except FileNotFoundError as exc:
        raise DocumentParsingError(f"File not found: {filepath}") from exc
    except UnicodeDecodeError as exc:
        raise DocumentParsingError(
            f"File '{filepath}' is not valid {encoding} text."
        ) from exc
    except OSError as exc:
        raise DocumentParsingError(f"Could not read file '{filepath}': {exc}") from exc

    return parse_qa_text(raw_text)
