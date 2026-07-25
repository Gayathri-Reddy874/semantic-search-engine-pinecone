import pytest

from src.document_loader import Document, parse_qa_text
from src.exceptions import DocumentParsingError, EmptyDocumentSetError


def test_parse_qa_text_happy_path():
    raw = "What is Python?\nA programming language.\nWhat is Pinecone?\nA vector database."
    docs = parse_qa_text(raw)

    assert len(docs) == 2
    assert docs[0] == Document(id="0", question="What is Python?", answer="A programming language.")
    assert docs[1].question == "What is Pinecone?"


def test_parse_qa_text_ignores_blank_lines():
    raw = "Q1\n\nA1\n\n\nQ2\nA2\n"
    docs = parse_qa_text(raw)
    assert len(docs) == 2


def test_parse_qa_text_drops_trailing_unmatched_question():
    raw = "Q1\nA1\nQ2 with no answer"
    docs = parse_qa_text(raw)
    assert len(docs) == 1
    assert docs[0].question == "Q1"


def test_parse_qa_text_empty_input_raises():
    with pytest.raises(EmptyDocumentSetError):
        parse_qa_text("")


def test_parse_qa_text_whitespace_only_raises():
    with pytest.raises(EmptyDocumentSetError):
        parse_qa_text("   \n\n   \n")


def test_parse_qa_text_none_raises():
    with pytest.raises(DocumentParsingError):
        parse_qa_text(None)


def test_document_text_property_formats_as_qa():
    doc = Document(id="0", question="Q?", answer="A.")
    assert doc.text == "Q: Q?\nA: A."


def test_parse_qa_text_single_unmatched_question_raises_empty_set():
    with pytest.raises(EmptyDocumentSetError):
        parse_qa_text("Only a question, no answer")
