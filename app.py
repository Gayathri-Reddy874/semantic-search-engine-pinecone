"""
Semantic Search Engine - Streamlit front end.

This module is intentionally "thin": it handles rendering and user
interaction only. All parsing, embedding, and storage logic lives in
src/, so it can be tested and reused without a browser.
"""

import streamlit as st

from src.config import get_settings
from src.exceptions import (
    DocumentParsingError,
    EmptyDocumentSetError,
    SemanticSearchError,
)
from src.logger import get_logger
from src.search_service import SemanticSearchService

st.set_page_config(page_title="Semantic Search Engine", page_icon="🔍", layout="wide")

st.markdown(
    """
    <style>
    .title { text-align: center; font-size: 42px; font-weight: bold; color: #1d4ed8; }
    .subtitle { text-align: center; color: #475569; margin-bottom: 25px; }
    .result-card {
        background: white; padding: 18px; border-radius: 12px;
        margin-bottom: 12px; box-shadow: 0px 3px 8px rgba(0,0,0,0.08);
        color: #0f172a;
    }
    .result-card div { color: #0f172a; }
    .score { color: #2563eb; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_service() -> SemanticSearchService:
    """Build the service once per session and cache it across reruns."""
    settings = get_settings()
    logger = get_logger(__name__, settings.log_level)
    logger.info("Initializing semantic search service.")
    return SemanticSearchService(settings)


def render_upload_and_search(
    service: SemanticSearchService, max_upload_mb: int, top_k: int
) -> None:
    uploaded_file = st.file_uploader(
        "📄 Upload a .txt file (alternating question / answer lines)",
        type=["txt"],
    )

    if not uploaded_file:
        st.info("📌 Upload a .txt file to begin.")
        return

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > max_upload_mb:
        st.error(f"File is {size_mb:.1f} MB, which exceeds the {max_upload_mb} MB limit.")
        return

    raw_text = uploaded_file.getvalue().decode("utf-8", errors="replace")

    if st.button("📤 Upload to Pinecone", type="primary"):
        with st.spinner("Embedding and uploading documents..."):
            try:
                count = service.ingest_text(raw_text)
                st.success(f"✅ Uploaded {count} document(s).")
                st.json(service.index_stats())
            except EmptyDocumentSetError as exc:
                st.warning(f"Nothing to upload: {exc}")
            except DocumentParsingError as exc:
                st.error(f"Could not parse file: {exc}")
            except SemanticSearchError as exc:
                st.error(f"Upload failed: {exc}")

    query = st.text_input("💬 Ask something...")
    if not query:
        return

    with st.spinner("Searching..."):
        try:
            hits = service.search(query, top_k=top_k)
        except SemanticSearchError as exc:
            st.error(f"Search failed: {exc}")
            return

    st.markdown("### 🔎 Results")
    if not hits:
        st.warning("No results found.")
        return

    for hit in hits:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="score">Score: {hit.score:.4f}</div>
                <br><div>{hit.text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.markdown('<div class="title">🔍 Semantic Search Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Powered by Pinecone Integrated Embeddings</div>',
        unsafe_allow_html=True,
    )

    try:
        settings = get_settings()
    except Exception as exc:
        st.error(f"⚠️ Configuration error: {exc}")
        st.stop()

    st.sidebar.title("⚙️ Settings")
    top_k = st.sidebar.slider("Top Results", 1, settings.max_top_k, settings.default_top_k)
    st.sidebar.caption(f"Index: `{settings.pinecone_index_name}`")
    st.sidebar.caption(f"Namespace: `{settings.pinecone_namespace}`")

    try:
        service = get_service()
    except SemanticSearchError as exc:
        st.error(f"⚠️ Could not connect to the vector store: {exc}")
        st.stop()

    render_upload_and_search(service, settings.max_upload_size_mb, top_k)


if __name__ == "__main__":
    main()
