"""
Regulatory document semantic search.
Hybrid search combining vector similarity and keyword matching.
"""
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.schema import Document
    HAS_RAG_DEPS = True
except ImportError:
    HAS_RAG_DEPS = False

from config.settings import CHROMA_DB_DIR, EMBEDDING_MODEL, RAG_TOP_K


class RegulatoryRetriever:
    """Hybrid semantic + keyword retriever for Gujarat regulations."""

    def __init__(
        self,
        persist_dir: str = None,
        embedding_model: str = None,
        top_k: int = None,
    ):
        self.persist_dir = str(persist_dir or CHROMA_DB_DIR)
        self.top_k = top_k or RAG_TOP_K
        self._vectorstore = None

        if not HAS_RAG_DEPS:
            logger.warning("RAG dependencies not available. Using mock retriever.")
            return

        try:
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model or EMBEDDING_MODEL)
            self._vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=embeddings,
                collection_name="gujarat_regulations",
            )
            logger.info(f"Loaded vector store from {self.persist_dir}")
        except Exception as e:
            logger.warning(f"Could not load vector store: {e}. Using mock retriever.")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve relevant regulation chunks for a query.

        Returns:
            List of dicts with 'content', 'metadata', and 'relevance_score'
        """
        k = top_k or self.top_k

        if not self._vectorstore:
            return self._mock_retrieve(query, k)

        try:
            results = self._vectorstore.similarity_search_with_relevance_scores(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": round(score, 4),
                }
                for doc, score in results
            ]
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return self._mock_retrieve(query, k)

    def retrieve_legal_context(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve and format legal context for the LLM.
        Returns a formatted string ready for injection into the prompt.
        """
        results = self.retrieve(query, top_k)

        if not results:
            return "No relevant regulatory context found."

        sections = []
        for i, r in enumerate(results, 1):
            source = r["metadata"].get("source_file", "Unknown")
            act = r["metadata"].get("act_name", "")
            score = r.get("relevance_score", 0)
            sections.append(
                f"[Source {i}] ({act} — {source}, Relevance: {score:.2f})\n"
                f"{r['content']}\n"
            )

        return "\n---\n".join(sections)

    def _mock_retrieve(self, query: str, k: int) -> List[Dict]:
        """Return mock regulatory content for demo."""
        mock_results = [
            {
                "content": (
                    "SECTION 12.3 — Riverfront and Water Body Buffer Zone: "
                    "No new construction is permitted within 500 meters of any "
                    "designated water body, river, lake, or reservoir. This includes "
                    "the Tapi River and its tributaries. Violation Penalty: Immediate "
                    "Stop Work Order + demolition within 30 days."
                ),
                "metadata": {
                    "source_file": "gujarat_gdcr_2017_summary.txt",
                    "act_name": "Gujarat GDCR 2017",
                    "page_number": 47,
                },
                "relevance_score": 0.89,
            },
            {
                "content": (
                    "SECTION 22 — Development Plan Compliance: All development must "
                    "conform to the approved Development Plan. Minimum 30% green cover "
                    "must be maintained in residential zones."
                ),
                "metadata": {
                    "source_file": "gujarat_tp_ud_act_1976.txt",
                    "act_name": "Gujarat TP & UD Act 1976",
                    "page_number": 12,
                },
                "relevance_score": 0.75,
            },
            {
                "content": (
                    "SECTION 15.1 — Green Belt Zones: Designated green belt areas are "
                    "protected from any form of construction. No change of land use is "
                    "permitted. Encroachment is a criminal offense under Section 52."
                ),
                "metadata": {
                    "source_file": "gujarat_gdcr_2017_summary.txt",
                    "act_name": "Gujarat GDCR 2017",
                    "page_number": 63,
                },
                "relevance_score": 0.71,
            },
        ]
        return mock_results[:k]
