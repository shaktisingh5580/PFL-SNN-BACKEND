"""Ingest Gujarat regulatory PDFs into the ChromaDB vector store."""
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chatbot.rag.ingest import ingest_regulations


def main():
    logger.info("Starting regulation ingestion...")
    vectorstore = ingest_regulations()
    if vectorstore:
        logger.info("Ingestion complete! Vector store ready for RAG queries.")
    else:
        logger.error("Ingestion failed. Check logs for details.")


if __name__ == "__main__":
    main()
