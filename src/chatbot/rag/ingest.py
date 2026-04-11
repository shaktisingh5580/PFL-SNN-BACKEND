"""
PDF → ChromaDB vector store ingestion pipeline.
Loads Gujarat regulatory documents, chunks them, and stores embeddings.
"""
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HAS_RAG_DEPS = True
except ImportError:
    HAS_RAG_DEPS = False
    logger.warning("RAG dependencies not installed.")

from config.settings import REGULATIONS_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def ingest_regulations(
    docs_dir: str = None,
    persist_dir: str = None,
    embedding_model: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> Optional[object]:
    """
    Ingest regulatory PDFs into ChromaDB vector store.

    Args:
        docs_dir: Directory containing regulation PDFs
        persist_dir: ChromaDB persistent storage directory
        embedding_model: HuggingFace embedding model name
        chunk_size: Text chunk size for splitting
        chunk_overlap: Overlap between chunks

    Returns:
        ChromaDB vector store instance
    """
    docs_dir = Path(docs_dir or REGULATIONS_DIR)
    persist_dir = str(persist_dir or CHROMA_DB_DIR)
    embedding_model = embedding_model or EMBEDDING_MODEL
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP

    if not HAS_RAG_DEPS:
        logger.error("Cannot ingest: langchain, chromadb, or sentence-transformers not installed")
        return None

    # Load documents
    documents = []

    # PDF files
    for pdf_path in docs_dir.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            for page in pages:
                page.metadata["source_file"] = pdf_path.name
                page.metadata["act_name"] = _extract_act_name(pdf_path.name)
            documents.extend(pages)
            logger.info(f"Loaded {len(pages)} pages from {pdf_path.name}")
        except Exception as e:
            logger.warning(f"Failed to load {pdf_path.name}: {e}")

    # Text files as fallback
    for txt_path in docs_dir.glob("*.txt"):
        try:
            loader = TextLoader(str(txt_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = txt_path.name
                doc.metadata["act_name"] = _extract_act_name(txt_path.name)
            documents.extend(docs)
            logger.info(f"Loaded {txt_path.name}")
        except Exception as e:
            logger.warning(f"Failed to load {txt_path.name}: {e}")

    if not documents:
        logger.warning(f"No documents found in {docs_dir}. Creating sample regulations...")
        documents = _create_sample_regulations(docs_dir)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="gujarat_regulations",
    )

    logger.info(f"ChromaDB vector store created at {persist_dir} with {len(chunks)} chunks")
    return vectorstore


def _extract_act_name(filename: str) -> str:
    """Extract act name from filename."""
    name = Path(filename).stem.replace("_", " ").replace("-", " ")
    return name.title()


def _create_sample_regulations(docs_dir: Path) -> list:
    """Create sample regulatory text for demo purposes."""
    from langchain.schema import Document

    docs_dir.mkdir(parents=True, exist_ok=True)

    sample_text = """
GUJARAT GENERAL DEVELOPMENT CONTROL REGULATIONS (GDCR) 2017
Summary for Satellite Compliance Engine

SECTION 12: SETBACKS AND BUFFER ZONES

12.1 General Setbacks
All new construction must maintain minimum setbacks from property boundaries:
- Front: 4.5 meters (residential), 6.0 meters (commercial)
- Side: 3.0 meters each side
- Rear: 3.0 meters

12.3 Riverfront and Water Body Buffer Zone
No new construction is permitted within 500 meters of any designated water body,
river, lake, or reservoir. This includes the Tapi River and its tributaries.
Violation Penalty: Immediate Stop Work Order + demolition within 30 days.

12.5 Coastal Regulation Zone
Properties within CRZ limits (where applicable) must comply with CRZ 2019.

SECTION 15: PROTECTED ZONES

15.1 Green Belt Zones
Designated green belt areas are protected from any form of construction.
No change of land use is permitted. Encroachment is a criminal offense under
Section 52 of the Gujarat TP & UD Act, 1976.

15.2 Heritage Conservation Zones
Buildings within 100m of a designated heritage structure must obtain NOC from
Heritage Conservation Committee.

SECTION 18: FLOOR SPACE INDEX (FSI)

18.1 Base FSI by Zone Type
- Residential (R1): 1.8
- Residential (R2): 2.0
- Commercial (C1): 2.5
- Industrial (I1): 1.0
- Mixed Use (MU): 2.2

---

GUJARAT TOWN PLANNING AND URBAN DEVELOPMENT ACT, 1976

SECTION 22: Development Plan Compliance
All development must conform to the approved Development Plan.
Minimum 30% green cover must be maintained in residential zones.

SECTION 52: Penalties for Unauthorized Development
Any person who constructs, reconstructs, or makes additions to any building
without the written permission of the appropriate authority shall be punishable
with imprisonment up to one year, or with fine up to Rs. 5,00,000, or both.

---

SURAT URBAN DEVELOPMENT AUTHORITY (SUDA) - DEVELOPMENT PLAN 2035

Zone Classification Schedule:
- Zone R1: Low-density Residential
- Zone R2: High-density Residential
- Zone C1: Central Commercial
- Zone I1: Industrial
- Zone GB: Green Belt (No construction)
- Zone TRB: Tapi Riverfront Buffer (500m)

Special Provisions for Surat:
1. Tapi Riverfront: 500m no-construction buffer on both banks
2. Diamond Industry Zones: Special FSI provisions in Varachha
3. GIDC Areas: Industrial use only, no residential conversion
4. Smart City Zone: Additional FSI premium of 0.5 available
"""

    sample_path = docs_dir / "gujarat_gdcr_2017_summary.txt"
    sample_path.write_text(sample_text)
    logger.info(f"Created sample regulation file: {sample_path}")

    doc = Document(
        page_content=sample_text,
        metadata={
            "source_file": "gujarat_gdcr_2017_summary.txt",
            "act_name": "Gujarat GDCR 2017 Summary",
        }
    )
    return [doc]
