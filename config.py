import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

def _get_int(name: str, default: str) -> int:
    """
    Read an integer setting from the environment.

    Everything in a .env file is text, so "400" arrives as the string "400".
    This helper converts it to a real integer and falls back to the default if the variable is missing or is not a valid number.
    Args:
        name: Environment variable name, for example "CHILD_CHUNK_SIZE". 
        default: Value to use when the variable is missing or invalid.
    Returns:
        The setting as an integer.
    """

    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value)
    except Exception as e:
        print(f"[config] '{name}={raw_value} is not a number. Using the value -> {default}")
        return default

# folders
PROJECT_ROOT = Path(__file__).parent

DOCUMENTS_DIR = PROJECT_ROOT/ "data" / "documents" 
EVALUATION_DIR = PROJECT_ROOT/ "data" / "evaluation" 
VECTORSTORE_DIR = PROJECT_ROOT/ "vectorstore"
FAISS_DIR = VECTORSTORE_DIR / "faiss"

DOC_STORE_PATH = VECTORSTORE_DIR / "doc_store.json"

# OPENAI Models
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
LLM_TEMPERATURE = 0.0

# FAISS / Pinecone
VECTOR_DB = os.getenv("VECTOR_DB", "faiss").strip().lower()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "advanced-rag")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws").strip().lower()
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1").strip().lower()

# Chunking
PARENT_CHUNK_SIZE = _get_int("PARENT_CHUNK_SIZE", default=1600)
PARENT_CHUNK_OVERLAP = _get_int("PARENT_CHUNK_OVERLAP", 200)
CHILD_CHUNK_SIZE = _get_int("CHILD_CHUNK_SIZE", default=400)
CHILD_CHUNK_OVERLAP = _get_int("CHILD_CHUNK_OVERLAP", 60)

# Retrieval Pipeline
NUM_QUERY_VARIANTS = _get_int("NUM_QUERY_VARIANTS", 3)
VECTOR_TOP_K = _get_int("VECTOR_TOP_K", 6)
BM25_TOP_K = _get_int("BM25_TOP_K", 6) 
HYBRID_TOP_K = _get_int("HYBRID_TOP_K", 10)
RERANK_TOP_K = _get_int("RERANK_TOP_K", 4)
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-4.0"))
MAX_CONTEXT_CHARS = _get_int("MAX_CONTEXT_CHARS", 8000)
MEMORY_WINDOW = _get_int("MEMORY_WINDOW", 4)

RERANKER=  os.getenv("RERANKER", "cross-encoder").strip().lower()
CROSS_ENCODER_MODEL=  os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


def check_openai_key() -> None:
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI API KEY is not set"
        )

def check_pinecone_config() -> None:
    if not PINECONE_API_KEY:
        raise ValueError(
            "VECTOR_DB = pinecone but PINECONE API KEY is NOT SET or EMPTY"
        )
    if not PINECONE_INDEX_NAME:
        raise ValueError(
            "VECTOR_DB = pinecone but PINECONE_INDEX_NAME is EMPTY"
        )

def describe_settings() -> dict:
    return {
        "Chat Model" : OPENAI_CHAT_MODEL,
        "Embedding Model" : OPENAI_EMBEDDING_MODEL,
        "VectorDB" : VECTOR_DB,
        "Parent Chunk Size" : PARENT_CHUNK_SIZE,
        "Child Chunk Size" : CHILD_CHUNK_SIZE
    }