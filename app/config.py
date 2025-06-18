# app/config.py

import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# ==== Directories ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "policy_docs")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")

# Ensure FAISS storage directory exists
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

EMBEDDING_DIM = 384  # Match this with your embedding model output dimension
FAISS_INDEX_FILE = os.path.join(VECTOR_STORE_DIR, "index.faiss")       # Updated to new index filenames
ID_MAPPING_FILE = os.path.join(VECTOR_STORE_DIR, "index.pkl")          # (LangChain uses index.faiss & index.pkl)

# ==== Model Configuration ====
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")             # Embedding model name (from .env)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION", "")
# ==== OpenAI or LLM Configuration ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ==== General Settings ====
CHUNK_SIZE = 300  # Number of words per chunk
