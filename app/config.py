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

# ==== FAISS Configuration ====
EMBEDDING_DIM = 384  # Match this with your embedding model (e.g. BGE, sentence-transformers)
FAISS_INDEX_FILE = os.path.join(VECTOR_STORE_DIR, "faiss_index.bin")
ID_MAPPING_FILE = os.path.join(VECTOR_STORE_DIR, "id_mapping.json")

# ==== OpenAI or LLM Configuration ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ==== General Settings ====
CHUNK_SIZE = 300  # Number of words per chunk
