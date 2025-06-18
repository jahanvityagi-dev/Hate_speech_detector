# rebuild_faiss_index.py
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Config
POLICY_DIR = Path("data/policy_docs/")
VECTOR_STORE_DIR = "app/vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load policy docs
def load_policy_docs():
    docs = []
    for file in POLICY_DIR.glob("*.txt"):
        content = file.read_text(encoding="utf-8")
        docs.append(Document(page_content=content, metadata={"source": file.name}))
    return docs

# Main build flow
def main():
    print("🔹 Step 1: Loading policy docs...")
    raw_docs = load_policy_docs()

    print(f"✅ Loaded {len(raw_docs)} documents from {POLICY_DIR}")

    print("🔹 Step 2: Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    split_docs = splitter.split_documents(raw_docs)

    print(f"✅ Split into {len(split_docs)} chunks.")

    print("🔹 Step 3: Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("🔹 Step 4: Building FAISS index...")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    print(f"🔹 Step 5: Saving FAISS index to {VECTOR_STORE_DIR} ...")
    vectorstore.save_local(VECTOR_STORE_DIR)

    print("✅ All done! You can now run the app with the updated FAISS index.")

if __name__ == "__main__":
    main()
