import argparse
import logging
from dotenv import load_dotenv
from app.services.embedding_service import regenerate_faiss_index_if_missing


load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RebuildIndexCLI")

def main():
    parser = argparse.ArgumentParser(description="Rebuild FAISS index and metadata from policy documents.")
    parser.add_argument(
        "--force", action="store_true", help="Force regeneration even if index already exists."
    )
    args = parser.parse_args()

    logger.info("Starting FAISS index regeneration script...")

    try:
        regenerate_faiss_index_if_missing(force=args.force)
        logger.info("FAISS index and metadata regeneration complete.")
    except Exception as e:
        logger.exception("An error occurred while regenerating the index.")

if __name__ == "__main__":
    main()
