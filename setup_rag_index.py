import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from config import config
from loguru import logger

def setup_rag_index(kb_dir: str, index_output_path: str) -> None:
    """
    Loads text files from the knowledge base, creates embeddings, 
    and saves a FAISS index.
    """
    kb_path = Path(kb_dir)
    index_path = Path(index_output_path)

    if not kb_path.exists():
        logger.error(f"Error: Knowledge base directory {kb_dir} not found.")
        return
    documents = []
    for filename in os.listdir(kb_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(kb_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())
    
    if not documents:
        logger.warning("No documents found in the knowledge base.")
        return

    logger.info(f"Loaded {len(documents)} documents from {kb_dir}")
    # Using a lightweight model suitable for clinical/text tasks
    logger.info("Loading embedding model (sentence-transformers)...")
    model = SentenceTransformer(config.RETRIEVER_MODEL_NAME)
    logger.info("Generating embeddings for documents...")
    embeddings = model.encode(documents)
    dimension = embeddings.shape[1]
    logger.info(f"Creating FAISS index with dimension {dimension}...")
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    
    # We also need to save the documents to retrieve them later
    doc_mapping_path = index_path.with_name(index_path.stem + "_docs.npy")
    np.save(str(doc_mapping_path), np.array(documents, dtype=object))

    logger.success(f"FAISS index saved to: {index_path}")
    logger.info(f"Document mapping saved to: {doc_mapping_path}")
    logger.info(f"Total vectors in index: {index.ntotal}")

if __name__ == "__main__":
    # Use paths relative to project root for portability
    KB_DIR = str(config.PROJECT_ROOT / "knowledge_base")
    INDEX_OUTPUT = str(config.INDEX_PATH)

    setup_rag_index(KB_DIR, INDEX_OUTPUT)
