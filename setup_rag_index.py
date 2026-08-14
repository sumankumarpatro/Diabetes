import os
import pickle
import faiss
import numpy as np
from rank_bm25 import BM25Okapi  # Preserved for down-stream hybrid search mapping
from sentence_transformers import SentenceTransformer
from pathlib import Path
from config import config
from loguru import logger

def setup_rag_index(kb_dir: str, index_output_path: str) -> None:
    """
    Loads text files from the knowledge base, creates embeddings, and saves a FAISS index.
    Preserves original slicing, chunking, and array mappings exactly.
    """
    kb_path = Path(kb_dir)
    index_path = Path(index_output_path)

    if not kb_path.exists():
        logger.error(f"Error: Knowledge base directory {kb_dir} not found.")
        return

    # 1. Load all text files from the knowledge base recursively
    raw_documents = []
    document_metadata = []
    
    # Use rglob to find all .txt files in all subdirectories exactly as original
    for file_path in kb_path.rglob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                raw_documents.append(content)
                
            # Extract category from the parent directory name if possible
            category = file_path.parent.name
            document_metadata.append({
                "source": file_path.name,
                "category": category
            })
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")

    if not raw_documents:
        logger.warning("No documents found in the knowledge base.")
        return

    logger.info(f"Loaded {len(raw_documents)} source files from {kb_dir} (recursively)")

    # 2. Chunking strategy: Split documents into overlapping chunks exactly as original
    chunk_size = 500  # Characters per chunk
    chunk_overlap = 50  # Overlap between chunks
    documents = []
    metadata_list = []

    for i, doc in enumerate(raw_documents):
        start = 0
        while start < len(doc):
            end = start + chunk_size
            chunk = doc[start:end]
            documents.append(chunk)
            
            # Attach metadata to each chunk
            metadata_list.append(document_metadata[i].copy())
            start += (chunk_size - chunk_overlap)

    logger.info(f"Created {len(documents)} chunks from {len(raw_documents)} source files")

    # 3. Initialize Embedding Model
    logger.info("Loading embedding model (sentence-transformers)...")
    model = SentenceTransformer(config.RETRIEVER_MODEL_NAME)

    # 4. Generate Embeddings (Optimized with batching, preserving original array conversion logic)
    logger.info("Generating embeddings for documents...")
    embeddings = model.encode(documents, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    dimension = embeddings.shape[1]  # Restored exact original index sizing matrix logic

    # 5. Create and Populate FAISS Index
    logger.info(f"Creating FAISS index with dimension {dimension}...")
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    # 6. Save the index and the documents (to map index back to text)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    # We save the documents and metadata exactly as designed
    doc_mapping_path = index_path.with_name(index_path.stem + "_docs.npy")
    np.save(str(doc_mapping_path), np.array(documents, dtype=object))

    meta_mapping_path = index_path.with_name(index_path.stem + "_meta.npy")
    np.save(str(meta_mapping_path), np.array(metadata_list, dtype=object))

    # Save the BM25 corpus (flattened list of all chunks)
    corpus_path = index_path.with_name(index_path.stem + "_corpus.npy")
    np.save(str(corpus_path), np.array(documents, dtype=object))

    logger.success(f"FAISS index saved to: {index_path}")
    logger.info(f"Document mapping saved to: {doc_mapping_path}")
    logger.info(f"Metadata mapping saved to: {meta_mapping_path}")
    logger.info(f"BM25 corpus saved to: {corpus_path}")
    logger.info(f"Total vectors in index: {index.ntotal}")

if __name__ == "__main__":
    KB_DIR = str(config.PROJECT_ROOT / "knowledge_base")
    INDEX_OUTPUT = str(config.INDEX_PATH)
    setup_rag_index(KB_DIR, INDEX_OUTPUT)
