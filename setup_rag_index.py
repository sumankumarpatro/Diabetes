import os
from pathlib import Path
import faiss
import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from config import config

def setup_rag_index(kb_dir: Path, index_output_path: Path) -> None:
    kb_path = Path(kb_dir)
    index_path = Path(index_output_path)

    if not kb_path.exists():
        logger.error(f"Knowledge base directory not found at: {kb_path}")
        return

    raw_documents = []
    document_metadata = []

    for file_path in sorted(kb_path.rglob("*.txt")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    raw_documents.append(content)
                    document_metadata.append({
                        "source": file_path.name,
                        "category": file_path.parent.name if file_path.parent != kb_path else "general"
                    })
        except (IOError, UnicodeDecodeError) as e:
            logger.error(f"Error reading file {file_path}: {e}")

    if not raw_documents:
        logger.warning("No valid text documents found in knowledge base.")
        return

    logger.info(f"Loaded {len(raw_documents)} medical knowledge source files from: {kb_path}")

    chunk_size = 500
    chunk_overlap = 50
    documents = []
    metadata_list = []

    for i, doc in enumerate(raw_documents):
        start = 0
        while start < len(doc):
            end = start + chunk_size
            chunk = doc[start:end].strip()
            if chunk:
                documents.append(chunk)
                metadata_list.append(document_metadata[i].copy())
            start += (chunk_size - chunk_overlap)

    logger.info(f"{len(documents)} chunks after overlap splitting")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Loading embedding model ({config.RETRIEVER_MODEL_NAME}) on device: {device}")
    model = SentenceTransformer(config.RETRIEVER_MODEL_NAME, device=device)

    logger.info("Computing dense vectors for knowledge chunks...")
    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")

    dimension = embeddings.shape[1]

    logger.info(f"indexing {len(documents)} chunks (dim={dimension})")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    doc_mapping_path = index_path.with_name(index_path.stem + "_docs.npy")
    meta_mapping_path = index_path.with_name(index_path.stem + "_meta.npy")
    corpus_path = index_path.with_name(index_path.stem + "_corpus.npy")

    np.save(str(doc_mapping_path), np.array(documents, dtype=object))
    np.save(str(meta_mapping_path), np.array(metadata_list, dtype=object))
    np.save(str(corpus_path), np.array(documents, dtype=object))

    logger.success(f"FAISS index saved successfully to: {index_path}")
    logger.info(f"Document mapping saved: {doc_mapping_path}")
    logger.info(f"Metadata mapping saved: {meta_mapping_path}")
    logger.info(f"BM25 corpus saved: {corpus_path}")
    logger.info(f"Total vector count in index: {index.ntotal}")

if __name__ == "__main__":
    KB_DIR = config.PROJECT_ROOT / "knowledge_base"
    INDEX_OUTPUT = config.INDEX_PATH
    setup_rag_index(KB_DIR, INDEX_OUTPUT)