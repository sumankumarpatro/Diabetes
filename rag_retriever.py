from loguru import logger
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from config import config

class RAGRetriever:
    def __init__(self, index_path: Path = None, docs_mapping_path: Path = None, model_name: str = None):
        self.index_path = index_path or config.INDEX_PATH
        self.docs_mapping_path = docs_mapping_path or config.DOCS_PATH
        self.model_name = model_name or config.RETRIEVER_MODEL_NAME
        self.index = None
        self.documents = None
        self.model = None

    def load(self) -> None:
        if not self.index_path.exists() or not self.docs_mapping_path.exists():
            logger.error(f"Missing files: index={self.index_path}, docs={self.docs_mapping_path}")
            raise FileNotFoundError("FAISS index or document mapping not found.")

        self._verify_integrity()

        logger.info(f"Loading FAISS index from: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        
        logger.info(f"Loading document mapping from: {self.docs_mapping_path}")
        self.documents = np.load(self.docs_mapping_path, allow_pickle=True)
        
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("Retriever loaded successfully.")

    def _verify_integrity(self):
        """
        Verifies the integrity of the loaded index and documents using basic checks.
        """
        if self.index_path.stat().st_size == 0 or self.docs_mapping_path.stat().st_size == 0:
            raise ValueError("Loaded index or document mapping is empty.")
        logger.debug("Integrity check passed (basic existence and size check).")

    def retrieve(self, query: str, k: int = None) -> list[str]:
        if self.index is None or self.model is None:
            raise RuntimeError("Retriever not loaded. Call load() first.")
        
        k = k or config.RETRIEVAL_K
        query_embedding = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        retrieved_docs = []
        for idx in indices[0]:
            if idx < len(self.documents):
                ret_doc = self.documents[idx]
                retrieved_docs.append(ret_doc)
        
        logger.debug(f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}...")
        return retrieved_docs

    def token_model_is_none(self) -> bool:
        return self.model is None

if __name__ == "__main__":
    # Initialize and load using config
    retriever = RAGRetriever()
    try:
        retriever.load()

        # Test queries
        test_queries = [
            "What are the symptoms of diabetes?",
            "How to treat hypoglycemia?",
            "Patient has high blood sugar."
        ]

        logger.info("--- Testing RAG Retrieval ---")
        for q in test_queries:
            logger.info(f"Query: {q}")
            results = retriever.retrieve(q, k=1)
            if results:
                logger.info(f"Retrieved Context: {results[0]}")
            else:
                logger.warning("No relevant context found.")
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        logger.exception(f"Failed to run RAGRetriever test: {e}")
