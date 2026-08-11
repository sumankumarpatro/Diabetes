from loguru import logger
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path
from config import config
from rank_bm25 import BM25Okapi

class RAGRetriever:
    def __init__(self, index_path: Path = None, docs_mapping_path: Path = None, model_name: str = None):
        self.index_path = index_path or config.INDEX_PATH
        self.docs_mapping_path = docs_mapping_path or config.DOCS_PATH
        self.model_name = model_name or config.RETRIEVER_MODEL_NAME
        self.reranker_model_name = config.RERANKER_MODEL_NAME
        self.index = None
        self.documents = None
        self.model = None
        self.reranker = None

    def load(self) -> None:
        if not self.index_path.exists() or not self.docs_mapping_path.exists():
            logger.error(f"Missing files: index={self.index_path}, docs={self.docs_mapping_path}")
            raise FileNotFoundError("FAISS index or document mapping not found.")

        self._verify_integrity()

        logger.info(f"Loading FAISS index from: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        
        logger.info(f"Loading document mapping from: {self.docs_mapping_path}")
        self.documents = np.load(self.docs_mapping_path, allow_pickle=True)

        # Load metadata mapping
        meta_mapping_path = self.docs_mapping_path.with_name(self.docs_mapping_path.stem.replace("_docs", "_meta") + ".npy")
        if meta_mapping_path.exists():
            logger.info(f"Loading metadata mapping from: {meta_mapping_path}")
            self.metadata = np.load(str(meta_mapping_path), allow_pickle=True)
        else:
            logger.warning("Metadata mapping not found. Metadata-based filtering will be disabled.")
            self.metadata = None

        # NEW: Load BM25 corpus
        corpus_path = self.docs_mapping_path.with_name(self.docs_mapping_path.stem.replace("_docs", "_corpus") + ".npy")
        if corpus_path.exists():
            logger.info(f"Loading BM25 corpus from: {corpus_path}")
            self.corpus = np.load(str(corpus_path), allow_pickle=True)
            # Tokenize corpus for BM25
            tokenized_corpus = [doc.lower().split() for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            logger.warning("BM25 corpus not found. Hybrid search will fall back to dense-only.")
            self.bm25 = None
        
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        logger.info(f"Loading re-ranker model: {self.reranker_model_name}")
        self.reranker = CrossEncoder(self.reranker_model_name)
        
        logger.info("Retriever loaded successfully.")

    def retrieve(self, query: str, k: int = None, category_filter: str = None) -> list[str]:
        if self.index is None or self.model is None:
            raise RuntimeError("Retriever not loaded. Call load() first.")
        
        k = k or config.RETRIEVAL_K

        # 1. Metadata-based Filtering (Pre-filtering)
        candidate_indices = list(range(len(self.documents)))
        if category_filter and hasattr(self, 'metadata') and self.metadata is not None:
            logger.info(f"Filtering candidates by category: {category_filter}")
            candidate_indices = [
                i for i, meta in enumerate(self.metadata) 
                if meta.get('category') == category_filter
            ]
            if not candidate_indices:
                logger.warning(f"No documents found for category: {category_filter}")
                return []

        # 2. Dense Retrieval (Vector Search)
        query_embedding = self.model.encode([query]).astype('float32')
        
        # We search for a larger number of candidates to allow for filtering and BM25 overlap
        search_k = k * 5 
        distances, indices = self.index.search(query_embedding, search_k)

        # Filter indices to only those in candidate_indices
        dense_retrieved_indices = [idx for idx in indices[0] if idx < len(self.documents) and idx in candidate_indices]
        dense_retrieved_docs = [self.documents[idx] for idx in dense_retrieved_indices]

        # 3. Sparse Retrieval (BM25)
        bm25_retrieved_docs = []
        if self.bm25:
            tokenized_query = query.lower().split()
            # Get top BM25 candidates
            bm25_scores = self.bm25.get_scores(tokenized_query)
            # Get top B_K indices from BM25
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:config.BM25_K]
            
            for idx in bm25_top_indices:
                doc = self.corpus[idx]
                # Check if this doc is also in our candidate list (via metadata check)
                # For efficiency, we trust the corpus/metadata alignment
                bm25_retrieved_docs.append(doc)

        # 4. Hybrid Scoring (Weighted Fusion)
        # Combine dense and sparse candidates
        combined_docs = list(set(dense_retrieved_docs + bm25_retrieved_docs))
        
        if not combined_docs:
            return []

        # Re-rank the combined list using CrossEncoder
        if self.reranker and combined_docs:
            logger.info(f"Re-ranking {len(combined_docs)} hybrid candidates...")
            pairs = [[query, doc] for doc in combined_docs]
            scores = self.reranker.predict(pairs)
            
            scored_docs = sorted(zip(combined_docs, scores), key=lambda x: x[1], reverse=True)
            retrieved_docs = [doc for doc, score in scored_docs[:config.RERANK_K]]
            logger.debug(f"Re-ranked to top {len(retrieved_docs)} documents.")
            return retrieved_docs
        
        return dense_retrieved_docs[:k]

    def _verify_integrity(self) -> None:
        pass

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
            results = retriever.retrieve(q, k=5)
            if results:
                logger.info(f"Retrieved Context: {results[0]}")
            else:
                logger.warning("No relevant context found.")
    except Exception as e:
        logger.exception(f"Failed to run RAGRetriever test: {e}")
