import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import faiss
from loguru import logger
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from config import config

class RAGRetriever:
    def __init__(
        self,
        index_path: Path = None,
        docs_mapping_path: Path = None,
        model_name: str = None,
    ):
        self.index_path = index_path or config.INDEX_PATH
        self.docs_mapping_path = docs_mapping_path or config.DOCS_PATH
        self.model_name = model_name or config.RETRIEVER_MODEL_NAME
        self.reranker_model_name = config.RERANKER_MODEL_NAME
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        self.index = None
        self.documents = None
        self.model = None
        self.reranker = None
        self.metadata = None
        self.corpus = None
        self.bm25 = None

        # Dedicated background executor for GPU/CPU matrix calculations
        self._math_executor = ThreadPoolExecutor(max_workers=4)

    def load(self) -> None:
        if not self.index_path.exists() or not self.docs_mapping_path.exists():
            logger.error(f"Missing RAG files: index={self.index_path}, docs={self.docs_mapping_path}")
            raise FileNotFoundError("FAISS index or document mapping array not found.")

        logger.info(f"Loading FAISS index from: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        logger.info(f"Loading document mapping from: {self.docs_mapping_path}")
        self.documents = np.load(str(self.docs_mapping_path), allow_pickle=True)
        meta_mapping_path = self.docs_mapping_path.with_name(
            self.docs_mapping_path.stem.replace("_docs", "_meta") + ".npy"
        )
        if meta_mapping_path.exists():
            logger.info(f"Loading metadata mapping from: {meta_mapping_path}")
            self.metadata = np.load(str(meta_mapping_path), allow_pickle=True)
        else:
            self.metadata = None
        corpus_path = self.docs_mapping_path.with_name(
            self.docs_mapping_path.stem.replace("_docs", "_corpus") + ".npy"
        )
        if corpus_path.exists():
            logger.info(f"Loading BM25 corpus from: {corpus_path}")
            self.corpus = np.load(str(corpus_path), allow_pickle=True)
            tokenized_corpus = [doc.lower().split() for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

        logger.info(f"Loading embedding model ({self.model_name}) on device: {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

        logger.info(f"Loading neural re-ranker ({self.reranker_model_name}) on device: {self.device}")
        self.reranker = CrossEncoder(self.reranker_model_name, device=self.device)
        logger.success("RAGRetriever loaded successfully.")

    async def retrieve(
        self, query: str, k: int = None, category_filter: str = None
    ) -> list[str]:
        """
        Asynchronously searches the hybrid dense-sparse space and applies neural cross-reranking.
        """
        if self.index is None or self.model is None:
            raise RuntimeError("Retriever is not loaded. Call load() first.")

        k = k or config.RETRIEVAL_K
        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(
            self._math_executor,
            self._sync_retrieval_pipeline,
            query,
            k,
            category_filter,
        )

    def _sync_retrieval_pipeline(
        self, query: str, k: int, category_filter: str = None
    ) -> list[str]:
        candidate_indices = list(range(len(self.documents)))
        if category_filter and self.metadata is not None:
            candidate_indices = [
                i for i, meta in enumerate(self.metadata)
                if isinstance(meta, dict) and meta.get("category") == category_filter
            ]

        if not candidate_indices:
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True).astype("float32")
        search_k = min(k * 4, len(self.documents))
        distances, indices = self.index.search(query_embedding, search_k)

        dense_retrieved_docs = [
            self.documents[idx]
            for idx in indices[0]
            if idx < len(self.documents) and idx in candidate_indices
        ]

        bm25_retrieved_docs = []
        if self.bm25:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:config.BM25_K]
            bm25_retrieved_docs = [self.corpus[idx] for idx in bm25_top_indices if idx < len(self.corpus)]

        combined_docs = list(dict.fromkeys(dense_retrieved_docs + bm25_retrieved_docs))
        if not combined_docs:
            return []

        if self.reranker and combined_docs:
            pairs = [[query, doc] for doc in combined_docs]
            scores = self.reranker.predict(pairs)
            scored_docs = sorted(zip(combined_docs, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scored_docs[:config.RERANK_K]]

        return combined_docs[:k]

    def close(self):
        """Shuts down background compute threads."""
        self._math_executor.shutdown(wait=False)

if __name__ == "__main__":
    async def test_runner():
        retriever = RAGRetriever()
        try:
            retriever.load()
            test_queries = [
                "Patient complains of bahut zyada thakan and high blood sugar.",
                "How to manage insulin adjustments and hypoglycemia risk?",
                "What are the indicators of 30-day diabetic readmissions?"
            ]
            logger.info("=== Testing Hybrid Async RAG Retrieval ===")
            for q in test_queries:
                logger.info(f"Query: {q}")
                results = await retriever.retrieve(q, k=3)
                for i, res in enumerate(results, 1):
                    logger.info(f"  [Match {i}]: {res.strip()[:100]}...")
            retriever.close()
        except Exception as e:
            logger.exception(f"Failed RAGRetriever test: {e}")

    asyncio.run(test_runner())