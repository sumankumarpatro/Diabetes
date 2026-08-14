import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import faiss
from loguru import logger
import numpy as np
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
        self.index = None
        self.documents = None
        self.model = None
        self.reranker = None
        self.metadata = None
        self.corpus = None
        self.bm25 = None

        # Dedicated isolated background computation pool for CPU/GPU matrix mathematics.
        # This keeps the main async orchestrator thread perfectly free to process network calls.
        self._math_executor = ThreadPoolExecutor(max_workers=8)

    def load(self) -> None:
        if not self.index_path.exists() or not self.docs_mapping_path.exists():
            logger.error(
                f"Missing files: index={self.index_path}, docs={self.docs_mapping_path}"
            )
            raise FileNotFoundError("FAISS index or document mapping not found.")

        self._verify_integrity()
        logger.info(f"Loading FAISS index from: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        logger.info(
            f"Loading document mapping from: {self.docs_mapping_path}"
        )
        self.documents = np.load(self.docs_mapping_path, allow_pickle=True)
        meta_mapping_path = self.docs_mapping_path.with_name(
            self.docs_mapping_path.stem.replace("_docs", "_meta") + ".npy"
        )
        if meta_mapping_path.exists():
            logger.info(f"Loading metadata mapping from: {meta_mapping_path}")
            self.metadata = np.load(str(meta_mapping_path), allow_pickle=True)
        else:
            logger.warning(
                "Metadata mapping not found. Metadata-based filtering will be disabled."
            )
            self.metadata = None
        corpus_path = self.docs_mapping_path.with_name(
            self.docs_mapping_path.stem.replace("_docs", "_corpus") + ".npy"
        )
        if corpus_path.exists():
            logger.info(f"Loading BM25 corpus from: {corpus_path}")
            self.corpus = np.load(str(corpus_path), allow_pickle=True)

            # Tokenize corpus for BM25
            tokenized_corpus = [doc.lower().split() for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            logger.warning(
                "BM25 corpus not found. Hybrid search will fall back to dense-only."
            )
            self.bm25 = None

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        logger.info(f"Loading re-ranker model: {self.reranker_model_name}")
        self.reranker = CrossEncoder(self.reranker_model_name)
        logger.info("Retriever loaded successfully.")

    async def retrieve(
        self, query: str, k: int = None, category_filter: str = None
    ) -> list[str]:
        """
        Asynchronously searches the hybrid dense-sparse vector space.
        Offloads computational bottlenecks to a specialized executor to protect runtime concurrency.
        """
        if self.index is None or self.model is None:
            raise RuntimeError("Retriever not loaded. Call load() first.")

        k = k or config.RETRIEVAL_K
        loop = asyncio.get_running_loop()

        # Safely offload the dense encoding, sparse lookup, and reranking matrix mathematics
        # out of the main thread pool loop.
        results = await loop.run_in_executor(
            self._math_executor,
            self._sync_retrieval_pipeline,
            query,
            k,
            category_filter,
        )
        return results

    def _sync_retrieval_pipeline(
        self, query: str, k: int, category_filter: str = None
    ) -> list[str]:
        """
        Internal pure mathematical synchronous pipeline worker running inside the ThreadPoolExecutor.
        Preserves your exact business, search, and filtration logic cleanly.
        """
        candidate_indices = list(range(len(self.documents)))
        if (
            category_filter
            and hasattr(self, "metadata")
            and self.metadata is not None
        ):
            candidate_indices = [
                i
                for i, meta in enumerate(self.metadata)
                if meta.get("category") == category_filter
            ]

        if not candidate_indices:
            return []
        query_embedding = self.model.encode([query]).astype("float32")
        search_k = k * 5
        distances, indices = self.index.search(query_embedding, search_k)

        # Filter indices to only those in candidate_indices
        dense_retrieved_indices = [
            idx
            for idx in indices[0]
            if idx < len(self.documents) and idx in candidate_indices
        ]
        dense_retrieved_docs = [
            self.documents[idx] for idx in dense_retrieved_indices
        ]
        bm25_retrieved_docs = []
        if self.bm25:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_top_indices = np.argsort(bm25_scores)[::-1][: config.BM25_K]
            for idx in bm25_top_indices:
                doc = self.corpus[idx]
                bm25_retrieved_docs.append(doc)
        combined_docs = list(
            set(dense_retrieved_docs + bm25_retrieved_docs)
        )
        if not combined_docs:
            return []

        # Re-rank the combined list using CrossEncoder
        if self.reranker and combined_docs:
            pairs = [[query, doc] for doc in combined_docs]
            scores = self.reranker.predict(pairs)
            scored_docs = sorted(
                zip(combined_docs, scores), key=lambda x: x[1], reverse=True
            )
            retrieved_docs = [
                doc for doc, score in scored_docs[: config.RERANK_K]
            ]
            return retrieved_docs

        return dense_retrieved_docs[:k]

    def _verify_integrity(self) -> None:
        pass

    def close(self):
        """Releases systemic threads gracefully upon teardown."""
        self._math_executor.shutdown(wait=True)


if __name__ == "__main__":
    # Test script adapted to execute within an async runtime context loop
    async def test_runner():
        retriever = RAGRetriever()
        try:
            retriever.load()
            test_queries = [
                "What are the symptoms of diabetes?",
                "How to treat hypoglycemia?",
                "Patient has high blood sugar.",
            ]
            logger.info("--- Testing Async RAG Retrieval ---")
            for q in test_queries:
                logger.info(f"Query: {q}")
                results = await retriever.retrieve(q, k=5)
                if results:
                    logger.info(f"Retrieved Context: {results[0][:120]}...")
                else:
                    logger.warning("No relevant context found.")
            retriever.close()
        except Exception as e:
            logger.exception(f"Failed to run RAGRetriever test: {e}")

    asyncio.run(test_runner())
