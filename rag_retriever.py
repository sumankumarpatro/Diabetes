import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

class RAGRetriever:
    def __init__(self, index_path, docs_mapping_path, model_name='all-MiniLM-L6-v2'):
        self.index_path = index_path
        self.docs_mapping_path = docs_mapping_path
        self.model_name = model_name
        self.index = None
        self.documents = None
        self.model = None

    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.docs_mapping_path):
            raise FileNotFoundError("FAISS index or document mapping not found.")

        print(f"Loading FAISS index from: {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        
        print(f"Loading document mapping from: {self.docs_mapping_path}")
        self.documents = np.load(self.docs_mapping_path, allow_pickle=True)
        
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print("Retriever loaded successfully.")

    def retrieve(self, query, k=2):
        if self.index is None or self.model is None:
            raise RuntimeError("Retriever not loaded. Call load() first.")
        query_embedding = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        retrieved_docs = []
        for idx in indices[0]:
            if idx < len(self.documents):
                ret_doc = self.documents[idx]
                retrieved_docs.append(ret_doc)
        
        return retrieved_docs

if __name__ == "__main__":
    # Configuration
    PROCESSED_DIR = "/Users/unasumankumarpatro/Documents/Diabetes/processed_data"
    INDEX_PATH = os.path.join(PROCESSED_DIR, 'medical_kb.index')
    DOCS_PATH = os.path.join(PROCESSED_DIR, 'medical_kb_docs.npy')

    # Initialize and load
    retriever = RAGRetriever(INDEX_PATH, DOCS_PATH)
    retriever.load()

    # Test queries
    test_queries = [
        "What are the symptoms of diabetes?",
        "How to treat hypoglycemia?",
        "Patient has high blood sugar."
    ]

    print("\n--- Testing RAG Retrieval ---")
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retriever.retrieve(q, k=1)
        if results:
            print(f"Retrieved Context: {results[0]}")
        else:
            print("No relevant context found.")
