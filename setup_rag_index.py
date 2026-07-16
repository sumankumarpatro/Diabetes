import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def setup_rag_index(kb_dir, index_output_path):
    """
    Loads text files from the knowledge base, creates embeddings, 
    and saves a FAISS index.
    """
    if not os.path.exists(kb_dir):
        print(f"Error: Knowledge base directory {kb_dir} not found.")
        return
    documents = []
    for filename in os.listdir(kb_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(kb_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())
    
    if not documents:
        print("No documents found in the knowledge base.")
        return

    print(f"Loaded {len(documents)} documents from {kb_dir}")
    # Using a lightweight model suitable for clinical/text tasks
    print("Loading embedding model (sentence-transformers)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Generating embeddings for documents...")
    embeddings = model.encode(documents)
    dimension = embeddings.shape[1]
    print(f"Creating FAISS index with dimension {dimension}...")
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    os.makedirs(os.path.dirname(index_output_path), exist_ok=True)
    faiss.write_index(index, index_output_path)
    
    # We also need to save the documents to retrieve them later
    doc_mapping_path = index_output_path.replace(".index", "_docs.npy")
    np.save(doc_mapping_path, np.array(documents, dtype=object))

    print(f"FAISS index saved to: {index_output_path}")
    print(f"Document mapping saved to: {doc_mapping_path}")
    print(f"Total vectors in index: {index.ntotal}")

if __name__ == "__main__":
    KB_DIR = "/Users/unasumankumarpatro/Documents/Diabetes/knowledge_base"
    INDEX_OUTPUT = "/Users/unasumankumarpatro/Documents/Diabetes/processed_data/medical_kb.index"

    setup_rag_index(KB_DIR, INDEX_OUTPUT)
