import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

class ProjectConfig(BaseSettings):
    """
    Centralized configuration for the Diabetes Clinical Agent project.
    Uses pydantic-settings to load from environment variables or .env file.
    """
    # Project Paths
    PROJECT_ROOT: Path = Path(os.getcwd())
    PROCESSED_DIR: Path = Field(default=Path("processed_data"))
    
    # Raw Data Path (The source of truth)
    RAW_DATA_PATH: Path = Field(default=Path("Diabetes paper/diabetic_data.csv"))
    
    # RAG Configuration
    INDEX_PATH: Path = Field(default=Path("processed_data/medical_kb.index"))
    DOCS_PATH: Path = Field(default=Path("processed_data/medical_kb_docs.npy"))
    RETRIEVER_MODEL_NAME: str = "paraphrase-multilingual-MiniLM-L12-v2"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RETRIEVAL_K: int = 10
    RERANK_K: int = 3
    BM25_K: int = 5
    HYBRID_WEIGHT: float = 0.5
    use_metadata_filtering: bool = True

    # LLM Configuration
    LLM_MODEL_NAME: str = "medictron-7b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_PROVIDER_TYPE: str = "ollama"

    # Processed Data Paths
    TRAIN_DATA_PATH: Path = Field(default=Path("processed_data/train.csv"))
    TEST_DATA_PATH: Path = Field(default=Path("processed_data/test.csv"))
    TRAIN_WITH_NOTES_PATH: Path = Field(default=Path("processed_data/train_with_notes.csv"))
    TEST_WITH_NOTES_PATH: Path = Field(default=Path("processed_data/test_with_notes.csv"))
    MODEL_PAYLOAD_PATH: Path = Field(default=Path("processed_data/optimized_xgb_pipeline.joblib"))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Singleton instance for easy access
config = ProjectConfig()
