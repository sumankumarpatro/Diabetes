import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

class ProjectConfig(BaseSettings):
    """
    Centralized configuration for the Diabetes Clinical Agent project.
    Uses pydantic-settings to load from environment variables or .env file.
    """
    # Execution Mode: 'baseline', 'bert', 'llm_enhanced', or 'hybrid'
    MODE: str = Field(default="baseline")
    # Ablation Mode: 'none', 'without_rag', 'without_negation', 'without_svd', or 'without_bert'
    ABLATION: str = Field(default="none")

    # Project Paths
    PROJECT_ROOT: Path = Path(os.getcwd())
    PROCESSED_DIR: Path = Field(default=Path("processed_data"))
    EXPERIMENTS_DIR: Path = Field(default=Path("experiments"))
    
    # Raw Data Path
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

    BERT_MODEL_NAME: str = "emilyalsentzer/Bio_ClinicalBERT"
    BERT_EMBEDDING_DIM: int = 768
    BERT_PCA_COMPONENTS: int = 32
    TRAIN_BERT_EMBEDDINGS_PATH: Path = Field(default=Path("processed_data/train_bert_embeddings.npy"))
    TEST_BERT_EMBEDDINGS_PATH: Path = Field(default=Path("processed_data/test_bert_embeddings.npy"))

    TRAIN_DATA_PATH: Path = Field(default=Path("processed_data/train.csv"))
    TEST_DATA_PATH: Path = Field(default=Path("processed_data/test.csv"))
    TRAIN_WITH_NOTES_PATH: Path = Field(default=Path("processed_data/train_with_notes.csv"))
    TEST_WITH_NOTES_PATH: Path = Field(default=Path("processed_data/test_with_notes.csv"))
    TRAIN_WITH_EXTRACTED_FEATURES_PATH: Path = Field(default=Path("processed_data/train_with_extracted_features.csv"))
    TEST_WITH_EXTRACTED_FEATURES_PATH: Path = Field(default=Path("processed_data/test_with_extracted_features.csv"))
    
    # Model Payload Paths
    MODEL_PAYLOAD_PATH_BASELINE: Path = Field(default=Path("experiments/xgb_model_baseline.joblib"))
    MODEL_PAYLOAD_PATH_BERT: Path = Field(default=Path("experiments/xgb_model_bert.joblib"))
    MODEL_PAYLOAD_PATH_LLM_ENHANCED: Path = Field(default=Path("experiments/xgb_model_llm_enhanced.joblib"))
    MODEL_PAYLOAD_PATH_HYBRID: Path = Field(default=Path("experiments/xgb_model_hybrid.joblib"))

    @property
    def active_train_path(self) -> Path:
        """ Returns the appropriate training path based on the current MODE. """
        if self.MODE == "baseline":
            return self.TRAIN_DATA_PATH
        elif self.MODE == "bert":
            return self.TRAIN_WITH_NOTES_PATH
        return self.TRAIN_WITH_EXTRACTED_FEATURES_PATH

    @property
    def active_test_path(self) -> Path:
        """ Returns the appropriate testing path based on the current MODE. """
        if self.MODE == "baseline":
            return self.TEST_DATA_PATH
        elif self.MODE == "bert":
            return self.TEST_WITH_NOTES_PATH
        return self.TEST_WITH_EXTRACTED_FEATURES_PATH

    @property
    def active_model_payload_path(self) -> Path:
        """ Returns the target model joblib path for the current MODE and ABLATION. """
        if self.ABLATION and self.ABLATION != "none":
            return self.EXPERIMENTS_DIR / f"xgb_model_{self.MODE}_ablation_{self.ABLATION}.joblib"
        mapping = {
            "baseline": self.MODEL_PAYLOAD_PATH_BASELINE,
            "bert": self.MODEL_PAYLOAD_PATH_BERT,
            "llm_enhanced": self.MODEL_PAYLOAD_PATH_LLM_ENHANCED,
            "hybrid": self.MODEL_PAYLOAD_PATH_HYBRID,
        }
        return mapping.get(self.MODE, self.MODEL_PAYLOAD_PATH_BASELINE)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

config = ProjectConfig()