# Clinical Decision Support Agent for Diabetes.

This repository contains a complete pipeline for a Clinical Decision Support Agent. The system is designed to take unstructured, multilingual (Hinglish) clinical notes, retrieve relevant medical context using RAG (Retrieage-Augmented Generation), and extract structured clinical features into a machine-readable JSON format.

## 🚀 Overview

The project implements a RAG-based orchestration pipeline:
1.  **Data Engineering**: Preprocessing of diabetes datasets into structured CSVs.
2.  **Baseline Modeling**: Training XGBoost/Random Forest models for diabetes prediction.
3.  **RAG Setup**: A FAISS-based vector store containing medical knowledge (ICD-10 descriptions, diabetes guidelines).
4.  **Agent Orchestration**: A `ClinicalOrchestratorAgent` that:
    *   Receives a messy clinical note.
    *   Uses semantic search and BM25 to retrieve relevant medical context from the knowledge base.
    *   Augments the note with retrieved context.
    *   Uses a specialized LLM (e.g., Medictron-7B) to parse the augmented text into structured JSON.

## 🛠️ Tech Stack

*   **Language**: Python 3.9+
*   **LLM Orchestration**: `transformers`, `torch`, `ollama`
*   **Vector Database**: `FAISS`
*   **Embeddings**: `sentence-transformers` (paraphrase-multilingual-MiniLM-L12-v2)
*   **Machine Learning**: `XGBoost`, `scikit-learn`, `Optuna`, `imbalanced-learn`
*   **Data Processing**: `pandas`, `numpy`

## 📋 Getting Started

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Diabetes
```

### 2. Set Up Environment
It is recommended to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Experiment Pipeline
The `run_experiment.py` script provides an automated way to execute the full pipeline or just the training step.

**Option A: Full End-to-End Pipeline (Preprocess $\rightarrow$ Notes $\rightarrow$ RAG $\rightarrow$ Features $\rightarrow$ Train)**
This is the most thorough method. It handles everything from raw data to the final model.
```bash
python3 run_experiment.py --mode llm_enhanced
```

**Option B: Training Only (Skip Setup)**
If you have already performed the preprocessing and feature extraction (e.g., you've run the pipeline once or manually created the features), you can jump straight to training to save time.
```bash
python3 run_experiment.py --mode llm_enhanced --train_only
```

**Option C: Baseline Mode (Standard Training)**
If you only want to run the baseline training on the original dataset:
```bash
python3 run_experiment.py --mode baseline
```

### 5. Setup Knowledge Base (RAG)
Before running the agent, you must initialize the FAISS index:
```bash
python3 setup_rag_index.py
```

### 6. Run the Clinical Agent
To test the orchestration pipeline with a sample Hinglish note:
```bash
python3 clinical_agent.py
```

## 📂 Project Structure

*   `run_experiment.py`: The master script to orchestrate the full pipeline or just the training step.
*   `clinical_agent.py`: The main Orchestrator Agent logic.
*   `rag_retriever.py`: The retrieval component for searching the FAISS index.
*   `llm_interface.py`: Interface for interacting with LLMs (Transformers/Ollama).
*   `setup_rag_index.py`: Script to build the vector store from text files.
*   `preprocess_data.py`: Data cleaning and feature engineering pipeline.
*   `train_baseline.py`: Training script for the predictive baseline model.
*   `knowledge_base/`: Contains the raw medical text files used for RAG.
*   `processed_data/`: Contains the FAISS index, embeddings, and trained models.

## 🧬 Model Architecture
The agent uses a **Retrieval-Augmented Generation (RAG)** architecture. By augmenting the unstructured clinical note with high-quality medical context, the LLM is significantly less prone to hallucinations and can accurately extract clinical entities even from noisy, multilingual input.

The predictive component uses an **Optimized XGBoost Pipeline** featuring:
* **Hyperparameter Tuning**: Automated optimization via `Optuna`.
* **Class Imbalance Handling**: `SMOTENC` for synthetic oversampling of categorical features and `scale_pos_weight` for cost-sensitive learning.
* **Native Categorical Support**: Leveraging XGBoost's internal handling of categorical data types.

## 🩺 Model Deployment Guide (Medictron-7B via Ollama)
<!-- https://huggingface.co/nikitaredy/medictron-7B -->
If you want to use the specialized `medictron-7B` model (a LoRA adapter) via Ollama, follow these steps. Note that since the repository contains an adapter, you must first merge it with the base `BioMistral-7B` model.

### **Prerequisites**
*   `ollama` installed and running.
*   `git`, `python3`, and `pip` installed.
*   Sufficient disk space (~20GB) and RAM for model merging.

### **Step 1: Download the Adapter Weights**
Download the adapter weights from Hugging Face into a local folder:
```bash
pip install huggingface_hub
export HF_TOKEN=<YOUR_TOKEN> && python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='nikitaredy/medictron-7B', local_dir='./medictron-7B', local_dir_use_symlinks=False)"
```

### **Step 2: Merge Adapter with Base Model**
Create a script named `merge_model.py` to merge the adapter with the `BioMistral-7B` base model.

```bash
python3 merge_model.py
```

### **Step   3: Convert to GGUF Format**
Clone `llama.cpp` and use its conversion script to create a GGUF file.

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt

# Convert the merged model
python3 convert_hf_to_gguf.py ../medictron-7B-merged --outfile ../medictron-7B.gguf
```

### **Step 4: Register in Ollama**
Create a `Modelfile` in your project root:

**`Modelfile` content:**
```dockerfile
FROM /path/to/your/project/medictron-7B.gguf

PARAMETER temperature 0.3

SYSTEM """You are a helpful, respectful, and honest medical assistant. Answer questions accurately based on your training."""
```

Finally, create and run the model in Ollama:
```bash
ollama create medictron-7b -f Modelfile
ollama run medictron-7b
```
