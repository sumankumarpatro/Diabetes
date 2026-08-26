import argparse
import asyncio
import logging
import subprocess
from pathlib import Path
import questionary
from loguru import logger
from tqdm import tqdm

from config import config
from generate_clinical_notes import main as generate_notes_main
from train_multimodal import train_optimized_model

def run_command(command: list[str], description: str):
    """Helper to run a shell command and log progress."""
    logger.info(f"Executing: {' '.join(command)} ({description})")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {result.stderr}")
        raise RuntimeError(f"Command '{description}' failed.")
    logger.info(result.stdout)
    return result.stdout

def run_experiment_pipeline(mode: str, setup: bool):
    """
    Executes the pipeline for any mode: 'baseline', 'bert', 'llm_enhanced', or 'hybrid'.
    """
    steps = []
    if setup:
        if not config.TRAIN_DATA_PATH.exists() or not config.TEST_DATA_PATH.exists():
            steps.append(("Preprocessing raw data", lambda: run_command(["python3", "preprocess_data.py"], "Preprocessing raw data")))
    if mode == "baseline":
        steps.append(("Training BASELINE model", lambda: train_optimized_model("baseline")))
    elif mode == "bert":
        if setup:
            if not config.TRAIN_WITH_NOTES_PATH.exists():
                steps.append(("Generating Hinglish clinical notes", lambda: generate_notes_main()))
            if not config.TRAIN_BERT_EMBEDDINGS_PATH.exists():
                steps.append(("Extracting Bio_ClinicalBERT embeddings", lambda: run_command(["python3", "extract_bert_embeddings.py"], "Extracting BERT embeddings")))
        steps.append(("Training BERT model", lambda: train_optimized_model("bert")))
    elif mode == "llm_enhanced":
        if setup:
            if not config.TRAIN_WITH_NOTES_PATH.exists():
                steps.append(("Generating Hinglish clinical notes", lambda: generate_notes_main()))
            if not config.INDEX_PATH.exists():
                steps.append(("Setting up RAG index", lambda: run_command(["python3", "setup_rag_index.py"], "Setting up RAG index")))
            if not config.TRAIN_WITH_EXTRACTED_FEATURES_PATH.exists():
                steps.append(("Extracting features from notes", lambda: run_command(["python3", "extract_features_from_notes.py"], "Extracting features from notes")))
        steps.append(("Training LLM-ENHANCED model", lambda: train_optimized_model("llm_enhanced")))
    elif mode == "hybrid":
        if setup:
            if not config.TRAIN_WITH_NOTES_PATH.exists():
                steps.append(("Generating Hinglish clinical notes", lambda: generate_notes_main()))
            if not config.INDEX_PATH.exists():
                steps.append(("Setting up RAG index", lambda: run_command(["python3", "setup_rag_index.py"], "Setting up RAG index")))
            if not config.TRAIN_BERT_EMBEDDINGS_PATH.exists():
                steps.append(("Extracting Bio_ClinicalBERT embeddings", lambda: run_command(["python3", "extract_bert_embeddings.py"], "Extracting BERT embeddings")))
            if not config.TRAIN_WITH_EXTRACTED_FEATURES_PATH.exists():
                steps.append(("Extracting features from notes", lambda: run_command(["python3", "extract_features_from_notes.py"], "Extracting features from notes")))
        steps.append(("Training HYBRID model", lambda: train_optimized_model("hybrid")))

    if not steps:
        logger.info("No execution steps configured.")
        return

    logger.info(f"🚀 Starting Pipeline: Mode={mode}, Setup={setup}")
    
    for description, step_func in tqdm(steps, desc="Pipeline Progress"):
        logger.info(f"Running: {description}")
        step_func()
    
    logger.success(f"✅ Pipeline completed successfully for [{mode.upper()}] (Setup={setup})!")

def main():
    parser = argparse.ArgumentParser(description="Run Experiment Pipeline.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "bert", "llm_enhanced", "hybrid"],
        help="The mode to run: baseline, bert, llm_enhanced, or hybrid.",
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="If set, only the training step will be executed, skipping preprocessing and feature extraction.",
    )
    parser.add_argument(
        "--setup_required",
        action="store_true",
        help="If set, the full setup (preprocessing, notes generation, etc.) will be executed.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("experiments"),
        help="Directory to save the trained model and results.",
    )
    args = parser.parse_args()

    # If no mode is provided via CLI, ask interactively
    if args.mode is None:
        mode = questionary.select(
            "Select Experiment Mode:",
            choices=["baseline", "bert", "llm_enhanced", "hybrid"]
        ).ask()
    else:
        mode = args.mode

    if args.train_only:
        setup_required = False
    elif args.setup_required:
        setup_required = True
    else:
        if args.mode is None:
            setup_required = questionary.confirm(
                "Do you want to run the full setup (preprocessing, notes generation, etc.)?",
                default=False
            ).ask()
        else:
            setup_required = False

    experiments_dir = args.output_dir
    experiments_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_experiment_pipeline(mode, setup_required)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        logger.error(f"Pipeline failed: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    main()
