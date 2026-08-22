import argparse
import logging
import subprocess
from pathlib import Path
from config import config
from train_xgboost import train_optimized_model
from loguru import logger
from generate_clinical_notes import main as generate_notes_main
from extract_features_from_notes import extract_features_from_dataset_async_wrapper # We'll need to create this wrapper if it doesn't exist
import asyncio
import questionary
from tqdm import tqdm

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
    Executes the pipeline.
    
    If mode="baseline":
        If setup=True: Preprocess -> Train Baseline.
        If setup=False: Train Baseline.
    
    If mode="llm_enhanced":
        If setup=True: Preprocess -> Generate Notes -> RAG Setup -> Feature Extraction -> Train LLM-enhanced.
        If setup=False: Train LLM-enhanced.
    """
    steps = []
    if mode == "baseline":
        if setup:
            steps = [
                ("Preprocessing raw data", lambda: run_command(["python3", "preprocess_data.py"], "Preprocessing raw data") if not config.TRAIN_DATA_PATH.exists() else logger.info("Skipping preprocessing (exists)")),
                ("Training BASELINE model", lambda: train_optimized_model(str(config.TRAIN_DATA_PATH), str(config.MODEL_PAYLOAD_PATH_BASELINE)))
            ]
        else:
            steps = [
                ("Training BASELINE model", lambda: train_optimized_model(str(config.TRAIN_DATA_PATH), str(config.MODEL_PAYLOAD_PATH_BASELINE)))
            ]
    elif mode == "llm_enhanced":
        if setup:
            steps = [
                ("Preprocessing raw data", lambda: run_command(["python3", "preprocess_data.py"], "Preprocessing raw data") if not config.TRAIN_DATA_PATH.exists() else logger.info("Skipping preprocessing (exists)")),
                ("Generating Hinglish clinical notes", lambda: generate_notes_main()),
                ("Setting up RAG index", lambda: run_command(["python3", "setup_rag_index.py"], "Setting up RAG index")),
                ("Extracting features from notes", lambda: run_command(["python3", "extract_features_from_notes.py"], "Extracting features from notes")),
                ("Training LLM-ENHANCED model", lambda: train_optimized_model(str(config.TRAIN_WITH_EXTRACTED_FEATURES_PATH), str(config.MODEL_PAYLOAD_PATH_LLM_ENHANCED)))
            ]
        else:
            steps = [
                ("Training LLM-ENHANCED model", lambda: train_optimized_model(str(config.TRAIN_WITH_EXTRACTED_FEATURES_PATH), str(config.MODEL_PAYLOAD_PATH_LLM_ENHANCED)))
            ]

    if not steps:
        return

    logger.info(f"🚀 Starting Pipeline: Mode={mode}, Setup={setup}")
    
    for description, step_func in tqdm(steps, desc="Pipeline Progress"):
        logger.info(f"Running: {description}")
        step_func()
    
    logger.success(f"✅ Pipeline completed for {mode} (Setup={setup})!")

def main():
    # For interactive mode, we use questionary. For CLI args, we use argparse.
    # We'll prioritize questionary if no args are passed that conflict.
    
    parser = argparse.ArgumentParser(description="Run Experiment Pipeline.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "llm_enhanced"],
        help="The mode to run: 'baseline' uses raw data, 'llm_enhanced' uses data with extracted features.",
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
            choices=["baseline", "llm_enhanced"]
        ).ask()
    else:
        mode = args.mode

    if args.train_only:
        setup_required = False
    else:
        # If mode was selected interactively, we need to decide on setup
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
        logger.error(f"Pipeline failed: {annotated_error(e)}")
        raise

def annotated_error(e):
    return f"{type(e).__name__}: {e}"

if __name__ == "__main__":
    main()
