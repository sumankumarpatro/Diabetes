import argparse
import logging
from pathlib import Path
from config import config
from train_xgboost import train_optimized_model
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="Run XGBoost training experiments.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "llm_enhanced"],
        required=True,
        help="The mode to run: 'baseline' uses raw data, 'llm_enhanced' uses data with extracted features.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("experiments"),
        help="Directory to save the trained model and results.",
    )
    args = parser.parse_args()

    # Create experiments directory if it doesn't exist
    experiments_dir = args.output_dir
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # Determine input dataset based on mode
    if args.mode == "baseline":
        data_path = config.TRAIN_DATA_PATH
        logger.info("Running baseline experiment with raw training data.")
    elif args.mode == "llm_enhanced":
        data_path = config.TRAIN_WITH_EXTRACTED_FEATURES_PATH
        logger.info("Running LLM-enhanced experiment with extracted features dataset.")
    else:
        # This part should not be reachable due to argparse choices
        logger.error("Invalid mode selected.")
        return

    # Define model output path
    model_name = f"xgb_model_{args.mode}.joblib"
    model_output_path = experiments_dir / model_name

    # Run the training pipeline
    try:
        logger.info(f"Starting training pipeline for mode: {args.mode}")
        train_optimized_model(str(data_path), str(model_output_path))
        logger.success(f"Experiment {args.mode} completed successfully. Model saved to {model_output_path}")
    except Exception as e:
        logger.error(f"Experiment {args.mode} failed: {e}")
        raise

if __name__ == "__main__":
    main()
