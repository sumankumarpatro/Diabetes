import pandas as pd
import joblib
from loguru import logger
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
from pathlib import Path
from config import config
import argparse

def evaluate_model(data_path: str, model_payload_path: str):
    """Loads a trained model payload and evaluates it on an independent test dataset."""
    # Convert string paths to Path objects for consistency
    data_path = Path(data_path)
    model_payload_path = Path(model_payload_path)

    if not data_path.exists():
        logger.error(f"Test data file not found: {data_path}")
        return

    if not model_payload_path.exists():
        logger.error(f"Model payload file not found: {model_payload_path}")
        return

    logger.info(f"Loading test data from: {data_path}")
    df_test = pd.read_csv(data_path)

    if 'readmitted_binary' not in df_test.columns:
        logger.error("Target column 'readmitted_binary' not found in test data.")
        return

    target_col = 'readmitted_binary'
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    logger.info(f"Loading model payload from: {model_payload_path}")
    payload = joblib.load(model_payload_path)
    pipeline = payload['model']
    threshold = payload['threshold']

    logger.info(f"Using optimized threshold: {threshold:.4f}")

    logger.info("Running inference on test set...")
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate metrics
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.info("\n--- Independent Test Set Evaluation ---")
    logger.info(f"AUC-ROC:    {auc:.4f}")
    logger.info(f"Precision:  {precision:.4f}")
    logger.info(f"Recall:     {recall:.4f}")
    logger.info(f"F1-Score:   {f1:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate XGBoost models on the test set.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "llm_enhanced"],
        required=True,
        help="The mode to evaluate: 'baseline' or 'llm_enhanced'.",
    )
    args = parser.parse_args()


    if args.mode == "baseline":
        TEST_DATA_PATH = str(config.TEST_DATA_PATH)
        MODEL_PAYLOAD_PATH = str(config.MODEL_PAYLOAD_PATH_BASELINE)
    elif args.mode == "llm_enhanced":
        TEST_DATA_PATH = str(config.TEST_WITH_EXTRACTED_FEATURES_PATH)
        MODEL_PAYLOAD_PATH = str(config.MODEL_PAYLOAD_PATH_LLM_ENHANCED)

    evaluate_model(TEST_DATA_PATH, MODEL_PAYLOAD_PATH)
