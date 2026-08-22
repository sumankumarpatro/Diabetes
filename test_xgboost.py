import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    average_precision_score, brier_score_loss, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score
)
from config import config

def print_evaluation_metrics(mode: str, metrics: dict, report: str):
    """Prints evaluation metrics to the terminal."""
    border = "=" * 65
    sub_border = "-" * 65
    
    print(f"\n{border}")
    print(f" PROFILE: {mode.upper()} DEPLOYMENT MODEL ".center(65, " "))
    print(f"{border}")
    print(f"  ● OUT-OF-SAMPLE AUC-ROC :  {metrics['auc']:.4f}")
    print(f"  ● AVERAGE PRECISION(PR) :  {metrics['auprc']:.4f}")
    print(f"  ● BRIER SCORE LOSS      :  {metrics['brier']:.4f}")
    print(f"  ● OPTIMIZED F1-SCORE    :  {metrics['f1']:.4f}")
    print(f"  ● MODEL PRECISION       :  {metrics['precision']:.4f}")
    print(f"  ● MODEL RECALL (SENS.)  :  {metrics['recall']:.4f}")
    print(f"  ● DECISION THRESHOLD   :  {metrics['threshold']:.4f}")
    print(f"{sub_border}")
    print(" DETAILED CLASSIFICATION MATRIX ".center(65, " "))
    print(f"{sub_border}")
    indented_report = "\n".join(f"   {line}" for line in report.splitlines())
    print(indented_report)
    print(f"{border}\n")

def evaluate_model(data_path: str, model_payload_path: str, mode: str):
    data_path = Path(data_path)
    model_payload_path = Path(model_payload_path)
    
    if not data_path.exists():
        logger.error(f"Test data file not found: {data_path}")
        return
    if not model_payload_path.exists():
        logger.error(f"Model payload file not found: {model_payload_path}")
        return
        
    logger.info(f"[{mode.upper()}] Loading independent test matrix from: {data_path}")
    df_test = pd.read_csv(data_path)
    
    if mode == "llm_enhanced":
        symptom_cols = [c for c in df_test.columns if c.startswith('symptom_')]
        if symptom_cols:
            df_test[symptom_cols] = df_test[symptom_cols].fillna(0).astype(int)
            
    target_col = 'readmitted_binary'
    if target_col not in df_test.columns:
        logger.error("Target column 'readmitted_binary' not found in test data.")
        return
        
    y_test = df_test[target_col]
    
    logger.info(f"Loading pipeline payload from: {model_payload_path}")
    payload = joblib.load(model_payload_path)
    
    pipeline = payload['model']
    threshold = payload['threshold']
    expected_feature_cols = payload.get('feature_cols', None)
    
    if expected_feature_cols is not None:
        # Align test features to match the exact columns seen during training
        X_test = pd.DataFrame(index=df_test.index)
        for col in expected_feature_cols:
            if col in df_test.columns:
                X_test[col] = df_test[col]
            else:
                X_test[col] = 0
    else:
        exclude_cols = [target_col, 'clinical_note', 'index', 'split', 'id', 'patient_id', 'encounter_id', 'patient_nbr']
        feature_cols = [c for c in df_test.columns if c not in exclude_cols]
        X_test = df_test[feature_cols].copy()
    
    string_cols = X_test.select_dtypes(include=['object']).columns
    if not string_cols.empty:
        X_test[string_cols] = X_test[string_cols].astype(str)
        
    logger.info("Evaluating model predictions on test set...")
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics_payload = {
        'auc': roc_auc_score(y_test, y_prob),
        'auprc': average_precision_score(y_test, y_prob),
        'brier': brier_score_loss(y_test, y_prob),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'threshold': threshold
    }
    
    native_report = classification_report(y_test, y_pred, zero_division=0)
    print_evaluation_metrics(mode, metrics_payload, native_report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate XGBoost models on the test set.")
    parser.add_argument(
        "--mode", 
        choices=["baseline", "llm_enhanced"], 
        required=True, 
        help="The mode to evaluate: 'baseline' or 'llm_enhanced'."
    )
    args = parser.parse_args()
    
    if args.mode == "baseline":
        TEST_DATA_PATH = str(config.TEST_DATA_PATH)
        MODEL_PAYLOAD_PATH = str(config.MODEL_PAYLOAD_PATH_BASELINE)
    elif args.mode == "llm_enhanced":
        TEST_DATA_PATH = str(config.TEST_WITH_EXTRACTED_FEATURES_PATH)
        MODEL_PAYLOAD_PATH = str(config.MODEL_PAYLOAD_PATH_LLM_ENHANCED)
        
    evaluate_model(TEST_DATA_PATH, MODEL_PAYLOAD_PATH, args.mode)