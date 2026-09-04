import argparse
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
from pathlib import Path
import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    classification_report, f1_score, precision_score, recall_score, roc_auc_score
)
from config import config

def print_evaluation_metrics(mode: str, metrics: dict, report: str):
    print(f"\nEvaluation Summary ({mode}):")
    labels = {
        'auc': 'AUC-ROC',
        'auprc': 'Average Precision (PR)',
        'brier': 'Brier Score Loss',
        'accuracy': 'Accuracy',
        'f1': 'F1 Score',
        'precision': 'Precision',
        'recall': 'Recall',
        'threshold': 'Threshold',
    }
    for key, label in labels.items():
        if key in metrics:
            print(f"  {label:<24}: {metrics[key]:.4f}")
    print(f"\nClassification Report:\n{report}\n")

def apply_negation_ablation(df: pd.DataFrame) -> pd.DataFrame:
    for aff in [c for c in df.columns if c.startswith('symptom_') and c.endswith('_affirmed')]:
        base = aff[:-9]
        neg = f"{base}_negated"
        if neg in df.columns:
            df[base] = (df[aff].fillna(0).astype(int) | df[neg].fillna(0).astype(int)).astype(int)
            df.drop(columns=[aff, neg], inplace=True)
        else:
            df[base] = df[aff].fillna(0).astype(int)
            df.drop(columns=[aff], inplace=True)
    return df

def evaluate_model(mode: str, ablation: str = "none"):
    config.MODE = mode
    config.ABLATION = ablation
    payload_path = config.active_model_payload_path

    if not payload_path.exists():
        logger.error(f"Model payload not found at: {payload_path}")
        return

    ablation_title = f" [{ablation.upper()}]" if ablation != "none" else ""
    logger.info(f"Loading [{mode.upper()}]{ablation_title} payload from: {payload_path}")
    payload = joblib.load(payload_path)
    pipeline = payload['model']
    threshold = payload['threshold']
    svd_model = payload.get('svd_model', None)
    expected_cols = payload['feature_cols']

    test_path = config.active_test_path
    logger.info(f"Loading independent test data from: {test_path}")
    df_test = pd.read_csv(test_path)

    if ablation == "without_negation":
        df_test = apply_negation_ablation(df_test)

    symptom_cols = [c for c in df_test.columns if c.startswith('symptom_')]
    if symptom_cols:
        df_test[symptom_cols] = df_test[symptom_cols].fillna(0).astype(int)

    target_col = 'readmitted_binary'
    if target_col not in df_test.columns:
        logger.error(f"Target column '{target_col}' not found in test dataset.")
        return

    y_test = df_test[target_col]

    exclude_cols = ['clinical_note', 'index', 'split', 'id', 'patient_id', 'encounter_id', 'patient_nbr', 'readmitted_binary']
    df_test = df_test.drop(columns=[c for c in exclude_cols if c in df_test.columns], errors='ignore')

    if mode in ["bert", "hybrid"] and ablation != "without_bert":
        bert_test_path = config.TEST_BERT_EMBEDDINGS_PATH
        if not bert_test_path.exists():
            logger.error(f"Test BERT embeddings not found at {bert_test_path}")
            return
        
        test_bert_dense = np.load(str(bert_test_path))
        if ablation == "without_svd":
            n_raw_dims = min(128, test_bert_dense.shape[1])
            for dim in range(n_raw_dims):
                df_test[f'bert_dim_{dim}'] = test_bert_dense[:, dim]
        elif svd_model:
            logger.info("Applying fitted TruncatedSVD to test BERT embeddings...")
            test_bert_svd = svd_model.transform(test_bert_dense)
            for dim in range(config.BERT_PCA_COMPONENTS):
                df_test[f'bert_dim_{dim}'] = test_bert_svd[:, dim]
    missing_cols = [c for c in expected_cols if c not in df_test.columns]
    if missing_cols:
        df_missing = pd.DataFrame(0, index=df_test.index, columns=missing_cols)
        df_test = pd.concat([df_test, df_missing], axis=1)
    
    X_test = df_test[expected_cols].copy()

    string_cols = X_test.select_dtypes(include=['object']).columns
    if not string_cols.empty:
        X_test[string_cols] = X_test[string_cols].astype(str)

    logger.info("Evaluating model predictions on test set...")
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob),
        'auprc': average_precision_score(y_test, y_prob),
        'brier': brier_score_loss(y_test, y_prob),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'threshold': threshold
    }

    report = classification_report(y_test, y_pred, zero_division=0)
    profile_label = f"{mode.upper()}{ablation_title}"
    print_evaluation_metrics(profile_label, metrics, report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate XGBoost models on independent test set.")
    parser.add_argument(
        "--mode", 
        choices=["baseline", "bert", "llm_enhanced", "hybrid"], 
        required=True, 
        help="Evaluation mode"
    )
    parser.add_argument(
        "--ablation",
        choices=["none", "without_rag", "without_negation", "without_svd", "without_bert"],
        default="none",
        help="Ablation configuration to evaluate"
    )
    args = parser.parse_args()

    evaluate_model(args.mode, ablation=args.ablation)