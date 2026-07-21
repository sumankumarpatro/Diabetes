from loguru import logger
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os
from config import config
from pathlib import Path

def train_baseline_model(train_path: str, val_path: str, model_output_path: str) -> None:
    train_path = Path(train_path)
    val_path = Path(val_path)
    model_output_path = Path(model_output_path)

    if not train_path.exists() or not val_path.exists():
        logger.error(f"Error: Data files not found. Ensure {train_path} and {val_path} exist.")
        return

    logger.info(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    logger.info(f"Loading validation data from: {val_path}")
    val_df = pd.read_csv(val_path)

    # The target variable is 'readmitted_binary'
    target_col = 'readmitted_binary'
    
    # Separate features and target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    logger.info(f"Training shape: {X_train.shape}")
    logger.info(f"Validation shape: {X_val.shape}")

    # Initialize XGBoost Classifier
    # Using scale_pos_weight to handle class imbalance (approx 65/35 ratio)
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"Class imbalance ratio (negative/positive): {ratio:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_scale=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=ratio
    )

    logger.info("Starting training...")
    model.fit(X_train, y_train)
    logger.info("Training complete.")

    # Predictions
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    # Evaluation
    auc = roc_auc_score(y_val, y_prob)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    logger.info("\n--- Baseline Model Evaluation (XGBoost) ---")
    logger.info(f"AUC-ROC:   {auc:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_val, y_pred))

    # Save the model
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(model_output_path))
    logger.success(f"Model saved to: {model_output_path}")

if __name__ == "__main__":
    # Use paths from config
    TRAIN_PATH = str(config.TRAIN_DATA_PATH)
    VAL_PATH = str(config.PROCESSED_DIR / 'val.csv')
    MODEL_SAVE_PATH = str(config.PROCESSED_DIR / 'baseline_xgb_model.joblib')

    train_baseline_model(TRAIN_PATH, VAL_PATH, MODEL_SAVE_PATH)
