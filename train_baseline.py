import xgboost as xgb
import optuna
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTENC
from loguru import logger
from config import config
from pathlib import Path

def objective(trial, X, y):
    """
    Optuna objective function for hyperparameter tuning.
    """
    # Identify categorical features for SMOTENC
    cat_features = X.select_dtypes(include=['category', 'object']).columns.tolist()
    
    # Apply SMOTENC to balance the training data for the objective function
    # We use a subset of the data to keep tuning fast
    smote = SMOTENC(categorical_features=cat_features, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'enable_categorical': True
    }

    # Calculate class imbalance ratio for scale_pos_weight
    ratio = (y == 0).sum() / (y == 1).sum()
    param['scale_pos_weight'] = ratio

    clf = xgb.XGBClassifier(**param)
    
    # Use cross-validation to evaluate the hyperparameters
    # Using 3-fold CV to keep it relatively fast during tuning
    score = cross_val_score(clf, X_resampled, y_resampled, n_jobs=-1, cv=3, scoring='roc_auc')
    return score.mean()

def train_optimized_model(train_path: str, val_path: str, model_output_path: str) -> None:
    """
    Trains an optimized XGBoost model using Optuna hyperparameter tuning.
    """
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

    target_col = 'readmitted_binary'
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    # Convert object columns to category for XGBoost
    for col in X_train.select_dtypes(include=['object']).columns:
        X_train[col] = X_train[col].astype('category')
        if col in X_val.columns:
            X_val[col] = X_val[col].astype('category')

    # --- Apply SMOTENC to balance the training data ---
    logger.info("Applying SMOTENC to balance training data...")
    cat_features = X_train.select_dtypes(include=['category', 'object']).columns.tolist()
    smote = SMOTENC(categorical_features=cat_features, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    logger.info(f"Resampled training shape: {X_train_resampled.shape}")

    logger.info(f"Training shape: {X_train.shape}")
    logger.info(f"Validation shape: {X_val.shape}")

    # --- Hyperparameter Tuning with Optuna ---
    logger.info("Starting Hyperparameter Tuning with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train_resampled, y_train_resampled), n_trials=20)

    logger.success(f"Best hyperparameters found: {study.best_params}")

    # --- Final Training with Best Parameters ---
    logger.info("Training final model with best parameters...")
    
    # Re-calculate ratio for the final model (using the original imbalanced data for scale_pos_weight)
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    
    best_params = study.best_params
    best_params['scale_pos_weight'] = ratio
    best_params['use_label_encoder'] = False
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = 42
    best_params['enable_categorical'] = True

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train_resampled, y_train_resampled)

    # Predictions
    y_pred = final_model.predict(X_val)
    y_prob = final_model.predict_proba(X_val)[:, 1]

    # Evaluation
    auc = roc_auc_score(y_val, y_prob)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    logger.info("\n--- Optimized XGBoost Model Evaluation ---")
    logger.info(f"AUC-ROC:   {auc:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_val, y_pred))

    # Save the model
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, str(model_output_path))
    logger.success(f"Optimized model saved to: {model_output_path}")

if __name__ == "__main__":
    TRAIN_PATH = str(config.TRAIN_DATA_PATH)
    VAL_PATH = str(config.PROCESSED_DIR / 'val.csv')
    MODEL_SAVE_PATH = str(config.PROCESSED_DIR / 'baseline_xgb_model.joblib')

    train_optimized_model(TRAIN_PATH, VAL_PATH, MODEL_SAVE_PATH)
