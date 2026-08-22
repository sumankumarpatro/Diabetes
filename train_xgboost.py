import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

from pathlib import Path
import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    f1_score, fbeta_score, make_scorer, precision_score, recall_score,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from config import config

optuna.logging.set_verbosity(optuna.logging.ERROR)

def get_feature_columns(X: pd.DataFrame):
    exclude_cols = ['clinical_note', 'index', 'split', 'id', 'patient_id', 'encounter_id', 'patient_nbr', 'readmitted_binary']
    available_cols = [c for c in X.columns if c not in exclude_cols]
    numeric_cols = [c for c in available_cols if np.issubdtype(X[c].dtype, np.number) and not c.startswith('symptom_')]
    categorical_cols = [c for c in available_cols if c not in numeric_cols]
    return numeric_cols, categorical_cols

def build_preprocessor(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

def build_classifier(scale_pos_weight=1.0, **kwargs):
    return xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        tree_method='hist',
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        **kwargs,
    )

def build_pipeline(preprocessor, classifier=None):
    if classifier is None:
        classifier = build_classifier()
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

def objective(trial, preprocessor, X_train, y_train, sqrt_scale: float, total_trials: int):
    trial_num = trial.number + 1
    logger.info(f"▶️  [Trial {trial_num:02d}/{total_trials:02d}] Evaluating sampled hyperparameters...")

    param = {
        'classifier__n_estimators': trial.suggest_int('classifier__n_estimators', 80, 350),
        'classifier__max_depth': trial.suggest_int('classifier__max_depth', 3, 7),
        'classifier__learning_rate': trial.suggest_float('classifier__learning_rate', 0.02, 0.15, log=True),
        'classifier__subsample': trial.suggest_float('classifier__subsample', 0.7, 1.0),
        'classifier__colsample_bytree': trial.suggest_float('classifier__colsample_bytree', 0.7, 1.0),
        'classifier__min_child_weight': trial.suggest_int('classifier__min_child_weight', 2, 7),
        'classifier__gamma': trial.suggest_float('classifier__gamma', 0.1, 3.0),
        'classifier__reg_alpha': trial.suggest_float('classifier__reg_alpha', 1e-2, 5.0, log=True),
        'classifier__reg_lambda': trial.suggest_float('classifier__reg_lambda', 1e-2, 5.0, log=True),
        # Tuned around square-root scaling (prevents massive false alarm spikes)
        'classifier__scale_pos_weight': trial.suggest_float('classifier__scale_pos_weight', 1.5, sqrt_scale * 1.5),
    }
    
    base_classifier = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        tree_method='hist',
        n_jobs=1
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', base_classifier)
    ])
    pipeline.set_params(**param)
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    # AUPRC (Average Precision) optimizes precision-recall discrimination
    score = cross_val_score(pipeline, X_train, y_train, n_jobs=-1, cv=cv, scoring='average_precision')
    return score.mean()

def tune_hyperparameters(preprocessor, X_train, y_train, sqrt_scale: float, n_trials=30):
    logger.info(f"Starting Hyperparameter Tuning ({n_trials} trials, AUPRC Precision-Recall Optimization)...")
    
    def trial_completion_callback(study, trial):
        best_score = f"{study.best_value:.4f}" if study.best_value is not None else "N/A"
        logger.info(
            f"🏁 [Trial {trial.number + 1:02d}/{n_trials:02d} Complete] "
            f"AUPRC: {trial.value:.4f} | Best AUPRC: {best_score}"
        )

    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, preprocessor, X_train, y_train, sqrt_scale, n_trials),
        n_trials=n_trials,
        callbacks=[trial_completion_callback]
    )
    logger.success(f"Best hyperparameters found: {study.best_params}")
    return study

def calibrate_clinical_threshold(y_true, y_prob):
    """
    Calibrates threshold using Youden's J Index to maximize Sensitivity (Recall) 
    while preserving Specificity and overall Accuracy.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Youden's J = TPR - FPR = Sensitivity + Specificity - 1
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = float(thresholds[best_idx])
    
    # Clip between 0.15 and 0.65 for stability
    best_threshold = np.clip(best_threshold, 0.15, 0.65)
    return best_threshold

def evaluate_pipeline(pipeline, X, y, threshold=0.5):
    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(y, y_pred),
        'auc': roc_auc_score(y, y_prob),
        'auprc': average_precision_score(y, y_prob),
        'brier': brier_score_loss(y, y_prob),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'f2': fbeta_score(y, y_pred, beta=2.0, zero_division=0),
        'y_prob': y_prob,
        'y_pred': y_pred,
    }

def train_optimized_model(data_path: str, model_output_path: str) -> None:
    data_path = Path(data_path)
    model_output_path = Path(model_output_path)
    
    if not data_path.exists():
        logger.error(f"Error: Data file not found at {data_path}")
        return
        
    logger.info(f"Loading data matrix from: {data_path}")
    df = pd.read_csv(data_path)
    
    if config.MODE == "llm_enhanced":
        symptom_cols = [c for c in df.columns if c.startswith('symptom_')]
        if symptom_cols:
            logger.info(f"[{config.MODE}] Aligning {len(symptom_cols)} symptom columns.")
            df[symptom_cols] = df[symptom_cols].fillna(0).astype(int)
            
    target_col = 'readmitted_binary'
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in {data_path}")
        return
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    string_cols = X.select_dtypes(include=['object']).columns
    if not string_cols.empty:
        X[string_cols] = X[string_cols].astype(str)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Calculate balanced square-root ratio
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    raw_ratio = neg_count / pos_count if pos_count > 0 else 1.0
    sqrt_ratio = np.sqrt(raw_ratio)
    logger.info(f"Class ratio: {raw_ratio:.2f}:1 | Square-Root Weight: {sqrt_ratio:.2f}")
    
    numeric_cols, categorical_cols = get_feature_columns(X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    
    study = tune_hyperparameters(preprocessor, X_train, y_train, sqrt_ratio, n_trials=30)
    
    final_classifier = build_classifier()
    final_pipeline = build_pipeline(preprocessor, classifier=final_classifier)
    final_pipeline.set_params(**study.best_params)
    
    logger.info("Fitting final pipeline on training set...")
    final_pipeline.fit(X_train, y_train)
    
    # Calibrate threshold via Youden's J (Optimal Recall-Specificity Tradeoff)
    raw_val_probs = final_pipeline.predict_proba(X_val)[:, 1]
    calibrated_threshold = calibrate_clinical_threshold(y_val, raw_val_probs)
    
    validation_results = evaluate_pipeline(final_pipeline, X_val, y_val, threshold=calibrated_threshold)
    
    logger.info("--- Validation Set Results (Calibrated Clinical Tradeoff) ---")
    logger.info(f"Validation Accuracy     : {validation_results['accuracy']:.4f}")
    logger.info(f"Validation AUROC        : {validation_results['auc']:.4f}")
    logger.info(f"Validation AUPRC (PR)   : {validation_results['auprc']:.4f}")
    logger.info(f"Validation Recall (Sens): {validation_results['recall']:.4f}")
    logger.info(f"Validation Precision    : {validation_results['precision']:.4f}")
    logger.info(f"Validation F1 / F2      : {validation_results['f1']:.4f} / {validation_results['f2']:.4f}")
    logger.info(f"Calibrated Threshold    : {calibrated_threshold:.4f}")
    
    payload = {
        'model': final_pipeline,
        'threshold': calibrated_threshold,
        'feature_cols': list(X.columns)
    }
    
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, str(model_output_path))
    logger.success(f"Optimized pipeline payload saved to: {model_output_path}")

if __name__ == "__main__":
    out_path = config.MODEL_PAYLOAD_PATH_BASELINE if config.MODE == "baseline" else config.MODEL_PAYLOAD_PATH_LLM_ENHANCED
    train_optimized_model(str(config.active_train_path), str(out_path))