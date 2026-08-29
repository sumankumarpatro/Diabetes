import argparse
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
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    f1_score, fbeta_score, precision_score, recall_score,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from config import config

optuna.logging.set_verbosity(optuna.logging.ERROR)

def get_tabular_columns(X: pd.DataFrame):
    """Identifies numeric and categorical tabular columns, excluding raw text and dense BERT features."""
    exclude_cols = ['clinical_note', 'index', 'split', 'id', 'patient_id', 'encounter_id', 'patient_nbr', 'readmitted_binary']
    available_cols = [c for c in X.columns if c not in exclude_cols and not c.startswith('bert_dim_')]
    
    numeric_cols = [
        c for c in available_cols 
        if np.issubdtype(X[c].dtype, np.number) and not c.startswith('symptom_')
    ]
    categorical_cols = [
        c for c in available_cols 
        if c not in numeric_cols
    ]
    return numeric_cols, categorical_cols

def build_preprocessor(numeric_cols, categorical_cols, bert_cols=None):
    transformers = [
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ]), numeric_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), categorical_cols)
    ]
    
    # Include BERT dense feature columns if present
    if bert_cols:
        transformers.append(('bert', 'passthrough', bert_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder='drop'
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

def calibrate_clinical_threshold(y_true, y_prob):
    """Calibrates threshold via Youden's J Index to maximize Sensitivity & Specificity."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(np.clip(thresholds[best_idx], 0.15, 0.65))

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

def train_optimized_model(mode: str, ablation: str = "none") -> None:
    config.MODE = mode
    config.ABLATION = ablation
    ablation_str = f" [ABLATION: {ablation.upper()}]" if ablation != "none" else ""
    logger.info(f"=== Starting Training Pipeline: Mode [{mode.upper()}]{ablation_str} ===")
    
    if mode in ["baseline", "bert"]:
        data_path = config.TRAIN_DATA_PATH if mode == "baseline" else config.TRAIN_WITH_NOTES_PATH
    else:
        data_path = config.TRAIN_WITH_EXTRACTED_FEATURES_PATH

    if not data_path.exists():
        logger.error(f"Training data not found at: {data_path}")
        return

    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    if ablation == "without_negation":
        logger.info("Applying Ablation: Merging affirmed and negated symptom flags...")
        symptom_affirmed_cols = [c for c in df.columns if c.startswith('symptom_') and c.endswith('_affirmed')]
        for aff_col in symptom_affirmed_cols:
            base_symptom = aff_col.replace('_affirmed', '')
            neg_col = f"{base_symptom}_negated"
            if neg_col in df.columns:
                df[base_symptom] = (df[aff_col].fillna(0).astype(int) | df[neg_col].fillna(0).astype(int)).astype(int)
                df.drop(columns=[aff_col, neg_col], inplace=True)
            else:
                df[base_symptom] = df[aff_col].fillna(0).astype(int)
                df.drop(columns=[aff_col], inplace=True)

    symptom_cols = [c for c in df.columns if c.startswith('symptom_')]
    if symptom_cols:
        df[symptom_cols] = df[symptom_cols].fillna(0).astype(int)

    target_col = 'readmitted_binary'
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found.")
        return

    exclude_cols = ['clinical_note', 'index', 'split', 'id', 'patient_id', 'encounter_id', 'patient_nbr', 'readmitted_binary']
    X = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors='ignore')
    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col]

    svd_model = None
    if mode in ["bert", "hybrid"] and ablation != "without_bert":
        bert_path = config.TRAIN_BERT_EMBEDDINGS_PATH
        if not bert_path.exists():
            logger.error(f"BERT embeddings not found at {bert_path}. Run extract_bert_embeddings.py first!")
            return
        
        logger.info(f"Loading Bio_ClinicalBERT dense matrix from: {bert_path}")
        bert_dense = np.load(str(bert_path))

        if ablation == "without_svd":
            logger.info("Applying Ablation: Passing Raw Uncompressed BERT Dimensions...")
            n_raw_dims = min(128, bert_dense.shape[1])
            for dim in range(n_raw_dims):
                X[f'bert_dim_{dim}'] = bert_dense[:, dim]
        else:
            logger.info(f"Fitting TruncatedSVD (768 -> {config.BERT_PCA_COMPONENTS} components)...")
            svd_model = TruncatedSVD(n_components=config.BERT_PCA_COMPONENTS, random_state=42)
            bert_svd = svd_model.fit_transform(bert_dense)

            for dim in range(config.BERT_PCA_COMPONENTS):
                X[f'bert_dim_{dim}'] = bert_svd[:, dim]

    string_cols = X.select_dtypes(include=['object']).columns
    if not string_cols.empty:
        X[string_cols] = X[string_cols].astype(str)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    sqrt_ratio = np.sqrt(neg_count / pos_count if pos_count > 0 else 1.0)
    logger.info(f"Class ratio: {neg_count/pos_count:.2f}:1 | Square-Root Weight: {sqrt_ratio:.2f}")

    numeric_cols, categorical_cols = get_tabular_columns(X_train)
    bert_cols = [c for c in X_train.columns if c.startswith('bert_dim_')]

    logger.info(f"Tabular Numeric: {len(numeric_cols)} | Categorical: {len(categorical_cols)} | Dense BERT: {len(bert_cols)}")
    
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, bert_cols=bert_cols)

    n_trials = 30
    logger.info(f"Starting Hyperparameter Tuning ({n_trials} trials, AUPRC scoring)...")

    def objective(trial):
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
            'classifier__scale_pos_weight': trial.suggest_float('classifier__scale_pos_weight', 1.5, sqrt_ratio * 1.5),
        }
        clf = xgb.XGBClassifier(random_state=42, eval_metric='logloss', tree_method='hist', n_jobs=1)
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
        pipe.set_params(**param)

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        score = cross_val_score(pipe, X_train, y_train, n_jobs=-1, cv=cv, scoring='average_precision')
        return score.mean()

    def callback(study, trial):
        logger.info(f"[Trial {trial.number + 1:02d}/{n_trials:02d}] AUPRC: {trial.value:.4f} | Best: {study.best_value:.4f}")

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    logger.success(f"Best parameters: {study.best_params}")

    final_clf = build_classifier()
    final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', final_clf)])
    final_pipeline.set_params(**study.best_params)

    logger.info("Fitting final pipeline on training set...")
    final_pipeline.fit(X_train, y_train)

    val_probs = final_pipeline.predict_proba(X_val)[:, 1]
    calibrated_threshold = calibrate_clinical_threshold(y_val, val_probs)

    validation_results = evaluate_pipeline(final_pipeline, X_val, y_val, threshold=calibrated_threshold)
    logger.info("--- Validation Set Results ---")
    logger.info(f"Validation AUROC        : {validation_results['auc']:.4f}")
    logger.info(f"Validation AUPRC (PR)   : {validation_results['auprc']:.4f}")
    logger.info(f"Validation Recall (Sens): {validation_results['recall']:.4f}")
    logger.info(f"Validation Precision    : {validation_results['precision']:.4f}")
    logger.info(f"Validation F2 Score     : {validation_results['f2']:.4f}")
    logger.info(f"Calibrated Threshold    : {calibrated_threshold:.4f}")

    if ablation != "none":
        output_payload_path = config.EXPERIMENTS_DIR / f"xgb_model_{mode}_ablation_{ablation}.joblib"
    else:
        output_payload_path = config.active_model_payload_path

    payload = {
        'model': final_pipeline,
        'threshold': calibrated_threshold,
        'svd_model': svd_model,
        'feature_cols': list(X.columns),
        'mode': mode,
        'ablation': ablation
    }

    output_payload_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, str(output_payload_path))
    logger.success(f"Model payload saved to: {output_payload_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model in any mode or ablation configuration.")
    parser.add_argument(
        "--mode", 
        choices=["baseline", "bert", "llm_enhanced", "hybrid"], 
        default="baseline", 
        help="Training mode"
    )
    parser.add_argument(
        "--ablation",
        choices=["none", "without_rag", "without_negation", "without_svd", "without_bert"],
        default="none",
        help="Ablation study configuration"
    )
    args = parser.parse_args()

    train_optimized_model(args.mode, ablation=args.ablation)