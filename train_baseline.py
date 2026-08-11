import inspect
from pathlib import Path
import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from config import config


def build_fit_params(pipeline, X_val, y_val, early_stopping_rounds: int = 20, verbose: bool = False) -> dict:
    """Build XGBoost fit params compatible with the installed version."""
    fit_params = {
        'classifier__verbose': verbose,
    }

    try:
        fit_signature = inspect.signature(xgb.XGBClassifier.fit)
    except (TypeError, ValueError):
        fit_signature = None

    if fit_signature and 'early_stopping_rounds' in fit_signature.parameters:
        try:
            # Transform the validation set through the pipeline preprocessor before passing it
            # to the classifier so XGBoost receives numeric data only.
            preprocessor = pipeline.named_steps['preprocessor']
            X_val_transformed = preprocessor.transform(X_val)
            fit_params['classifier__eval_set'] = [(X_val_transformed, y_val)]
            fit_params['classifier__early_stopping_rounds'] = early_stopping_rounds
        except (AttributeError, KeyError, ValueError) as exc:
            logger.warning(
                'Unable to build XGBoost eval_set for early stopping: %s. Falling back to no eval_set.',
                exc,
            )

    return fit_params

def objective(trial, pipeline, X_train, y_train):
    """
    Optuna objective function for hyperparameter tuning using a pipeline.
    """
    param = {
        'classifier__n_estimators': trial.suggest_int('classifier__n_estimators', 50, 300),
        'classifier__max_depth': trial.suggest_int('classifier__max_depth', 3, 10),
        'classifier__learning_rate': trial.suggest_float('classifier__learning_rate', 0.01, 0.3, log=True),
        'classifier__subsample': trial.suggest_float('classifier__subsample', 0.5, 1.0),
        'classifier__colsample_bytree': trial.suggest_float('classifier__colsample_bytree', 0.5, 1.0),
        'classifier__min_child_weight': trial.suggest_int('classifier__min_child_weight', 1, 10),
        'classifier__gamma': trial.suggest_float('classifier__gamma', 0, 5),
    }

    pipeline.set_params(**param)
    score = cross_val_score(pipeline, X_train, y_train, n_jobs=-1, cv=3, scoring='roc_auc')
    return score.mean()

def train_optimized_model(data_path: str, model_output_path: str) -> None:
    """
    Trains an optimized XGBoost model using a standard Pipeline.
    """
    data_path = Path(data_path)
    model_output_path = Path(model_output_path)

    if not data_path.exists():
        logger.error(f"Error: Data file not found. Ensure {data_path} exists.")
        return

    logger.info(f"Loading cleaned data from: {data_path}")
    df = pd.read_csv(data_path)

    target_col = 'readmitted_binary'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # We'll use a validation set for the final evaluation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Show class distribution after split
    logger.info(f"Training set class distribution:\n{y_train.value_counts()}")
    logger.info(f"Validation set class distribution:\n{y_val.value_counts()}")

    # Identify feature types
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Numeric features: {len(numeric_cols)}")
    logger.info(f"Categorical features: {len(categorical_cols)}")
    # We use Imbalanced-learn Pipeline to ensure SMOTE is only applied to training folds during CV
    # and only to the training set during final fit.
    
    # For simplicity in this refactor, we'll use a single pipeline with SMOTENC.
    # Note: SMOTENC requires categorical feature indices in the transformed array.
    cat_indices = list(range(len(numeric_cols), len(numeric_cols) + len(categorical_cols)))
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # The full pipeline
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTENC(categorical_features=cat_indices, random_state=42)),
        ('classifier', xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        ))
    ])
    logger.info("Starting Hyperparameter Tuning with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, pipeline, X_train, y_train), n_trials=30)

    logger.success(f"Best hyperparameters found: {study.best_params}")
    logger.info("Training final model with best parameters...")
    
    best_params = study.best_params
    pipeline.set_params(**best_params)

    # Pre-fit the preprocessor so the validation set can be transformed for XGBoost eval_set.
    if 'preprocessor' in pipeline.named_steps:
        pipeline.named_steps['preprocessor'].fit(X_train)
    
    # Use the validation set during fitting to avoid overfitting and to select the best threshold.
    fit_params = build_fit_params(pipeline, X_val, y_val, early_stopping_rounds=20, verbose=False)
    pipeline.fit(X_train, y_train, **fit_params)
    logger.info("Evaluating on validation set...")
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_prob >= 0.5).astype(int)
    val_f1 = f1_score(y_val, y_val_pred)
    logger.info(f"Validation F1 at 0.5: {val_f1:.4f}")

    optimized_threshold = self_optimize_threshold(y_val, y_val_prob)
    logger.info(f"Optimized threshold from validation: {optimized_threshold:.2f}")
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= optimized_threshold).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.info("\n--- Optimized Pipeline Evaluation ---")
    logger.info(f"AUC-ROC:   {auc:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    payload = {
        'model': pipeline,
        'threshold': optimized_threshold
    }
    joblib.dump(payload, str(model_output_path))
    logger.success(f"Optimized pipeline payload saved to: {model_output_path}")


def self_optimize_threshold(y_true, y_prob):
    """Selects a threshold based on the best F1 score on validation probabilities."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        preds = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    return best_threshold


if __name__ == "__main__":
    # Use the path from config to ensure consistency with test_model.py
    MODEL_SAVE_PATH = str(config.MODEL_PAYLOAD_PATH)
    # We need to point to the new train.csv created by preprocess_data.py
    TRAIN_DATA_PATH = str(config.PROCESSED_DIR / 'train.csv')

    train_optimized_model(TRAIN_DATA_PATH, MODEL_SAVE_PATH)
