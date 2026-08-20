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


def get_feature_columns(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
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


def build_smote_cat_indices(numeric_cols, categorical_cols):
    return list(range(len(numeric_cols), len(numeric_cols) + len(categorical_cols)))


def build_classifier(**kwargs):
    return xgb.XGBClassifier(
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss',
        **kwargs,
    )


def build_pipeline(preprocessor, categorical_feature_indices, classifier=None):
    if classifier is None:
        classifier = build_classifier()

    return ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTENC(categorical_features=categorical_feature_indices, random_state=42)),
        ('classifier', classifier)
    ])


def objective(trial, pipeline, X_train, y_train):
    """Optuna objective function for hyperparameter tuning using a pipeline."""
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


def split_train_val_test(X, y, test_size=0.2, val_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def log_split_summary(y_train, y_val):
    logger.info(f"Training set class distribution:\n{y_train.value_counts()}")
    logger.info(f"Validation set class distribution:\n{y_val.value_counts()}")


def log_feature_summary(numeric_cols, categorical_cols):
    logger.info(f"Numeric features: {len(numeric_cols)}")
    logger.info(f"Categorical features: {len(categorical_cols)}")


def tune_hyperparameters(pipeline, X_train, y_train, n_trials=30):
    logger.info("Starting Hyperparameter Tuning with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, pipeline, X_train, y_train), n_trials=n_trials)
    logger.success(f"Best hyperparameters found: {study.best_params}")
    return study


def train_final_pipeline(pipeline, X_train, y_train, X_val, y_val, early_stopping_rounds=20, verbose=False):
    if 'preprocessor' in pipeline.named_steps:
        pipeline.named_steps['preprocessor'].fit(X_train)

    fit_params = build_fit_params(pipeline, X_val, y_val, early_stopping_rounds=early_stopping_rounds, verbose=verbose)
    pipeline.fit(X_train, y_train, **fit_params)
    return pipeline


def evaluate_pipeline(pipeline, X, y, threshold=0.5):
    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        'auc': roc_auc_score(y, y_prob),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'y_prob': y_prob,
        'y_pred': y_pred,
    }


def train_optimized_model(data_path: str, model_output_path: str) -> None:
    """Trains an optimized XGBoost model using a modular pipeline."""
    data_path = Path(data_path)
    model_output_path = Path(model_output_path)

    if not data_path.exists():
        logger.error(f"Error: Data file not found. Ensure {data_path} exists.")
        return

    logger.info(f"Loading cleaned data from: {data_path}")
    df = pd.read_csv(data_path)

    if 'readmitted_binary' not in df.columns:
        logger.error(f"Target column 'readmitted_binary' not found in {data_path}")
        return

    target_col = 'readmitted_binary'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split only into Train and Validation to prevent leakage from the independent Test set
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log_split_summary(y_train, y_val)

    numeric_cols, categorical_cols = get_feature_columns(X_train)
    log_feature_summary(numeric_cols, categorical_cols)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    cat_indices = build_smote_cat_indices(numeric_cols, categorical_cols)
    pipeline = build_pipeline(preprocessor, cat_indices)

    study = tune_hyperparameters(pipeline, X_train, y_train)
    pipeline.set_params(**study.best_params)

    logger.info("Training final model with best parameters...")
    pipeline = train_final_pipeline(pipeline, X_train, y_train, X_val, y_val)

    logger.info("Evaluating on validation set...")
    validation_results = evaluate_pipeline(pipeline, X_val, y_val)
    
    logger.info("\n--- Validation Set Evaluation ---")
    logger.info(f"Validation AUC:    {validation_results['auc']:.4f}")
    logger.info(f"Validation Precision: {validation_results['precision']:.4f}")
    logger.info(f"Validation Recall:    {validation_results['recall']:.4f}")
    logger.info(f"Validation F1:     {validation_results['f1']:.4f}")
    logger.info("\nValidation Classification Report:")
    logger.info(classification_report(y_val, validation_results['y_pred']))

    optimized_threshold = self_optimize_threshold(y_val, validation_results['y_prob'])
    logger.info(f"Optimized threshold from validation: {optimized_threshold:.2f}")

    # Save the trained model payload (pipeline + threshold)
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
    MODEL_SAVE_PATH = str(config.MODEL_PAYLOAD_PATH_BASELINE)
    TRAIN_DATA_PATH = str(config.TRAIN_DATA_PATH)

    train_optimized_model(TRAIN_DATA_PATH, MODEL_SAVE_PATH)
