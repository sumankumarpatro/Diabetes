from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC
import xgboost as xgb
import optuna
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score, f1_score
from loguru import logger
from config import config
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
    
    # Calculate ratio for scale_pos_weight
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    param['classifier__scale_pos_weight'] = ratio
    
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
    # Note: SMOTENC requires categorical features indices.
    cat_indices = [X_train.columns.get_loc(col) for col in categorical_cols]
    
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
        ('classifier', xgb.XGBClassifier(random_state=42, enable_categorical=True))
    ])
    logger.info("Starting Hyperparameter Tuning with Optuna...")
    study = optuna.create_study(direction='maximize')
    
    # Reverting to full number of trials for proper optimization
    study.optimize(lambda trial: objective(trial, pipeline, X_train, y_train), n_trials=20)

    logger.success(f"Best hyperparameters found: {study.best_params}")
    logger.info("Training final model with best parameters...")
    
    best_params = study.best_params
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    best_params['classifier__scale_pos_weight'] = ratio
    
    pipeline.set_params(**best_params)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

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
        'threshold': 0.5  # Default threshold, can be optimized in the future
    }
    joblib.dump(payload, str(model_output_path))
    logger.success(f"Optimized pipeline payload saved to: {model_output_path}")

if __name__ == "__main__":
    # Use the path from config to ensure consistency with test_model.py
    MODEL_SAVE_PATH = str(config.MODEL_PAYLOAD_PATH)
    # We need to point to the new train.csv created by preprocess_data.py
    TRAIN_DATA_PATH = str(config.PROCESSED_DIR / 'train.csv')

    train_optimized_model(TRAIN_DATA_PATH, MODEL_SAVE_PATH)
