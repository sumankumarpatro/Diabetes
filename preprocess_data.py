from loguru import logger
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from config import config
from pathlib import Path

def preprocess_diabetes_data(data_path: str, output_dir: str) -> None:
    """
    Preprocesses the diabetes dataset: cleaning, imputation, and encoding.
    """
    input_path = Path(data_path)
    output_path = Path(output_dir)

    if not input_path.exists():
        logger.error(f"Input data not found: {data_path}")
        return

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df.replace('?', np.nan, inplace=True)
    cols_to_drop = ['encounter_id', 'patient_nbr', 'payer_code']
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=existing_drops, inplace=True)
    logger.info(f"Dropped columns: {existing_drops}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        modes = df[col].mode()
        if not modes.empty:
            df[col] = df[col].fillna(modes[0])
        else:
            df[col] = df[col].fillna("Unknown")
    if 'readmitted' in df.columns:
        df['readmitted_binary'] = df['readmitted'].apply(lambda x: 1 if x == '>30' else 0)
    else:
        logger.error("Target column 'readmitted' not found in dataset.")
        return
    le = LabelEncoder()
    # We want to encode all categorical columns except the original target 'readmitted'
    cols_to_encode = [col for col in categorical_cols if col != 'readmitted']
    
    for col in cols_to_encode:
        df[col] = le.fit_transform(df[col].astype(str))
    # Features: everything except the original target and the new binary target
    cols_to_exclude = ['readmitted', 'readmitted_binary']
    X = df.drop(columns=[c for c in cols_to_exclude if c in df.columns])
    y = df['readmitted_binary']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    train_path = output_path / 'train.csv'
    val_path = output_path / 'val.csv'
    test_path = output_path / 'test.csv'

    # Re-attach target for the saved CSVs
    pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
    
    pd.concat([X_val, y_val], axis=1).to_csv(val_path, index=False)
    
    pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)

    logger.success(f"Preprocessing complete. Files saved in {output_dir}")
    logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    logger.info(f"Class distribution (binary): \n{y.value_counts(normalize=True)}")

if __name__ == "__main__":
    # Use paths from config for consistency
    # Use the RAW_DATA_PATH from config as the source of truth
    DATA_PATH = config.RAW_DATA_PATH
    OUTPUT_DIR = str(config.PROCESSED_DIR)
    
    preprocess_diabetes_data(str(DATA_PATH), OUTPUT_DIR)
