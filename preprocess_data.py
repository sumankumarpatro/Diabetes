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

    # 1. Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # 2. Basic Cleaning
    cols_to_drop = ['encounter_id', 'patient_nbr', 'payer_code']
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=existing_drops, inplace=True)
    logger.info(f"Dropped columns: {existing_drops}")

    # 3. Handle Missing Values
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

    # 4. Advanced Feature Engineering
    logger.info("Performing advanced feature engineering...")

    # A. Medication Count: Sum up all the binary medication columns
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glime_pioglitazone', 'metformin-rosiglitazone',
        'metformin-pioglitazone'
    ]
    # Filter to only those that actually exist in the dataframe
    existing_med_cols = [c for c in med_cols if c in df.columns]
    if existing_med_cols:
        df['total_medications_count'] = df[existing_med_cols].sum(axis=1)
        logger.info(f"Engineered 'total_medications_count' using {len(existing_med_cols)} columns.")

    # B. Age Binning: Create clinically relevant age groups
    if 'age' in df.columns:
        # The 'age' column contains ranges like '[70-80)'. 
        # We need to extract the lower bound to create numeric bins.
        try:
            # Extract the first number from the range string
            df['age_numeric'] = df['age'].str.extract(r'(\d+)').astype(float)
            bins = [0, 18, 35, 50, 65, 80, 120]
            labels = ['Pediatric', 'Young Adult', 'Adult', 'Middle-Acent', 'Senior', 'Elderly']
            df['age_group'] = pd.cut(df['age_numeric'], bins=bins, labels=labels, right=False)
            df.drop(columns=['age_numeric'], inplace=True)
            logger.info("Engineered 'age_group' via binning from range strings.")
        except Exception as e:
            logger.warning(f"failed to bin age: {e}")

    # C. Complexity Score: Proxy using number of diagnoses and procedures
    if 'number_diagnoses' in df.columns and 'num_procedures' in df.columns:
        df['clinical_complexity_score'] = df['number_diagnoses'] + df['num_procedures']
        logger.info("Engineered 'clinical_complexity_score'.")

    # 5. Encode Target Variable: 'readmitted'
    if 'readmitted' in df.columns:
        df['readmitted_binary'] = df['readmitted'].apply(lambda x: 1 if x == '>30' else 0)
    else:
        logger.error("Target column 'readmitted' not found in dataset.")
        return

    # 6. Encode Categorical Features
    le = LabelEncoder()
    # We want to encode all categorical columns except the original target 'readmitted'
    cols_to_encode = [col for col_name in categorical_cols if col_name != 'readmitted']
    
    for col in cols_to_encode:
        # Ensure we handle potential NaN values after encoding
        df[col] = le.fit_transform(df[col].astype(str))

    # 7. Split Data
    # Features: everything except the original target and the new binary target
    cols_to_exclude = ['readmitted', 'readmitted_binary']
    X = df.drop(columns=[c for c in cols_to_exclude if c in df.columns])
    y = df['readmitted_binary']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # 8. Save processed data
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
