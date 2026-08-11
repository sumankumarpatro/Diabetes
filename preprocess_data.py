from loguru import logger
import pandas as pd
import numpy as np
import os
from config import config
from pathlib import Path
from sklearn.model_selection import train_test_split

def preprocess_diabetes_data(data_path: str, output_dir: str) -> None:
    """
    Preprocesses the diabetes dataset: cleaning and feature engineering.
    Note: Imputation and Encoding are handled in the training pipeline to prevent leakage.
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

    # 3. Advanced Feature Engineering
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
            labels = ['Pediatric', 'Young Adult', 'Adult', 'Middle-Aged', 'Senior', 'Elderly']
            df['age_group'] = pd.cut(df['age_numeric'], bins=bins, labels=labels, right=False)
            df['age_group'] = df['age_group'].astype('category')
            df['age_group'] = df['age_group'].cat.add_categories(['Unknown']).fillna('Unknown')
            df.drop(columns=['age_numeric'], inplace=True)
            logger.info("Engineered 'age_group' via binning from range strings.")
        except Exception as e:
            logger.warning(f"failed to bin age: {e}")

    # C. Complexity Score: Proxy using number of diagnoses and procedures
    if 'number_diagnoses' in df.columns and 'num_procedures' in df.columns:
        df['clinical_complexity_score'] = df['number_diagnoses'] + df['num_procedures']
        logger.info("Engineered 'clinical_complexity_score'.")

    # 4. Encode Target Variable: 'readmitted'
    if 'readmitted' in df.columns:
        df['readmitted_binary'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
        # IMPORTANT: Drop the original 'readmitted' column to prevent target leakage
        df.drop(columns=['readmitted'], inplace=True)
        logger.info("Encoded 'readmitted' to 'readmitted_binary' and dropped original column to prevent leakage.")
    else:
        logger.error("Target column 'readmitted' not found in dataset.")
        return

    # 5. Split data into train and test sets to ensure isolation
    logger.info("Splitting data into train and test sets...")
    target_col = 'readmitted_binary'
    train_val_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[target_col] if target_col in df.columns else None
    )

    # Save test set
    test_output_file = output_path / 'test.csv'
    test_df.to_csv(test_output_file, index=False)
    logger.success(f"Test data saved in {test_output_file}")

    # Save train/val set
    train_output_file = output_path / 'train.csv'
    train_val_df.to_csv(train_output_file, index=False)
    logger.success(f"Train/Val data saved in {train_output_file}")

    # Show class distribution for the training set
    target_col = 'readmitted_binary'
    if target_col in train_val_df.columns:
        logger.info(f"Train/Val class distribution:\n{train_val_df[target_col].value_counts()}")
    else:
        logger.warning(f"Target column '{target_col}' not found for class distribution logging.")

    logger.info(f"Total records: {len(df)} | Train/Val: {len(train_val_df)} | Test: {len(test_df)}")

if __name__ == "__main__":
    # Use paths from config for consistency
    DATA_PATH = config.RAW_DATA_PATH
    OUTPUT_DIR = str(config.PROCESSED_DIR)
    
    preprocess_diabetes_data(str(DATA_PATH), OUTPUT_DIR)
