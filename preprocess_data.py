import os
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from config import config

def preprocess_diabetes_data(data_path: Path, output_dir: Path) -> None:
    if not data_path.exists():
        logger.error(f"Input data not found: {data_path}")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading raw dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    df.replace('?', np.nan, inplace=True)
    
    if 'readmitted' in df.columns:
        df['readmitted_binary'] = df['readmitted'].apply(lambda x: 1 if str(x).strip() == '<30' else 0)
        df.drop(columns=['readmitted'], inplace=True)
        logger.info("Encoded 'readmitted' to 'readmitted_binary'.")
    else:
        logger.error("Target column 'readmitted' not found in dataset.")
        return

    logger.info("Performing feature engineering...")
    
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 
        'glime_pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    existing_med_cols = [c for c in med_cols if c in df.columns]
    if existing_med_cols:
        active_meds = df[existing_med_cols].apply(lambda col: col.astype(str).str.lower().isin(['steady', 'up', 'down'])).astype(int)
        df['total_medications_count'] = active_meds.sum(axis=1)
        logger.info(f"Engineered 'total_medications_count' using {len(existing_med_cols)} columns.")
        
    if 'age' in df.columns:
        try:
            df['age_numeric'] = df['age'].str.extract(r'\[?(\d+)-').astype(float)
            bins = [0, 18, 35, 50, 65, 80, 120]
            labels = ['Pediatric', 'Young Adult', 'Adult', 'Middle-Aged', 'Senior', 'Elderly']
            df['age_group'] = pd.cut(df['age_numeric'], bins=bins, labels=labels, right=False)
            df['age_group'] = df['age_group'].astype(str).fillna('Unknown')
            df.drop(columns=['age_numeric'], inplace=True)
            logger.info("Engineered 'age_group'.")
        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to bin age: {e}")
            
    if 'number_diagnoses' in df.columns and 'num_procedures' in df.columns:
        df['clinical_complexity_score'] = df['number_diagnoses'] + df['num_procedures']
        logger.info("Engineered 'clinical_complexity_score'.")

    if 'number_inpatient' in df.columns and 'number_diagnoses' in df.columns:
        df['heavy_utilizer_score'] = df['number_inpatient'] * df['number_diagnoses']
        logger.info("Engineered 'heavy_utilizer_score'.")

    if 'diabetesMed' in df.columns:
        df['diabetesMed_binary'] = df['diabetesMed'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
        df.drop(columns=['diabetesMed'], inplace=True)
        logger.info("Binarized 'diabetesMed'.")
        
    target_col = 'readmitted_binary'
    if 'patient_nbr' in df.columns:
        logger.info("Splitting data by patient_nbr (GroupShuffleSplit)...")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(df, df[target_col], groups=df['patient_nbr']))
        train_val_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
    else:
        logger.info("Splitting data with stratified train_test_split...")
        train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])

    cols_to_drop = [
        'encounter_id', 'patient_nbr', 'payer_code', 
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id'
    ]
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    train_val_df.drop(columns=existing_drops, inplace=True, errors='ignore')
    test_df.drop(columns=existing_drops, inplace=True, errors='ignore')
    
    train_output_file = config.TRAIN_DATA_PATH
    test_output_file = config.TEST_DATA_PATH
    
    train_val_df.to_csv(train_output_file, index=False)
    test_df.to_csv(test_output_file, index=False)
    
    logger.success(f"Train data saved: {train_output_file} ({len(train_val_df)} records)")
    logger.success(f"Test data saved: {test_output_file} ({len(test_df)} records)")

if __name__ == "__main__":
    preprocess_diabetes_data(config.RAW_DATA_PATH, config.PROCESSED_DIR)