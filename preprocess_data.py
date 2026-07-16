import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def preprocess_diabetes_data(data_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df.replace('?', np.nan, inplace=True)
    # Drop columns with too many missing values or irrelevant info for the baseline
    # encounter_id and patient_nbr are identifiers
    cols_to_drop = ['encounter_id', 'patient_nbr', 'payer_code']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    # For simplicity in this baseline, we'll fill numeric NaNs with median and categorical with mode
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    # The target is '..'
    # Let's make it binary: 1 if '>30', 0 otherwise.
    df['readmitted_binary'] = df['readmitted'].apply(lambda x: 1 if x == '>30' else 0)
    le = LabelEncoder()
    # We want to encode all categorical columns except the original target 'readmitted'
    cols_to_encode = [col for col in categorical_cols if col != 'readmitted']
    for col in cols_to_encode:
        df[col] = le.fit_transform(df[col].astype(str))
    # Features: everything except the original target and the new binary target
    X = df.drop(columns=['readmitted', 'readmitted_binary'])
    y = df['readmitted_binary']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    # Re-attach target for the saved CSVs
    pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
    pd.concat([X_val, y_val], axis=1).to_csv(val_path, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)

    print(f"Preprocessing complete. Files saved in {output_dir}")
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    print(f"Class distribution (binary): \n{y.value_counts(normalize=True)}")

def flags_median(df, cols):
    # Helper to avoid error in my logic above
    return df[cols].median()

if __name__ == "__main__":
    DATA_PATH = "/Users/unasumankumarpatro/Documents/Diabetes/Diabetes paper/diabetic_data.csv"
    OUTPUT_DIR = "/Users/unasumankumarpatro/Documents/Diabetes/processed_data"
    preprocess_diabetes_data(DATA_PATH, OUTPUT_DIR)
