import argparse
import os
import re
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
from pathlib import Path

import joblib
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from config import config

plt.rcParams.update({'figure.dpi': 300})

def clean_feature_names(feature_names):
    clean_names = []
    for name in feature_names:
        name = re.sub(r'^(num__|cat__|remainder__)', '', name)
        name = name.replace('symptom_', 'Symptom: ')
        name = name.replace('_affirmed', ' (Yes)')
        name = name.replace('_negated', ' (No)')
        name = name.replace('llm_glucose_status', 'LLM Glucose Status')
        name = name.replace('total_medications_count', 'Total Meds')
        name = name.replace('time_in_hospital', 'Days in Hospital')
        name = name.replace('number_inpatient', 'Prior Inpatient Visits')
        clean_names.append(name)
    return clean_names

def generate_shap_plots(mode: str):
    config.MODE = mode
    payload_path = config.active_model_payload_path
    
    if not payload_path.exists():
        logger.error(f"Model payload not found: {payload_path}. Train the model first!")
        return
        
    logger.info(f"Loading [{mode.upper()}] payload from: {payload_path}")
    payload = joblib.load(payload_path)
    pipeline = payload['model']
    expected_cols = payload['feature_cols']
    svd_model = payload.get('svd_model', None)
    
    test_path = config.active_test_path
    logger.info(f"Loading independent test data from: {test_path}")
    df_test = pd.read_csv(test_path)
    
    symptom_cols = [c for c in df_test.columns if c.startswith('symptom_')]
    if symptom_cols:
        df_test[symptom_cols] = df_test[symptom_cols].fillna(0).astype(int)
        
    if mode in ["bert", "hybrid"] and svd_model:
        bert_test_path = config.TEST_BERT_EMBEDDINGS_PATH
        test_bert_dense = np.load(str(bert_test_path))
        test_bert_svd = svd_model.transform(test_bert_dense)
        bert_cols = {f'bert_dim_{dim}': test_bert_svd[:, dim] for dim in range(config.BERT_PCA_COMPONENTS)}
        df_test = pd.concat([df_test, pd.DataFrame(bert_cols, index=df_test.index)], axis=1)

    missing_cols = [c for c in expected_cols if c not in df_test.columns]
    if missing_cols:
        df_missing = pd.DataFrame(0, index=df_test.index, columns=missing_cols)
        df_test = pd.concat([df_test, df_missing], axis=1)
    
    X_test = df_test[expected_cols].copy()

    string_cols = X_test.select_dtypes(include=['object']).columns
    if not string_cols.empty:
        X_test[string_cols] = X_test[string_cols].astype(str)

    logger.info("Applying preprocessing transformations...")
    preprocessor = pipeline.named_steps['preprocessor']
    X_test_transformed = preprocessor.transform(X_test)
    
    try:
        raw_feature_names = preprocessor.get_feature_names_out()
    except (AttributeError, TypeError):
        raw_feature_names = X_test.columns
        
    feature_names = clean_feature_names(raw_feature_names)
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)
    
    sample_size = min(2500, len(X_test_transformed_df))
    X_sample = X_test_transformed_df.sample(n=sample_size, random_state=42)
    
    logger.info(f"Calculating SHAP values for {sample_size} test samples...")
    xgb_model = pipeline.named_steps['classifier']
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_sample)

    out_dir = config.PROJECT_ROOT / "plots" / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating SHAP Bar Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
    plt.title(f"Top 15 Predictors of 30-Day Readmission ({mode.upper()})", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_bar_{mode}.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Generating SHAP Summary Dot Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.title(f"SHAP Value Impact on Prediction ({mode.upper()})", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_summary_{mode}.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Generating SHAP Waterfall Plot...")
    probs = xgb_model.predict_proba(X_sample)[:, 1]
    high_risk_idx = np.argmax(probs)
    
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[high_risk_idx], max_display=10, show=False)
    plt.title(f"Individual Patient Risk Explanation (Probability: {probs[high_risk_idx]:.2f})", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_waterfall_{mode}.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.success(f"All SHAP plots saved successfully to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SHAP plots for trained models.")
    parser.add_argument(
        "--mode", 
        choices=["baseline", "bert", "llm_enhanced", "hybrid"], 
        default="hybrid",
        help="Target model mode for SHAP analysis"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate SHAP plots for all 4 model profiles (baseline, bert, llm_enhanced, hybrid)"
    )
    args = parser.parse_args()
    import warnings
    warnings.filterwarnings("ignore")

    if args.all:
        for m in ["baseline", "bert", "llm_enhanced", "hybrid"]:
            logger.info(f"Generating SHAP plots for profile: {m}")
            generate_shap_plots(m)
    else:
        generate_shap_plots(args.mode)