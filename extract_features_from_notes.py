import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger
from pathlib import Path
from llm_interface import LLMInterface
from llm_providers import OllamaProvider
from rag_retriever import RAGRetriever
from clinical_agent import ClinicalOrchestratorAgent
from config import config

def extract_features_from_dataset(input_path: str, output_path: str):
    """
    Processes a dataset with clinical notes, uses the ClinicalOrchestratorAgent 
    to extract structured features, and saves the augmented dataset.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Loading dataset: {input_path}")
    df = pd.read_csv(input_path)

    # Initialize Agent Dependencies
    # Note: In a real scenario, you'd use the same setup as your main app
    provider = OllamaProvider() 
    llm = LLMInterface(provider)
    retriever = RAGRetriever() # Assumes RAG index is already set up
    retriever.load()
    agent = ClinicalOrchestratorAgent(retriever, llm)

    new_features = []

    logger.info(f"Starting feature extraction for {len(df)} rows...")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        note = row['clinical_note']
        # We don't know the true readmission prediction here, 
        # so we use a placeholder or a dummy value. 
        # The extraction logic in clinical_agent.py doesn't strictly depend on it for parsing.
        prior_readmission_indicator = False 

        try:
            report = agent.orchestrate(note, prior_readmission_indicator)
            
            if report:
                features = report.features
                # Flatten the extracted features into a dictionary
                feature_dict = {
                    'num_symptoms': len(features.symptoms),
                    'num_medications': len(features.meds), # Note: check field name in ClinicalFeatures
                    'glucose_status_encoded': 0 if features.glucose_status == 'Normal' else (1 if features.glucose_status == 'Hyperglycemia' else 2),
                    'has_symptoms': 1 if len(features.symptoms) > 0 else 0,
                    'has_medications': 1 if len(features.medications) > 0 else 0,
                    'extracted_age_group': features.age_group
                }
                # Wait, I need to check the actual field names in ClinicalFeatures from clinical_agent.py
                # Looking back at clinical_agent.py: 
                # age_group, symptoms, medications, hospital_stay_days, glucose_status
                
                # Let's re-map correctly
                feature_dict = {
                    'llm_num_symptoms': len(features.symptoms),
                    'llm_num_medications': len(features.medications),
                    'llm_glucose_status': features.glucose_status,
                    'llm_has_meds': 1 if len(features.medications) > 0 else 0,
                    'llm_has_symptoms': 1 if len(features.symptoms) > 0 else 0,
                }
                new_features.append(feature_dict)
            else:
                # If extraction fails, fill with NaNs/Defaults
                new_features.append({
                    'llm_num_symptoms': np.nan,
                    'llm_num_medications': np.nan,
                    'llm_glucose_status': 'Unknown',
                    'llm_has_meds': 0,
                    'llm_has_symptoms': 0,
                })
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error processing row {index}: {e}")
            new_features.append({
                'llm_num_symptoms': np.nan,
                'llm_num_medications': np.nan,
                'llm_glucose_status': 'Unknown',
                'llm_has_meds': 0,
                'llm_has_symptoms': 0,
            })

    # Create DataFrame from new features
    features_df = pd.DataFrame(new_features)
    
    # Concatenate with original DataFrame
    final_df = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

    # Save the result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logger.success(f"Feature extraction complete. Saved to: {output_path}")

if __name__ == "__main__":
    # Define paths
    TRAIN_WITH_NOTES = "/Users/unasumankumarpatro/Documents/Diabetes/processed_data/train_with_notes.csv"
    TEST_WITH_NOTES = "/Users/unasumankumarpatro/Documents/Diabetes/processed_data/test_with_notes.csv"
    
    OUTPUT_TRAIN = "/Users/unasumankumarpatro/Documents/Diabetes/processed_data/train_with_extracted_features.csv"
    OUTPUT_TEST = "/Users/unasumankumarpatro/Documents/Diabetes/processed_data/test_with_extracted_features.csv"

    extract_features_from_dataset(TRAIN_WITH_NOTES, OUTPUT_TRAIN)
    extract_features_from_dataset(TEST_WITH_NOTES, OUTPUT_TEST)
