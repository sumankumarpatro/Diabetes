import asyncio
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger

from llm_interface import LLMInterface
from llm_providers import OllamaProvider
from rag_retriever import RAGRetriever
from clinical_agent import ClinicalOrchestratorAgent
from config import config

CONCURRENT_REQUESTS = 6

def standardize_categorical_fields(patient_features: dict) -> dict:
    glucose = str(patient_features.get('llm_glucose_status', '')).strip().lower()
    if any(k in glucose for k in ['high', 'hyper', 'increased']):
        patient_features['llm_glucose_status'] = 'Hyperglycemia'
    elif any(k in glucose for k in ['low', 'hypo', 'decreased']):
        patient_features['llm_glucose_status'] = 'Hypoglycemia'
    elif any(k in glucose for k in ['normal', 'normo', 'stable']):
        patient_features['llm_glucose_status'] = 'Normoglycemia'
    else:
        patient_features['llm_glucose_status'] = 'Unknown'
        
    age = str(patient_features.get('llm_age_group', '')).strip().lower()
    if 'adult' in age:
        patient_features['llm_age_group'] = 'Adult'
    elif any(k in age for k in ['pediatric', 'child', 'infant']):
        patient_features['llm_age_group'] = 'Pediatric'
    elif any(k in age for k in ['geriatric', 'elderly', 'senior']):
        patient_features['llm_age_group'] = 'Geriatric'
    else:
        patient_features['llm_age_group'] = 'Unknown'
        
    return patient_features

async def process_row_async(sem, agent, index, note):
    async with sem:
        try:
            report = await agent.orchestrate(note, skip_reflection=False, generate_recs=False)
            
            if report is None or not hasattr(report, 'features'):
                raise ValueError(f"Extractor returned invalid report structure for row {index}")

            features = report.features

            patient_features = {
                'llm_num_symptoms': len(features.symptoms) if hasattr(features, 'symptoms') else 0,
                'llm_num_medications': len(features.medications) if hasattr(features, 'medications') else 0,
                'llm_glucose_status': getattr(features, 'glucose_status', 'Unknown'),
                'llm_has_meds': 1 if (hasattr(features, 'medications') and len(features.medications) > 0) else 0,
                'llm_has_symptoms': 1 if (hasattr(features, 'symptoms') and len(features.symptoms) > 0) else 0,
                'llm_age_group': getattr(features, 'age_group', 'Unknown')
            }

            if hasattr(features, 'symptoms') and features.symptoms:
                for symptom_obj in features.symptoms:
                    s_name = symptom_obj.name.lower().replace(" ", "_").strip()
                    is_negated = getattr(symptom_obj, 'is_negated', False)
                    
                    affirmed_key = f"symptom_{s_name}_affirmed"
                    negated_key = f"symptom_{s_name}_negated"
                    
                    patient_features[affirmed_key] = 0 if is_negated else 1
                    patient_features[negated_key] = 1 if is_negated else 0

            patient_features = standardize_categorical_fields(patient_features)
            return index, patient_features

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error processing row {index}: {e}")
            return index, {
                'llm_num_symptoms': 0,
                'llm_num_medications': 0,
                'llm_glucose_status': 'Error',
                'llm_has_meds': 0,
                'llm_has_symptoms': 0,
                'llm_age_group': 'Error'
            }

async def extract_features_from_dataset_async(sem, input_path: Path, output_path: Path):
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    
    checkpoint_path = output_path.with_suffix('.jsonl')
    processed_indices = set()
    cached_encounters = {}

    if checkpoint_path.exists():
        logger.info(f"Found existing checkpoint file: {checkpoint_path}. Resuming...")
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as cp_file:
                for line in cp_file:
                    if line.strip():
                        record = json.loads(line)
                        idx = record.pop('index')
                        processed_indices.add(idx)
                        cached_encounters[idx] = record
            logger.info(f"Loaded {len(processed_indices)} completed records from checkpoint.")
        except (json.JSONDecodeError, KeyError, IOError) as e:
            logger.warning(f"Failed to parse checkpoint ({e}). Starting fresh.")
            processed_indices = set()
            cached_encounters = {}

    provider = OllamaProvider()
    llm = LLMInterface(provider)
    retriever = RAGRetriever()
    retriever.load()
    agent = ClinicalOrchestratorAgent(retriever, llm)

    tasks = [
        process_row_async(sem, agent, idx, row['clinical_note'])
        for idx, row in df.iterrows()
        if idx not in processed_indices
    ]

    total_tasks = len(tasks)
    if total_tasks == 0:
        logger.info(f"All records in {input_path.name} are already complete in checkpoint.")
    else:
        logger.info(f"Starting async extraction for {total_tasks} remaining rows (concurrency: {CONCURRENT_REQUESTS}).")
        
        with open(checkpoint_path, 'a', encoding='utf-8') as cp_file:
            for future in tqdm(asyncio.as_completed(tasks), total=total_tasks, desc=f"Extracting {input_path.name}"):
                index, patient_features = await future
                cached_encounters[index] = patient_features
                
                checkpoint_record = {'index': index, **patient_features}
                cp_file.write(json.dumps(checkpoint_record) + '\n')
                cp_file.flush()

    logger.info("Compiling final ordered dataset structure...")
    sorted_indices = sorted(cached_encounters.keys())
    ordered_features = [cached_encounters[idx] for idx in sorted_indices]
    features_df = pd.DataFrame(ordered_features)

    symptom_cols = [c for c in features_df.columns if c.startswith('symptom_')]
    if symptom_cols:
        features_df[symptom_cols] = features_df[symptom_cols].fillna(0).astype(int)
    
    records_df = pd.concat([df.loc[sorted_indices].reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records_df.to_csv(output_path, index=False)

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info(f"Cleaned up checkpoint cache: {checkpoint_path}")

    await provider.close()
    retriever.close()
    logger.success(f"Extracted features saved to: {output_path}")

def harmonize_train_test_columns(train_path: Path, test_path: Path):
    logger.info("Harmonizing feature columns between Train and Test splits...")
    if not train_path.exists() or not test_path.exists():
        logger.error("Cannot harmonize: One or both feature files do not exist.")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    all_symptom_cols = sorted(list(set(
        [c for c in train_df.columns if c.startswith('symptom_')] +
        [c for c in test_df.columns if c.startswith('symptom_')]
    )))

    for col in all_symptom_cols:
        if col not in train_df.columns:
            train_df[col] = 0
        if col not in test_df.columns:
            test_df[col] = 0

    train_df[all_symptom_cols] = train_df[all_symptom_cols].fillna(0).astype(int)
    test_df[all_symptom_cols] = test_df[all_symptom_cols].fillna(0).astype(int)

    base_cols = [c for c in train_df.columns if not c.startswith('symptom_')]
    aligned_column_order = base_cols + all_symptom_cols

    train_df = train_df[aligned_column_order]
    test_df = test_df[aligned_column_order]

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.success(f"Successfully harmonized {len(all_symptom_cols)} symptom feature columns across Train and Test datasets.")

async def main():
    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    train_input = config.TRAIN_WITH_NOTES_PATH
    test_input = config.TEST_WITH_NOTES_PATH
    output_train = config.TRAIN_WITH_EXTRACTED_FEATURES_PATH
    output_test = config.TEST_WITH_EXTRACTED_FEATURES_PATH

    await extract_features_from_dataset_async(sem, train_input, output_train)
    await extract_features_from_dataset_async(sem, test_input, output_test)
    harmonize_train_test_columns(output_train, output_test)

if __name__ == "__main__":
    asyncio.run(main())