import asyncio
import json
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

# --- CONFIGURATION ---
CONCURRENT_REQUESTS = 6

async def process_row_async(sem, agent, index, note, dummy_prediction):
    """
    Processes a single row natively using pure async/await mechanics.
    Leverages a semaphore to control concurrent traffic to the local Ollama socket.
    """
    async with sem:
        try:
            # Invoking the natively async orchestrate pipeline
            report = await agent.orchestrate(note, dummy_prediction)
            
            if report and hasattr(report, 'features') and report.features:
                features = report.features
                return index, {
                    'llm_num_symptoms': len(features.symptoms) if hasattr(features, 'symptoms') else 0,
                    'llm_num_medications': len(features.medications) if hasattr(features, 'medications') else 0,
                    'llm_glucose_status': getattr(features, 'glucose_status', 'Unknown'),
                    'llm_has_meds': 1 if (hasattr(features, 'medications') and len(features.medications) > 0) else 0,
                    'llm_has_symptoms': 1 if (hasattr(features, 'symptoms') and len(features.symptoms) > 0) else 0,
                    'llm_age_group': getattr(features, 'age_group', 'Unknown')
                }
            else:
                return index, {
                    'llm_num_symptoms': np.nan,
                    'llm_num_medications': np.nan,
                    'llm_glucose_status': 'Unknown',
                    'llm_has_meds': 0,
                    'llm_has_symptoms': 0,
                    'llm_age_group': 'Unknown'
                }
        except Exception as e:
            logger.error(f"Error processing row {index}: {e}")
            return index, {
                'llm_num_symptoms': np.nan,
                'llm_num_medications': np.nan,
                'llm_glucose_status': 'Error',
                'llm_has_meds': 0,
                'llm_has_symptoms': 0,
                'llm_age_group': 'Error'
            }

async def extract_features_from_dataset_async(sem, input_path: Path, output_path: Path):
    """
    Processes a dataset concurrently using native async loops with streaming
    JSONL checkpoint recovery to safeguard massive 100k data execution runs.
    """
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Loading dataset: {input_path}")
    # Inside extract_features_from_dataset_async, right after reading the CSV:
    df = pd.read_csv(input_path)

    # --- SORT BY CATEGORY TO OPTIMIZE HARDWARE KV CACHE ---
    if 'readmitted_binary' in df.columns:
        df = df.sort_values(by='readmitted_binary').reset_index(drop=True)
    
    # Establish a streaming checkpoint file layout next to the target output destination
    checkpoint_path = output_path.with_suffix('.jsonl')
    processed_indices = set()
    checkpoint_results = {}

    # --- PROGRESS RESUMPTION MECHANISM ---
    if checkpoint_path.exists():
        logger.info(f"Found runtime checkpoint mapping file: {checkpoint_path}. Parsing history...")
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as cp_file:
                for line in cp_file:
                    if line.strip():
                        record = json.loads(line)
                        idx = record.pop('index')
                        processed_indices.add(idx)
                        checkpoint_results[idx] = record
            logger.info(f"Successfully loaded checkpoints. Skipping {len(processed_indices)} already processed entries.")
        except Exception as e:
            logger.warning(f"Could not read historical checkpoint track ({e}). Processing from scratch.")
            processed_indices = set()
            checkpoint_results = {}

    # Initialize Agent Async-Ready Dependencies
    provider = OllamaProvider()
    llm = LLMInterface(provider)
    retriever = RAGRetriever()
    retriever.load()
    agent = ClinicalOrchestratorAgent(retriever, llm)

    # Compile the filtered async background tasks list
    tasks = []
    for index, row in df.iterrows():
        if index in processed_indices:
            continue
        note = row['clinical_note']
        dummy_prediction = False
        tasks.append(process_row_async(sem, agent, index, note, dummy_prediction))

    total_tasks = len(tasks)
    if total_tasks == 0:
        logger.info("All records in this targeted dataset are already flagged as complete via checkpoint history.")
    else:
        logger.info(f"Starting native async feature extraction for {total_tasks} remaining rows (Concurrency: {CONCURRENT_REQUESTS}).")
        
        # Open the checkpoint file tracker in Append mode to instantly log completions to storage drive
        with open(checkpoint_path, 'a', encoding='utf-8') as cp_file:
            for future in tqdm(asyncio.as_completed(tasks), total=total_tasks, desc="Processing Rows"):
                index, feature_dict = await future
                checkpoint_results[index] = feature_dict
                
                # Instantly write the dictionary down to storage disk to secure execution data state
                checkpoint_record = {'index': index, **feature_dict}
                cp_file.write(json.dumps(checkpoint_record) + '\n')
                cp_file.flush()

    # --- DATA INTEGRITY CONSOLIDATION STAGE ---
    logger.info("Compiling final ordered dataset structure allocations...")
    
    # Reassemble un-ordered completed task mappings to line up with the original base index arrangement
    sorted_indices = sorted(checkpoint_results.keys())
    ordered_features = [checkpoint_results[idx] for idx in sorted_indices]
    features_df = pd.DataFrame(ordered_features)
    
    # Flatten indexes prior to horizontal concatenation axes execution
    final_df = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    
    # Commit cleanly compiled CSV to permanent disk storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    # Clean up tracking assets and close provider sockets
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Cleaned up operational checkpoint cache files.")
        
    await provider.close()
    retriever.close()
    logger.success(f"Feature extraction successfully completed. Target file compiled: {output_path}")

async def main():
    # Instantiate the asyncio Semaphore context block inside the active event loops
    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    train_input = config.OUTPUT_DATA_PATH
    test_input = config.TEST_DATA_PATH
    
    output_train = config.PROCESSED_DIR / "train_with_extracted_features.csv"
    output_test = config.PROCESSED_DIR / "test_with_extracted_features.csv"
    
    # Process dataset groups sequentially
    await extract_features_from_dataset_async(sem, train_input, output_train)
    await extract_features_from_dataset_async(sem, test_input, output_test)

if __name__ == "__main__":
    asyncio.run(main())
