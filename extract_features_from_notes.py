import asyncio
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
CONCURRENT_REQUESTS = 32  # Increased for Mac Studio performance

async def process_row_async(sem, agent, index, note, prior_readmission_indicator):
    """Processes a single row using the agent with concurrency protection."""
    async with sem:
        try:
            # The agent's orchestrate method is synchronous, 
            # so we run it in an executor to avoid blocking the event loop.
            loop = asyncio.get_running_loop()
            report = await loop.run_in_executor(
                None, agent.orchestrate, note, prior_readmission_indicator
            )
            
            if report:
                features = report.features
                return index, {
                    'llm_num_symptoms': len(features.symptoms),
                    'llm_num_medications': len(features.medications),
                    'llm_glucose_status': features.glucose_status,
                    'llm_has_meds': 1 if len(features.medications) > 0 else 0,
                    'llm_has_symptoms': 1 if len(features.symptoms) > 0 else 0,
                    'llm_age_group': features.age_group
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
        except (KeyError, ValueError, TypeError) as e:
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
    Processes a dataset concurrently using asyncio.
    """
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Loading dataset: {input_path}")
    df = pd.read_csv(input_path)

    # Initialize Agent Dependencies
    provider = OllamaProvider() 
    llm = LLMInterface(provider)
    retriever = RAGRetriever()
    retriever.load()
    agent = ClinicalOrchestratorAgent(retriever, llm)

    logger.info(f"Starting concurrent feature extraction for {len(df)} rows (Concurrency: {CONCURRENT_REQUESTS})...")

    # Prepare tasks
    tasks = []
    for index, row in df.iterrows():
        note = row['clinical_note']
        prior_readmission_indicator = False
        tasks.append(process_row_async(sem, agent, index, note, prior_readmission_indicator))

    # Dictionary to store results by index to maintain order
    results_dict = {}
    
    # Process tasks as they complete
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Rows"):
        index, feature_dict = await future
        results_dict[index] = feature_dict

    # Create DataFrame from results, ensuring we follow the original index order
    sorted_indices = sorted(results_dict.keys())
    new_features_lagged = [results_dict[idx] for idx in sorted_indices]
    features_df = pd.DataFrame(new_features_lagged)
    
    # Concatenate with original DataFrame
    final_df = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

    # Save the result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logger.success(f"Feature extraction complete. Saved to: {output_path}")

async def main():
    # Create the semaphore INSIDE the running event loop
    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)

    # Using paths from centralized config instead of raw strings
    train_input = config.OUTPUT_DATA_PATH
    test_input = config.TEST_DATA_PATH
    
    # Constructing output paths within processed_dir
    output_train = config.PROCESSED_DIR / "train_with_extracted_features.csv"
    output_test = config.PROCESSED_DIR / "test_with_extracted_features.csv"

    await extract_features_from_dataset_async(sem, train_input, output_train)
    await extract_features_from_dataset_async(sem, test_input, output_test)

if __name__ == "__main__":
    asyncio.run(main())